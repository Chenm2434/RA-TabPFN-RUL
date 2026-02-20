#!/usr/bin/env python3
"""Run model comparisons between RA-TabPFN and deep learning baselines.

Fast Access Version: Requires preprocessed data in data/ directory.
Download from GitHub Release and extract to data/ before running.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
try:
    import torch
    import torch.nn as nn
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency: torch. Install PyTorch (see requirements.txt) and rerun."
    ) from e
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from core.data_utils import extract_q_sequence, load_phase2_with_metadata
from models.cnn import CNN1DClassifier, CNN1DRegressor
from models.mlp import MLPClassifier, MLPRegressor
from models.transformer import TransformerClassifier, TransformerRegressor
from core.ra_tabpfn_cd_diff import RATabPFNCDDiff, load_cd_diff_features

def _stable_int_hash(text: str) -> int:
    digest = hashlib.md5(text.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _seed_everything(seed: int, determinism: str = "warn", tf32: bool = False) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(int(seed))
    np.random.seed(int(seed))

    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = bool(tf32)
    if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = bool(tf32)

    determinism = (determinism or "").lower().strip()
    if determinism == "strict":
        torch.use_deterministic_algorithms(True)
    elif determinism == "warn":
        torch.use_deterministic_algorithms(True, warn_only=True)
    elif determinism == "off":
        torch.use_deterministic_algorithms(False)
    else:
        raise ValueError("--determinism must be one of: strict, warn, off")


@dataclass
class RunResult:
    dataset: str
    model: str
    method: str
    k: Optional[int]
    feature_type: str
    eval_split: str
    input_dim: int
    n_train_samples: int
    n_test_samples: int
    rul_mae_before_k2o: Optional[float]
    risk_acc: Optional[float]
    train_time_risk_s: Optional[float]
    train_time_k2o_s: Optional[float]
    fit_time_s: Optional[float]
    predict_time_risk_total_s: Optional[float]
    predict_time_k2o_total_s: Optional[float]
    predict_time_total_s: Optional[float]
    predict_time_risk_per_sample_ms: Optional[float]
    predict_time_k2o_per_sample_ms: Optional[float]
    predict_time_per_sample_ms: Optional[float]
    predictions_csv: str
    status: str
    error: str


def _sanitize(name: str) -> str:
    return (
        name.replace(" ", "_")
        .replace("/", "-")
        .replace("\\", "-")
        .replace(":", "-")
        .replace("+", "plus")
    )


def _resolve_device(device_str: str) -> torch.device:
    device_str = (device_str or "").lower()
    if device_str in {"cuda", "gpu"}:
        if torch.cuda.is_available():
            return torch.device("cuda")
    return torch.device("cpu")


def _maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _extract_k2o_cycles(data: Dict) -> np.ndarray:
    if "k2o" in data:
        k2o_values = data["k2o"]
    elif "k2o_labels" in data:
        k2o_values = data["k2o_labels"]
    else:
        raise KeyError("Cannot find k2o or k2o_labels in cache data")

    if isinstance(k2o_values, torch.Tensor):
        k2o_values = k2o_values.detach().cpu().numpy()
    k2o_values = np.asarray(k2o_values, dtype=float)

    return k2o_values


def _compute_rul_from_k2o(k2o_cycles: np.ndarray, cycle_numbers: np.ndarray) -> np.ndarray:
    rul = np.asarray(k2o_cycles, dtype=float) - np.asarray(cycle_numbers, dtype=float)
    return np.maximum(rul, 0.0)


def _make_rul_scaler(train_raw: Dict) -> StandardScaler:
    cycle_numbers = np.asarray(train_raw["cycle_numbers"], dtype=float)
    k2o_cycles = _extract_k2o_cycles(train_raw)
    y_rul = k2o_cycles - cycle_numbers
    valid = (~np.isnan(y_rul)) & (cycle_numbers <= k2o_cycles)
    y_valid = np.asarray(y_rul[valid], dtype=float)
    if y_valid.size == 0:
        y_valid = np.asarray([0.0], dtype=float)

    scaler = StandardScaler()
    scaler.fit(y_valid.reshape(-1, 1))
    return scaler


def _subset_raw_by_indices(raw: Dict, indices: np.ndarray) -> Dict:
    idx = np.asarray(indices, dtype=int)
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        if isinstance(v, np.ndarray):
            out[k] = v[idx]
        elif isinstance(v, torch.Tensor):
            out[k] = v[idx]
        elif isinstance(v, list):
            out[k] = [v[i] for i in idx]
        else:
            out[k] = v
    return out


def _split_train_val_by_battery(train_raw: Dict, val_ratio_batteries: float, seed: int) -> Tuple[Dict, Dict, List[str]]:
    battery_ids = list(train_raw.get("battery_ids", []))
    unique_batteries = sorted(set(battery_ids))
    if len(unique_batteries) == 0:
        raise ValueError("train_raw has no battery_ids")

    n_val = max(1, int(round(len(unique_batteries) * float(val_ratio_batteries))))
    rng = np.random.RandomState(int(seed))
    val_batts = rng.choice(unique_batteries, size=min(n_val, len(unique_batteries)), replace=False).tolist()
    val_set = set(val_batts)

    mask_val = np.asarray([bid in val_set for bid in battery_ids], dtype=bool)
    val_idx = np.where(mask_val)[0]
    train_idx = np.where(~mask_val)[0]

    train_fit_raw = _subset_raw_by_indices(train_raw, train_idx)
    val_raw = _subset_raw_by_indices(train_raw, val_idx)
    return train_fit_raw, val_raw, val_batts


def _compute_metrics_from_pred_df(pred_df: pd.DataFrame) -> Tuple[Optional[float], Optional[float]]:
    if pred_df is None or pred_df.empty:
        return None, None

    acc = float(np.mean(pred_df["risk_true"].values == pred_df["risk_pred"].values) * 100.0)
    sub = pred_df[pred_df["cycle_number"] < pred_df["k2o_true"]]
    if sub.empty:
        return None, acc

    mae = float(np.mean(np.abs(sub["rul_pred"].values - sub["rul_true"].values)))
    return mae, acc


def _load_cd_feature_matrix(
    dataset_name: str,
    feature_type: str,
    battery_ids: List[str],
    cd_features_dir: Path,
    cd_df_cache: Dict[Tuple[str, str], pd.DataFrame],
) -> np.ndarray:
    key = (dataset_name.upper(), feature_type)
    if key not in cd_df_cache:
        cd_df_cache[key] = load_cd_diff_features(dataset_name.upper(), cd_features_dir, feature_type)

    cd_df = cd_df_cache[key]
    feature_cols = [c for c in cd_df.columns if c != "battery_id"]
    cd_map = cd_df.set_index("battery_id")[feature_cols]

    mat = cd_map.reindex(pd.Index(battery_ids)).fillna(0.0).to_numpy(dtype=float)
    mat = np.nan_to_num(mat, nan=0.0, posinf=0.0, neginf=0.0)
    return mat


def _build_features(
    train_raw: Dict,
    test_raw: Dict,
    dataset_name: str,
    variant: str,
    feature_type: str,
    cd_features_dir: Path,
    cd_df_cache: Dict[Tuple[str, str], pd.DataFrame],
) -> Tuple[np.ndarray, np.ndarray, int, str]:
    X_train_q = extract_q_sequence(train_raw["sequences"])
    X_test_q = extract_q_sequence(test_raw["sequences"])

    if variant == "qseq":
        ft_out = "none"
        if "|" in str(feature_type):
            ft_out = f"none|{str(feature_type).split('|', 1)[1]}"
        return X_train_q, X_test_q, int(X_train_q.shape[1]), ft_out

    if variant == "qseq_cddiff":
        cd_train = _load_cd_feature_matrix(dataset_name, feature_type, train_raw["battery_ids"], cd_features_dir, cd_df_cache)
        cd_test = _load_cd_feature_matrix(dataset_name, feature_type, test_raw["battery_ids"], cd_features_dir, cd_df_cache)
        X_train = np.hstack([X_train_q, cd_train])
        X_test = np.hstack([X_test_q, cd_test])
        return X_train, X_test, int(X_train.shape[1]), feature_type

    raise ValueError(f"Unknown feature variant: {variant}")


def _make_train_loader(X_t: torch.Tensor, y_t: torch.Tensor, batch_size: int, device: torch.device, seed: int) -> DataLoader:
    g = torch.Generator()
    g.manual_seed(int(seed))

    return DataLoader(
        TensorDataset(X_t, y_t),
        batch_size=batch_size,
        shuffle=True,
        generator=g,
        pin_memory=(device.type == "cuda"),
    )


def _train_torch_classifier(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
) -> nn.Module:
    """Train classifier with battery-level val set and early stopping based on validation loss."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Empty train/val split for classifier")

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

    loader = _make_train_loader(X_t, y_t, batch_size=batch_size, device=device, seed=seed)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for _epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_seen = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_X)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(batch_X.shape[0])
            n_seen += int(batch_X.shape[0])

        # Early stopping based on validation loss
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_loss = float(criterion(val_logits, y_val_t).item())
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def _train_torch_regressor(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    seed: int,
) -> nn.Module:
    """Train regressor with battery-level val set and early stopping based on validation loss."""
    model = model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    if len(X_train) == 0 or len(X_val) == 0:
        raise ValueError("Empty train/val split for regressor")

    X_t = torch.tensor(X_train, dtype=torch.float32)
    y_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32).reshape(-1, 1).to(device)

    loader = _make_train_loader(X_t, y_t, batch_size=batch_size, device=device, seed=seed)

    best_loss = float("inf")
    best_state = None
    patience_counter = 0

    for _epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_seen = 0

        for batch_X, batch_y in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            batch_y = batch_y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            pred = model(batch_X)
            loss = criterion(pred, batch_y)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * int(batch_X.shape[0])
            n_seen += int(batch_X.shape[0])

        # Early stopping based on validation loss
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = float(criterion(val_pred, y_val_t).item())
        
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    return model


def _predict_torch_classifier(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_t),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            logits = model(batch_X)
            pred = logits.argmax(dim=1).detach().cpu().numpy()
            preds.append(pred)

    return np.concatenate(preds, axis=0)


def _predict_torch_regressor(model: nn.Module, X: np.ndarray, device: torch.device, batch_size: int = 2048) -> np.ndarray:
    model.eval()
    X_t = torch.tensor(X, dtype=torch.float32)

    loader = DataLoader(
        TensorDataset(X_t),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    preds: List[np.ndarray] = []
    with torch.no_grad():
        for (batch_X,) in loader:
            batch_X = batch_X.to(device, non_blocking=True)
            out = model(batch_X)
            pred = out.detach().cpu().numpy().reshape(-1)
            preds.append(pred)

    return np.concatenate(preds, axis=0)


def _build_predictions_df(
    dataset: str,
    model: str,
    method: str,
    k: Optional[int],
    feature_type: str,
    battery_ids: List[str],
    cycle_numbers: np.ndarray,
    risk_true: np.ndarray,
    risk_pred: np.ndarray,
    k2o_true: np.ndarray,
    k2o_pred: np.ndarray,
    risk_proba: Optional[np.ndarray] = None,
    similarities_dict: Optional[Dict[str, List[float]]] = None,
    neighbor_stage_dist_dict: Optional[Dict[str, List[float]]] = None,
    similar_batteries_dict: Optional[Dict[str, List[str]]] = None,
) -> pd.DataFrame:
    rul_true = _compute_rul_from_k2o(k2o_true, cycle_numbers)
    rul_pred = _compute_rul_from_k2o(k2o_pred, cycle_numbers)

    df_dict = {
        "dataset": dataset,
        "model": model,
        "method": method,
        "k": k,
        "feature_type": feature_type,
        "battery_id": battery_ids,
        "cycle_number": cycle_numbers.astype(int),
        "risk_true": risk_true.astype(int),
        "risk_pred": risk_pred.astype(int),
        "k2o_true": k2o_true.astype(float),
        "k2o_pred": k2o_pred.astype(float),
        "rul_true": rul_true.astype(float),
        "rul_pred": rul_pred.astype(float),
    }

    if risk_proba is not None:
        df_dict["risk_proba_0"] = risk_proba[:, 0].astype(float)
        df_dict["risk_proba_1"] = risk_proba[:, 1].astype(float)

    if similarities_dict is not None:
        sim_values = []
        for bid in battery_ids:
            sims = similarities_dict.get(bid, [0.0])
            sim_values.append(float(np.mean(sims)) if sims else 0.0)
        df_dict["avg_similarity"] = sim_values

    if similar_batteries_dict is not None:
        neighbor_batteries_list = []
        for bid in battery_ids:
            neighbors = similar_batteries_dict.get(bid, [])
            neighbor_str = ";".join(neighbors) if neighbors else ""
            neighbor_batteries_list.append(neighbor_str)
        df_dict["neighbor_batteries"] = neighbor_batteries_list

    if neighbor_stage_dist_dict is not None:
        stage_0_20 = []
        stage_20_40 = []
        stage_40_60 = []
        stage_60_80 = []
        stage_80_100 = []
        for bid in battery_ids:
            dist = neighbor_stage_dist_dict.get(bid, [0.0, 0.0, 0.0, 0.0, 0.0])
            stage_0_20.append(dist[0] if len(dist) > 0 else 0.0)
            stage_20_40.append(dist[1] if len(dist) > 1 else 0.0)
            stage_40_60.append(dist[2] if len(dist) > 2 else 0.0)
            stage_60_80.append(dist[3] if len(dist) > 3 else 0.0)
            stage_80_100.append(dist[4] if len(dist) > 4 else 0.0)
        df_dict["neighbor_stage_0_20_pct"] = stage_0_20
        df_dict["neighbor_stage_20_40_pct"] = stage_20_40
        df_dict["neighbor_stage_40_60_pct"] = stage_40_60
        df_dict["neighbor_stage_60_80_pct"] = stage_60_80
        df_dict["neighbor_stage_80_100_pct"] = stage_80_100

    df = pd.DataFrame(df_dict)
    return df


def _run_ra_tabpfn(
    dataset: str,
    feature_type: str,
    k: int,
    train_raw: Dict,
    test_raw: Dict,
    cd_features_dir: Path,
    device: torch.device,
    base_seed: int,
    determinism: str,
    tf32: bool,
    eval_split: str,
) -> Tuple[pd.DataFrame, RunResult]:
    model_name = "RA-TabPFN"
    method = "RA-CDDiff-Retrieval"
    run_seed = int(base_seed) + _stable_int_hash(f"{dataset}|{model_name}|{method}|{feature_type}|k={k}|split={eval_split}")
    _seed_everything(run_seed, determinism=determinism, tf32=tf32)

    cycle_numbers = np.asarray(test_raw["cycle_numbers"], dtype=float)
    risk_true = np.asarray(test_raw["risk_labels"], dtype=int)
    k2o_true = _extract_k2o_cycles(test_raw)

    scaler = _make_rul_scaler(train_raw)

    error = ""
    status = "OK"

    fit_time_s: Optional[float] = None
    pred_time_risk_s: Optional[float] = None
    pred_time_k2o_s: Optional[float] = None
    pred_time_total_s: Optional[float] = None

    try:
        ra = RATabPFNCDDiff(
            k_neighbors=int(k),
            cd_features_dir=cd_features_dir,
            feature_type=feature_type,
            device=str(device),
        )

        _maybe_sync(device)
        t0 = time.perf_counter()
        ra.fit(train_raw, scaler, dataset)
        _maybe_sync(device)
        fit_time_s = time.perf_counter() - t0

        _maybe_sync(device)
        t_risk = time.perf_counter()
        preds_risk = ra.predict(test_raw, task="risk")
        _maybe_sync(device)
        pred_time_risk_s = time.perf_counter() - t_risk

        _maybe_sync(device)
        t_rul = time.perf_counter()
        preds_rul = ra.predict(test_raw, task="rul")
        _maybe_sync(device)
        pred_time_k2o_s = time.perf_counter() - t_rul

        pred_time_total_s = float(pred_time_risk_s) + float(pred_time_k2o_s)

        risk_pred = np.asarray(preds_risk["risk_pred"], dtype=int)
        risk_proba = preds_risk.get("risk_proba", None)
        similarities_dict = preds_risk.get("similarities", {})
        neighbor_stage_dist_dict = preds_risk.get("neighbor_stage_dist", {})
        similar_batteries_dict = preds_risk.get("similar_batteries", {})

        rul_pred_raw = np.asarray(preds_rul["rul_pred"], dtype=float)
        rul_pred = np.maximum(rul_pred_raw, 0.0)
        k2o_pred = cycle_numbers + rul_pred

        pred_df = _build_predictions_df(
            dataset=dataset,
            model=model_name,
            method=method,
            k=int(k),
            feature_type=feature_type,
            battery_ids=list(test_raw["battery_ids"]),
            cycle_numbers=cycle_numbers,
            risk_true=risk_true,
            risk_pred=risk_pred,
            k2o_true=k2o_true,
            k2o_pred=k2o_pred,
            risk_proba=risk_proba,
            similarities_dict=similarities_dict,
            neighbor_stage_dist_dict=neighbor_stage_dist_dict,
            similar_batteries_dict=similar_batteries_dict,
        )

        pred_df["eval_split"] = str(eval_split)

        rul_mae, risk_acc = _compute_metrics_from_pred_df(pred_df)

        n_test = int(len(pred_df))
        risk_ms = (pred_time_risk_s / max(n_test, 1)) * 1000.0 if pred_time_risk_s is not None else None
        k2o_ms = (pred_time_k2o_s / max(n_test, 1)) * 1000.0 if pred_time_k2o_s is not None else None
        per_sample_ms = (pred_time_total_s / max(n_test, 1)) * 1000.0 if pred_time_total_s is not None else None

        run = RunResult(
            dataset=dataset,
            model=model_name,
            method=method,
            k=int(k),
            feature_type=feature_type,
            eval_split=str(eval_split),
            input_dim=140,
            n_train_samples=int(len(train_raw["battery_ids"])),
            n_test_samples=n_test,
            rul_mae_before_k2o=rul_mae,
            risk_acc=risk_acc,
            train_time_risk_s=None,
            train_time_k2o_s=None,
            fit_time_s=fit_time_s,
            predict_time_risk_total_s=pred_time_risk_s,
            predict_time_k2o_total_s=pred_time_k2o_s,
            predict_time_total_s=pred_time_total_s,
            predict_time_risk_per_sample_ms=risk_ms,
            predict_time_k2o_per_sample_ms=k2o_ms,
            predict_time_per_sample_ms=per_sample_ms,
            predictions_csv="",
            status=status,
            error=error,
        )

        return pred_df, run

    except Exception as e:
        status = "FAILED"
        error = str(e)

        empty_df = pd.DataFrame(
            columns=[
                "dataset", "model", "method", "k", "feature_type",
                "battery_id", "cycle_number", "risk_true", "risk_pred",
                "k2o_true", "k2o_pred", "rul_true", "rul_pred",
            ]
        )

        run = RunResult(
            dataset=dataset,
            model=model_name,
            method=method,
            k=int(k),
            feature_type=feature_type,
            eval_split=str(eval_split),
            input_dim=140,
            n_train_samples=int(len(train_raw.get("battery_ids", []))),
            n_test_samples=int(len(test_raw.get("battery_ids", []))),
            rul_mae_before_k2o=None,
            risk_acc=None,
            train_time_risk_s=None,
            train_time_k2o_s=None,
            fit_time_s=fit_time_s,
            predict_time_risk_total_s=pred_time_risk_s,
            predict_time_k2o_total_s=pred_time_k2o_s,
            predict_time_total_s=pred_time_total_s,
            predict_time_risk_per_sample_ms=None,
            predict_time_k2o_per_sample_ms=None,
            predict_time_per_sample_ms=None,
            predictions_csv="",
            status=status,
            error=error,
        )

        return empty_df, run


def _run_ra_tabpfn_random(
    dataset: str,
    feature_type: str,
    k: int,
    train_raw: Dict,
    test_raw: Dict,
    cd_features_dir: Path,
    device: torch.device,
    base_seed: int,
    determinism: str,
    tf32: bool,
    eval_split: str,
) -> Tuple[pd.DataFrame, RunResult]:
    """RA-TabPFN with RANDOM retrieval (baseline to prove similarity-based retrieval value)."""
    model_name = "RA-TabPFN"
    method = "RA-Random-Retrieval"
    run_seed = int(base_seed) + _stable_int_hash(f"{dataset}|{model_name}|{method}|{feature_type}|k={k}|split={eval_split}")
    _seed_everything(run_seed, determinism=determinism, tf32=tf32)

    cycle_numbers = np.asarray(test_raw["cycle_numbers"], dtype=float)
    risk_true = np.asarray(test_raw["risk_labels"], dtype=int)
    k2o_true = _extract_k2o_cycles(test_raw)

    scaler = _make_rul_scaler(train_raw)

    error = ""
    status = "OK"

    fit_time_s: Optional[float] = None
    pred_time_risk_s: Optional[float] = None
    pred_time_k2o_s: Optional[float] = None
    pred_time_total_s: Optional[float] = None

    try:
        from core.ra_tabpfn_cd_diff import RATabPFNCDDiff
        from core.data_utils import extract_q_sequence
        from tabpfn import TabPFNClassifier
        from tabpfn.regressor import TabPFNRegressor

        ra = RATabPFNCDDiff(
            k_neighbors=int(k),
            cd_features_dir=cd_features_dir,
            feature_type=feature_type,
            device=str(device),
        )

        _maybe_sync(device)
        t0 = time.perf_counter()
        ra.fit(train_raw, scaler, dataset)
        _maybe_sync(device)
        fit_time_s = time.perf_counter() - t0

        ra.test_X = extract_q_sequence(test_raw['sequences'])

        train_battery_ids = list(ra.train_cd_features.keys())
        rng = np.random.RandomState(run_seed)

        test_battery_ids = test_raw["battery_ids"]
        unique_test_batteries = sorted(set(test_battery_ids))

        n_test = len(test_battery_ids)
        risk_pred = np.zeros(n_test, dtype=int)
        k2o_pred = np.zeros(n_test, dtype=float)
        random_batteries_dict = {}

        tabpfn_clf = TabPFNClassifier(
            device=ra.device,
            ignore_pretraining_limits=True,
            n_estimators=1,
            memory_saving_mode=False,
        )
        tabpfn_reg = TabPFNRegressor(
            device=ra.device,
            ignore_pretraining_limits=True,
            n_estimators=1,
            memory_saving_mode=False,
        )

        _maybe_sync(device)
        t_risk = time.perf_counter()

        for test_battery_id in unique_test_batteries:
            top_k_batteries = rng.choice(train_battery_ids, size=min(k, len(train_battery_ids)), replace=False).tolist()
            random_batteries_dict[test_battery_id] = top_k_batteries

            train_indices_k = []
            for battery_id in top_k_batteries:
                battery_mask = np.array([bid == battery_id for bid in ra.train_battery_ids])
                train_indices_k.extend(np.where(battery_mask)[0])

            test_battery_mask = np.array([bid == test_battery_id for bid in test_battery_ids])
            test_battery_indices = np.where(test_battery_mask)[0]

            X_train_k = ra.train_X[train_indices_k]
            y_risk_train_k = ra.train_risk_labels[train_indices_k]
            y_rul_train_k = ra.train_rul_labels[train_indices_k]
            X_test_battery = ra.test_X[test_battery_indices]

            try:
                tabpfn_clf.fit(X_train_k, y_risk_train_k)
                risk_pred_battery = tabpfn_clf.predict(X_test_battery)
                risk_pred[test_battery_indices] = risk_pred_battery
            except Exception as e_risk:
                print(f"    [WARN] Risk prediction failed for {test_battery_id}: {e_risk}")
                risk_pred[test_battery_indices] = 0

            valid_mask = ~np.isnan(y_rul_train_k)
            X_train_k_rul = X_train_k[valid_mask]
            y_rul_train_k_valid = y_rul_train_k[valid_mask]

            if len(X_train_k_rul) > 0:
                try:
                    y_scaled = scaler.transform(y_rul_train_k_valid.reshape(-1, 1)).flatten()
                    tabpfn_reg.fit(X_train_k_rul, y_scaled)
                    rul_pred_scaled = tabpfn_reg.predict(X_test_battery)
                    rul_pred_raw = scaler.inverse_transform(rul_pred_scaled.reshape(-1, 1)).flatten()
                    rul_pred_battery = np.maximum(rul_pred_raw, 0.0)
                    k2o_pred[test_battery_indices] = cycle_numbers[test_battery_indices] + rul_pred_battery
                except Exception as e_rul:
                    print(f"    [WARN] RUL prediction failed for {test_battery_id}: {e_rul}")
                    k2o_pred[test_battery_indices] = cycle_numbers[test_battery_indices]

        _maybe_sync(device)
        pred_time_risk_s = time.perf_counter() - t_risk
        pred_time_k2o_s = pred_time_risk_s
        pred_time_total_s = pred_time_risk_s

        pred_df = _build_predictions_df(
            dataset=dataset,
            model=model_name,
            method=method,
            k=int(k),
            feature_type=feature_type,
            battery_ids=list(test_battery_ids),
            cycle_numbers=cycle_numbers,
            risk_true=risk_true,
            risk_pred=risk_pred,
            k2o_true=k2o_true,
            k2o_pred=k2o_pred,
            similar_batteries_dict=random_batteries_dict,
        )

        pred_df["eval_split"] = str(eval_split)

        rul_mae, risk_acc = _compute_metrics_from_pred_df(pred_df)

        n_test = int(len(pred_df))
        risk_ms = (pred_time_risk_s / max(n_test, 1)) * 1000.0 if pred_time_risk_s is not None else None
        k2o_ms = (pred_time_k2o_s / max(n_test, 1)) * 1000.0 if pred_time_k2o_s is not None else None
        per_sample_ms = (pred_time_total_s / max(n_test, 1)) * 1000.0 if pred_time_total_s is not None else None

        run = RunResult(
            dataset=dataset,
            model=model_name,
            method=method,
            k=int(k),
            feature_type=feature_type,
            eval_split=str(eval_split),
            input_dim=140,
            n_train_samples=int(len(train_raw["battery_ids"])),
            n_test_samples=n_test,
            rul_mae_before_k2o=rul_mae,
            risk_acc=risk_acc,
            train_time_risk_s=None,
            train_time_k2o_s=None,
            fit_time_s=fit_time_s,
            predict_time_risk_total_s=pred_time_risk_s,
            predict_time_k2o_total_s=pred_time_k2o_s,
            predict_time_total_s=pred_time_total_s,
            predict_time_risk_per_sample_ms=risk_ms,
            predict_time_k2o_per_sample_ms=k2o_ms,
            predict_time_per_sample_ms=per_sample_ms,
            predictions_csv="",
            status=status,
            error=error,
        )

        return pred_df, run

    except Exception as e:
        import traceback
        status = "FAILED"
        error = str(e)
        print(f"    [ERROR] Random Retrieval failed: {error}")
        print(f"    [TRACEBACK] {traceback.format_exc()}")

        empty_df = pd.DataFrame(
            columns=[
                "dataset", "model", "method", "k", "feature_type",
                "battery_id", "cycle_number", "risk_true", "risk_pred",
                "k2o_true", "k2o_pred", "rul_true", "rul_pred",
            ]
        )

        run = RunResult(
            dataset=dataset,
            model=model_name,
            method=method,
            k=int(k),
            feature_type=feature_type,
            eval_split=str(eval_split),
            input_dim=140,
            n_train_samples=int(len(train_raw.get("battery_ids", []))),
            n_test_samples=int(len(test_raw.get("battery_ids", []))),
            rul_mae_before_k2o=None,
            risk_acc=None,
            train_time_risk_s=None,
            train_time_k2o_s=None,
            fit_time_s=fit_time_s,
            predict_time_risk_total_s=pred_time_risk_s,
            predict_time_k2o_total_s=pred_time_k2o_s,
            predict_time_total_s=pred_time_total_s,
            predict_time_risk_per_sample_ms=None,
            predict_time_k2o_per_sample_ms=None,
            predict_time_per_sample_ms=None,
            predictions_csv="",
            status=status,
            error=error,
        )

        return empty_df, run


def _create_deep_models(model_name: str, input_dim: int) -> Tuple[nn.Module, nn.Module]:
    model_name = model_name.upper()

    if model_name == "MLP":
        return MLPClassifier(input_dim=input_dim), MLPRegressor(input_dim=input_dim)

    if model_name == "CNN":
        return CNN1DClassifier(input_dim=input_dim), CNN1DRegressor(input_dim=input_dim)

    if model_name == "TRANSFORMER":
        return TransformerClassifier(input_dim=input_dim), TransformerRegressor(input_dim=input_dim)

    raise ValueError(f"Unknown deep model: {model_name}")


def _run_deep_model(
    dataset: str,
    model_name: str,
    variant: str,
    feature_type: str,
    train_raw: Dict,
    val_raw: Dict,
    test_raw: Dict,
    cd_features_dir: Path,
    cd_df_cache: Dict[Tuple[str, str], pd.DataFrame],
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    base_seed: int,
    determinism: str,
    tf32: bool,
    ckpt_dir: Optional[Path],
    save_checkpoints: bool,
    use_saved_checkpoints: bool,
) -> Tuple[pd.DataFrame, RunResult]:
    method = "DL-Qseq" if variant == "qseq" else "DL-Qseq+CDDiff"
    run_seed = int(base_seed) + _stable_int_hash(f"{dataset}|{model_name}|{method}|{feature_type}")
    _seed_everything(run_seed, determinism=determinism, tf32=tf32)

    X_train, X_test, input_dim, feature_type_out = _build_features(
        train_raw=train_raw,
        test_raw=test_raw,
        dataset_name=dataset,
        variant=variant,
        feature_type=feature_type,
        cd_features_dir=cd_features_dir,
        cd_df_cache=cd_df_cache,
    )

    X_val, _X_dummy, _in_dim_dummy, _ft_dummy = _build_features(
        train_raw=val_raw,
        test_raw=test_raw,
        dataset_name=dataset,
        variant=variant,
        feature_type=feature_type,
        cd_features_dir=cd_features_dir,
        cd_df_cache=cd_df_cache,
    )

    risk_true = np.asarray(test_raw["risk_labels"], dtype=int)
    k2o_true = _extract_k2o_cycles(test_raw)
    cycle_numbers = np.asarray(test_raw["cycle_numbers"], dtype=float)

    y_risk_train = np.asarray(train_raw["risk_labels"], dtype=int)
    y_risk_val = np.asarray(val_raw["risk_labels"], dtype=int)
    y_k2o_train = _extract_k2o_cycles(train_raw)
    y_k2o_val = _extract_k2o_cycles(val_raw)
    train_cycle_numbers = np.asarray(train_raw["cycle_numbers"], dtype=float)
    val_cycle_numbers = np.asarray(val_raw["cycle_numbers"], dtype=float)
    y_rul_train = y_k2o_train - train_cycle_numbers
    y_rul_val = y_k2o_val - val_cycle_numbers

    # Plan A+: RUL training only uses pre-K2O data (filter post-K2O samples)
    valid_rul_mask = (~np.isnan(y_rul_train)) & (train_cycle_numbers <= y_k2o_train)
    X_train_rul = X_train[valid_rul_mask]
    y_rul_train_valid = y_rul_train[valid_rul_mask]

    valid_rul_mask_val = (~np.isnan(y_rul_val)) & (val_cycle_numbers <= y_k2o_val)
    X_val_rul = X_val[valid_rul_mask_val]
    y_rul_val_valid = y_rul_val[valid_rul_mask_val]

    if len(X_train_rul) == 0 or len(X_val_rul) == 0:
        raise ValueError("Empty train/val split for regressor")

    rul_scaler = _make_rul_scaler(train_raw)
    y_rul_train_scaled = rul_scaler.transform(np.asarray(y_rul_train_valid, dtype=float).reshape(-1, 1)).flatten()
    y_rul_val_scaled = rul_scaler.transform(np.asarray(y_rul_val_valid, dtype=float).reshape(-1, 1)).flatten()

    error = ""
    status = "OK"

    try:
        ckpt_stem = _sanitize(f"{dataset}_{model_name}_{method}_{feature_type_out}_{run_seed}")
        risk_ckpt_path = (ckpt_dir / f"{ckpt_stem}_risk.pt") if ckpt_dir is not None else None
        rul_ckpt_path = (ckpt_dir / f"{ckpt_stem}_rul.pt") if ckpt_dir is not None else None

        ckpt_available = (
            bool(use_saved_checkpoints)
            and risk_ckpt_path is not None
            and rul_ckpt_path is not None
            and risk_ckpt_path.exists()
            and rul_ckpt_path.exists()
        )

        clf, reg = _create_deep_models(model_name, input_dim)

        train_time_risk_s: Optional[float]
        train_time_k2o_s: Optional[float]

        if ckpt_available:
            clf.load_state_dict(torch.load(risk_ckpt_path, map_location="cpu"))
            reg.load_state_dict(torch.load(rul_ckpt_path, map_location="cpu"))
            clf = clf.to(device)
            reg = reg.to(device)
            train_time_risk_s = 0.0
            train_time_k2o_s = 0.0
        else:
            _maybe_sync(device)
            t0 = time.perf_counter()
            clf = _train_torch_classifier(
                model=clf,
                X_train=X_train,
                y_train=y_risk_train,
                X_val=X_val,
                y_val=y_risk_val,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                seed=run_seed + 1000,
            )
            _maybe_sync(device)
            train_time_risk_s = time.perf_counter() - t0
            _maybe_sync(device)
            t1 = time.perf_counter()
            reg = _train_torch_regressor(
                model=reg,
                X_train=X_train_rul,
                y_train=y_rul_train_scaled,
                X_val=X_val_rul,
                y_val=y_rul_val_scaled,
                device=device,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                patience=patience,
                seed=run_seed + 2000,
            )
            _maybe_sync(device)
            train_time_k2o_s = time.perf_counter() - t1

            if save_checkpoints and risk_ckpt_path is not None and rul_ckpt_path is not None:
                ckpt_dir.mkdir(parents=True, exist_ok=True)
                torch.save({k: v.detach().cpu() for k, v in clf.state_dict().items()}, risk_ckpt_path)
                torch.save({k: v.detach().cpu() for k, v in reg.state_dict().items()}, rul_ckpt_path)

        _maybe_sync(device)
        t2 = time.perf_counter()
        risk_pred = _predict_torch_classifier(clf, X_test, device=device)
        _maybe_sync(device)
        pred_time_risk_s = time.perf_counter() - t2

        _maybe_sync(device)
        t3 = time.perf_counter()
        rul_pred_raw = _predict_torch_regressor(reg, X_test, device=device)
        _maybe_sync(device)
        pred_time_k2o_s = time.perf_counter() - t3

        rul_pred_unscaled = rul_scaler.inverse_transform(np.asarray(rul_pred_raw, dtype=float).reshape(-1, 1)).flatten()
        rul_pred = np.maximum(rul_pred_unscaled, 0.0)
        k2o_pred = cycle_numbers + rul_pred

        pred_df = _build_predictions_df(
            dataset=dataset,
            model=model_name,
            method=method,
            k=None,
            feature_type=feature_type_out,
            battery_ids=list(test_raw["battery_ids"]),
            cycle_numbers=cycle_numbers,
            risk_true=risk_true,
            risk_pred=risk_pred,
            k2o_true=k2o_true,
            k2o_pred=k2o_pred,
        )

        rul_mae, risk_acc = _compute_metrics_from_pred_df(pred_df)

        n_test = int(len(pred_df))
        risk_ms = (pred_time_risk_s / max(n_test, 1)) * 1000.0
        k2o_ms = (pred_time_k2o_s / max(n_test, 1)) * 1000.0
        total_pred_s = pred_time_risk_s + pred_time_k2o_s
        total_ms = (total_pred_s / max(n_test, 1)) * 1000.0

        run = RunResult(
            dataset=dataset,
            model=model_name,
            method=method,
            k=None,
            feature_type=feature_type_out,
            eval_split="test",
            input_dim=input_dim,
            n_train_samples=int(len(train_raw["battery_ids"])),
            n_test_samples=n_test,
            rul_mae_before_k2o=rul_mae,
            risk_acc=risk_acc,
            train_time_risk_s=train_time_risk_s,
            train_time_k2o_s=train_time_k2o_s,
            fit_time_s=None,
            predict_time_risk_total_s=pred_time_risk_s,
            predict_time_k2o_total_s=pred_time_k2o_s,
            predict_time_total_s=total_pred_s,
            predict_time_risk_per_sample_ms=risk_ms,
            predict_time_k2o_per_sample_ms=k2o_ms,
            predict_time_per_sample_ms=total_ms,
            predictions_csv="",
            status=status,
            error=error,
        )

        return pred_df, run

    except Exception as e:
        status = "FAILED"
        error = str(e)

        empty_df = pd.DataFrame(
            columns=[
                "dataset", "model", "method", "k", "feature_type",
                "battery_id", "cycle_number", "risk_true", "risk_pred",
                "k2o_true", "k2o_pred", "rul_true", "rul_pred",
            ]
        )

        run = RunResult(
            dataset=dataset,
            model=model_name,
            method=method,
            k=None,
            feature_type=feature_type_out if "feature_type_out" in locals() else feature_type,
            eval_split="test",
            input_dim=input_dim if "input_dim" in locals() else 0,
            n_train_samples=int(len(train_raw.get("battery_ids", []))),
            n_test_samples=int(len(test_raw.get("battery_ids", []))),
            rul_mae_before_k2o=None,
            risk_acc=None,
            train_time_risk_s=None,
            train_time_k2o_s=None,
            fit_time_s=None,
            predict_time_risk_total_s=None,
            predict_time_k2o_total_s=None,
            predict_time_total_s=None,
            predict_time_risk_per_sample_ms=None,
            predict_time_k2o_per_sample_ms=None,
            predict_time_per_sample_ms=None,
            predictions_csv="",
            status=status,
            error=error,
        )

        return empty_df, run


def _save_run(run_dict: Dict, runs_summary_path: Path) -> None:
    """Append a single run result to runs_summary.csv."""
    run_df = pd.DataFrame([run_dict])
    if runs_summary_path.exists():
        run_df.to_csv(runs_summary_path, mode="a", index=False, header=False)
    else:
        run_df.to_csv(runs_summary_path, mode="w", index=False, header=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="RA-TabPFN vs Deep Learning Models Comparison")

    parser.add_argument("--datasets", nargs="+", default=["XJTU", "MIT"],
                        help="Datasets to run (default: XJTU MIT)")
    parser.add_argument("--models", nargs="+", default=["RA-TabPFN", "MLP", "CNN", "Transformer"],
                        help="Models to run (default: all)")
    parser.add_argument("--ks", nargs="+", type=int, default=[3, 5, 7, 10, 15],
                        help="K values for RA-TabPFN (default: 3 5 7 10 15)")
    parser.add_argument("--device", default="cuda",
                        help="Device: cuda/cpu (default: cuda)")
    parser.add_argument("--val_only", action="store_true",
                        help="RA-TabPFN: only run validation, skip best-k selection and test")
    parser.add_argument("--test_only", action="store_true",
                        help="RA-TabPFN: skip validation, read val results to select best k, then run test")

    args = parser.parse_args()

    if args.val_only and args.test_only:
        parser.error("--val_only and --test_only are mutually exclusive")

    # Fixed parameters
    seed = 42
    determinism = "warn"
    tf32 = False
    save_checkpoints = True
    use_saved_checkpoints = False
    val_battery_ratio = 0.10
    val_split_seed = 42
    epochs = 100
    batch_size = 256
    lr = 1e-3
    patience = 10

    _seed_everything(seed, determinism=determinism, tf32=tf32)
    device = _resolve_device(args.device)

    from config.paths import CACHE_UNIFIED_DIR, CDFEATURE_DIR, COMPARISON_RESULTS_DIR, CHECKPOINTS_DIR

    cache_dir = Path(CACHE_UNIFIED_DIR)
    cd_features_dir = Path(CDFEATURE_DIR)
    out_dir = Path(COMPARISON_RESULTS_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_dir = Path(CHECKPOINTS_DIR) if save_checkpoints or use_saved_checkpoints else None

    val_pred_path = out_dir / "predictions_val.csv"
    test_pred_path = out_dir / "predictions_test.csv"
    runs_summary_path = out_dir / "runs_summary.csv"
    best_k_json_path = out_dir / "best_k_by_val.json"

    existing_runs = set()

    def _make_run_key(dataset, model, method, k, feature_type):
        k_str = "nan" if (pd.isna(k) or str(k).lower() == 'nan') else str(k)
        return (str(dataset), str(model), str(method), k_str, str(feature_type))

    if runs_summary_path.exists():
        try:
            existing_summary = pd.read_csv(runs_summary_path)
            for _, row in existing_summary.iterrows():
                key = _make_run_key(
                    row.get('dataset', ''), row.get('model', ''),
                    row.get('method', ''), row.get('k', ''),
                    row.get('feature_type', '')
                )
                existing_runs.add(key)
            print(f"[INFO] Loaded {len(existing_runs)} existing runs, will skip them")
        except Exception as e:
            print(f"[WARN] Failed to load runs_summary.csv: {e}")
            existing_runs = set()
    else:
        print(f"[INFO] No existing runs found, will run all experiments")

    run_rows: List[Dict] = []
    cd_df_cache: Dict[Tuple[str, str], pd.DataFrame] = {}
    best_k_by_dataset: Dict[str, int] = {}

    for dataset in [d.upper() for d in args.datasets]:
        print(f"\n{'='*80}")
        print(f"Dataset: {dataset}")
        print(f"{'='*80}")

        train_cache = cache_dir / f"{dataset}_train.pt"
        test_cache = cache_dir / f"{dataset}_test.pt"

        train_raw_full = load_phase2_with_metadata(train_cache)
        test_raw = load_phase2_with_metadata(test_cache)

        split_seed = int(val_split_seed) + _stable_int_hash(f"valsplit|{dataset}")
        train_raw, val_raw, val_batts = _split_train_val_by_battery(
            train_raw_full,
            val_ratio_batteries=float(val_battery_ratio),
            seed=split_seed,
        )
        print(f"  [VAL] Battery-level split: {len(val_batts)} / {len(set(train_raw_full.get('battery_ids', [])))} batteries")

        feature_type = "type1|yStd"

        if any(m.lower() in {"ra-tabpfn", "ra", "ratabpfn"} for m in args.models):

            # ---- Phase 1: Validation (skip if --test_only) ----
            val_mae_by_k: Dict[int, float] = {}
            if not args.test_only:
                print("\n  [VAL] Selecting best k by validation (CDDiff only)...")

                for k in args.ks:
                    print(f"  -> [VAL] RA-CDDiff k={k}...", end=" ", flush=True)
                    pred_df, run = _run_ra_tabpfn(
                        dataset=dataset,
                        feature_type=feature_type,
                        k=int(k),
                        train_raw=train_raw,
                        test_raw=val_raw,
                        cd_features_dir=cd_features_dir,
                        device=device,
                        base_seed=seed,
                        determinism=determinism,
                        tf32=tf32,
                        eval_split="val",
                    )

                    if run.status == "OK":
                        pred_df.to_csv(val_pred_path, mode="a", index=False, header=not val_pred_path.exists())
                        if run.rul_mae_before_k2o is not None:
                            val_mae_by_k[int(k)] = float(run.rul_mae_before_k2o)

                    run_dict = asdict(run)
                    _save_run(run_dict, runs_summary_path)
                    run_rows.append(run_dict)
                    print(f"Done ({run.status})")

                print("\n  [VAL] Running Random baseline on validation...")
                for k in args.ks:
                    print(f"  -> [VAL] RA-Random k={k}...", end=" ", flush=True)
                    pred_df, run = _run_ra_tabpfn_random(
                        dataset=dataset,
                        feature_type=feature_type,
                        k=int(k),
                        train_raw=train_raw,
                        test_raw=val_raw,
                        cd_features_dir=cd_features_dir,
                        device=device,
                        base_seed=seed,
                        determinism=determinism,
                        tf32=tf32,
                        eval_split="val",
                    )

                    if run.status == "OK":
                        pred_df.to_csv(val_pred_path, mode="a", index=False, header=not val_pred_path.exists())

                    run_dict = asdict(run)
                    _save_run(run_dict, runs_summary_path)
                    run_rows.append(run_dict)
                    print(f"Done ({run.status})")

            # ---- Phase 2: Best-k selection + Test (skip if --val_only) ----
            if args.val_only:
                if val_mae_by_k:
                    print(f"\n  [VAL_ONLY] Val MAE summary for {dataset}:")
                    for kk, mae in sorted(val_mae_by_k.items()):
                        print(f"    k={kk}: val MAE = {mae:.4f}")
                    tentative_best = int(min(val_mae_by_k.items(), key=lambda kv: kv[1])[0])
                    print(f"    (Tentative best from this batch: k={tentative_best})")
                print(f"  [VAL_ONLY] Skipping best-k selection and test evaluation for {dataset}")
            else:
                if args.test_only:
                    print(f"\n  [TEST_ONLY] Reading val results from {runs_summary_path}...")
                    if not runs_summary_path.exists():
                        raise RuntimeError(f"runs_summary.csv not found. Run --val_only first.")
                    existing_summary = pd.read_csv(runs_summary_path)
                    val_cddiff = existing_summary[
                        (existing_summary["dataset"] == dataset)
                        & (existing_summary["method"] == "RA-CDDiff-Retrieval")
                        & (existing_summary["eval_split"] == "val")
                        & (existing_summary["status"] == "OK")
                        & (existing_summary["rul_mae_before_k2o"].notna())
                    ]
                    if val_cddiff.empty:
                        raise RuntimeError(f"No val results for dataset={dataset}. Run --val_only first.")
                    val_mae_all: Dict[int, float] = {}
                    for _, row in val_cddiff.iterrows():
                        val_mae_all[int(row["k"])] = float(row["rul_mae_before_k2o"])
                    print(f"  [TEST_ONLY] Found val results for k = {sorted(val_mae_all.keys())}")
                    for kk, mae in sorted(val_mae_all.items()):
                        print(f"    k={kk}: val MAE = {mae:.4f}")
                    best_k = int(min(val_mae_all.items(), key=lambda kv: kv[1])[0])
                    best_k_by_dataset[str(dataset)] = best_k
                    print(f"  [TEST_ONLY] Best k for {dataset}: k={best_k}")
                else:
                    if not val_mae_by_k:
                        raise RuntimeError(f"No valid val MAE to select best k for dataset={dataset}")
                    best_k = int(min(val_mae_by_k.items(), key=lambda kv: kv[1])[0])
                    best_k_by_dataset[str(dataset)] = best_k
                    print(f"  [VAL] Best k selected for {dataset}: k={best_k} (min val MAE)")

                print("\n  [TEST] Evaluating on test once using best k...")
                for method_name in ["RA-CDDiff-Retrieval", "RA-Random-Retrieval"]:
                    if method_name == "RA-CDDiff-Retrieval":
                        print(f"  -> [TEST] RA-CDDiff k={best_k}...", end=" ", flush=True)
                        pred_df, run = _run_ra_tabpfn(
                            dataset=dataset,
                            feature_type=feature_type,
                            k=int(best_k),
                            train_raw=train_raw_full,
                            test_raw=test_raw,
                            cd_features_dir=cd_features_dir,
                            device=device,
                            base_seed=seed,
                            determinism=determinism,
                            tf32=tf32,
                            eval_split="test",
                        )
                    else:
                        print(f"  -> [TEST] RA-Random k={best_k}...", end=" ", flush=True)
                        pred_df, run = _run_ra_tabpfn_random(
                            dataset=dataset,
                            feature_type=feature_type,
                            k=int(best_k),
                            train_raw=train_raw_full,
                            test_raw=test_raw,
                            cd_features_dir=cd_features_dir,
                            device=device,
                            base_seed=seed,
                            determinism=determinism,
                            tf32=tf32,
                            eval_split="test",
                        )

                    if run.status == "OK":
                        pred_df.to_csv(test_pred_path, mode="a", index=False, header=not test_pred_path.exists())

                    run_dict = asdict(run)
                    _save_run(run_dict, runs_summary_path)
                    run_rows.append(run_dict)
                    print(f"Done ({run.status})")

        deep_models = [m for m in args.models if m.lower() not in {"ra-tabpfn", "ra", "ratabpfn"}]
        for model_name in deep_models:
            run_key_qseq = _make_run_key(dataset, model_name.upper(), "DL-Qseq", None, "none")
            if run_key_qseq not in existing_runs:
                print(f"  -> {model_name} (Qseq)...", end=" ", flush=True)
                pred_df, run = _run_deep_model(
                    dataset=dataset,
                    model_name=model_name,
                    variant="qseq",
                    feature_type="type1",
                    train_raw=train_raw,
                    val_raw=val_raw,
                    test_raw=test_raw,
                    cd_features_dir=cd_features_dir,
                    cd_df_cache=cd_df_cache,
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    patience=patience,
                    base_seed=seed,
                    determinism=determinism,
                    tf32=tf32,
                    ckpt_dir=ckpt_dir,
                    save_checkpoints=save_checkpoints,
                    use_saved_checkpoints=use_saved_checkpoints,
                )

                file_name = _sanitize(f"predictions_{dataset}_{model_name}_DL-Qseq_none.csv")
                pred_path = out_dir / file_name

                if run.status == "OK":
                    pred_df.to_csv(pred_path, index=False)
                    pred_df.to_csv(test_pred_path, mode="a", index=False, header=not test_pred_path.exists())

                run.predictions_csv = pred_path.name
                run_dict = asdict(run)
                _save_run(run_dict, runs_summary_path)
                run_rows.append(run_dict)
                print(f"Done ({run.status})")
            else:
                print(f"  -> {model_name} (Qseq)... SKIP (already exists)")

            run_key_cddiff = _make_run_key(dataset, model_name.upper(), "DL-Qseq+CDDiff", None, feature_type)
            if run_key_cddiff not in existing_runs:
                print(f"  -> {model_name} (Qseq+CDDiff)...", end=" ", flush=True)
                pred_df, run = _run_deep_model(
                    dataset=dataset,
                    model_name=model_name,
                    variant="qseq_cddiff",
                    feature_type=feature_type,
                    train_raw=train_raw,
                    val_raw=val_raw,
                    test_raw=test_raw,
                    cd_features_dir=cd_features_dir,
                    cd_df_cache=cd_df_cache,
                    device=device,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    patience=patience,
                    base_seed=seed,
                    determinism=determinism,
                    tf32=tf32,
                    ckpt_dir=ckpt_dir,
                    save_checkpoints=save_checkpoints,
                    use_saved_checkpoints=use_saved_checkpoints,
                )

                file_name = _sanitize(f"predictions_{dataset}_{model_name}_DL-QseqplusCDDiff_{feature_type}.csv")
                pred_path = out_dir / file_name

                if run.status == "OK":
                    pred_df.to_csv(pred_path, index=False)
                    pred_df.to_csv(test_pred_path, mode="a", index=False, header=not test_pred_path.exists())

                run.predictions_csv = pred_path.name
                run_dict = asdict(run)
                _save_run(run_dict, runs_summary_path)
                run_rows.append(run_dict)
                print(f"Done ({run.status})")
            else:
                print(f"  -> {model_name} (Qseq+CDDiff)... SKIP (already exists)")

    # Save best-k mapping
    if best_k_by_dataset:
        try:
            existing_best_k: Dict[str, int] = {}
            if best_k_json_path.exists():
                with open(best_k_json_path, "r", encoding="utf-8") as f:
                    existing_best_k = json.load(f)
            existing_best_k.update(best_k_by_dataset)
            with open(best_k_json_path, "w", encoding="utf-8") as f:
                json.dump(existing_best_k, f, indent=2, ensure_ascii=False)
            print(f"\n[OK] Saved best_k_by_val.json: {best_k_json_path}")
        except Exception as e:
            print(f"\n[WARN] Failed to save best_k_by_val.json: {e}")
    else:
        print(f"\n[INFO] No best-k selected in this run, best_k_by_val.json unchanged")

    print(f"\n{'='*80}")
    print(f"Experiments completed. Results saved to: {out_dir}")
    if run_rows:
        print(f"New runs completed: {len(run_rows)}")
    else:
        print(f"No new runs (all already exist)")
    if runs_summary_path.exists():
        total_runs = len(pd.read_csv(runs_summary_path))
        print(f"Total runs in summary: {total_runs}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

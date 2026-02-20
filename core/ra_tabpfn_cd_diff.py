#!/usr/bin/env python3
# RA-TabPFN with CD-diff retrieval features.

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from core.data_utils import extract_q_sequence, diversify_features

try:
    from tabpfn import TabPFNClassifier
    from tabpfn.regressor import TabPFNRegressor
    TABPFN_AVAILABLE = True
except ImportError:
    TABPFN_AVAILABLE = False


def load_cd_diff_features(dataset_name: str, cd_features_dir: Path, feature_type: str = 'type1') -> pd.DataFrame:
    """Load CD-diff features for a dataset."""
    cd_file = cd_features_dir / f'cd_diff_features_{dataset_name}.csv'
    if not cd_file.exists():
        raise FileNotFoundError(f"Missing CD-diff feature file: {cd_file}")
    return pd.read_csv(cd_file)


def extract_cd_features_from_df(
    battery_ids: List[str],
    cd_df: pd.DataFrame
) -> Dict[str, np.ndarray]:
    battery_cd_features = {}
    unique_batteries = sorted(set(battery_ids))
    
    feature_cols = [col for col in cd_df.columns if col != 'battery_id']
    
    for battery_id in unique_batteries:
        row = cd_df[cd_df['battery_id'] == battery_id]
        
        if len(row) > 0:
            features = row[feature_cols].iloc[0].values.astype(float)
            features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            battery_cd_features[battery_id] = features
        else:
            battery_cd_features[battery_id] = np.zeros(len(feature_cols))
    
    return battery_cd_features


def cosine_similarity_with_nan(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    
    if np.sum(mask) == 0:
        return 0.0
    
    x_valid = x[mask]
    y_valid = y[mask]
    
    norm_x = np.linalg.norm(x_valid)
    norm_y = np.linalg.norm(y_valid)
    
    if norm_x == 0 or norm_y == 0:
        return 0.0
    
    similarity = np.dot(x_valid, y_valid) / (norm_x * norm_y)
    
    return float(similarity)


def retrieve_top_k_batteries_cd(
    test_battery_id: str,
    test_cd_features: Dict[str, np.ndarray],
    train_cd_features: Dict[str, np.ndarray],
    k: int = 10
) -> Tuple[List[str], List[float]]:
    test_feat = test_cd_features[test_battery_id]
    
    train_battery_ids = list(train_cd_features.keys())
    similarities = []
    
    for train_id in train_battery_ids:
        train_feat = train_cd_features[train_id]
        sim = cosine_similarity_with_nan(test_feat, train_feat)
        similarities.append(sim)
    
    similarities = np.array(similarities)
    
    top_k_indices = np.argsort(similarities)[-k:][::-1]
    top_k_battery_ids = [train_battery_ids[i] for i in top_k_indices]
    top_k_similarities = [similarities[i] for i in top_k_indices]
    
    return top_k_battery_ids, top_k_similarities


class RATabPFNCDDiff:
    def __init__(
        self,
        k_neighbors: int = 10,
        cd_features_dir: Path = None,
        feature_type: str = 'type1',
        device: str = 'cuda',
        stage_aware: bool = False,
        stage_delta: float = 0.15,
        min_stage_samples: int = 100
    ):
        self.k = k_neighbors
        self.cd_features_dir = cd_features_dir or Path(__file__).parent
        self.feature_type = feature_type

        self.stage_aware = bool(stage_aware)
        self.stage_delta = float(stage_delta)
        self.min_stage_samples = int(min_stage_samples)
        
        if not TABPFN_AVAILABLE:
            raise ImportError("TabPFN is not installed")

        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        
        self.device = str(device) if isinstance(device, torch.device) else device
    
    def fit(
        self,
        train_data: Dict,
        scaler,
        dataset_name: str
    ):
        self.dataset_name = dataset_name
        self.train_sequences = train_data['sequences']
        self.train_battery_ids = train_data['battery_ids']
        self.train_cycle_numbers = train_data['cycle_numbers']
        self.train_risk_labels = train_data['risk_labels']
        self.train_rul_labels = train_data['rul_labels']
        self.scaler = scaler

        self.train_X = extract_q_sequence(self.train_sequences)
        cd_df = load_cd_diff_features(dataset_name, self.cd_features_dir, self.feature_type)
        
        self.train_cd_features = extract_cd_features_from_df(
            self.train_battery_ids,
            cd_df
        )

        cycle_arr = np.asarray(self.train_cycle_numbers, dtype=float)
        rul_arr = np.asarray(self.train_rul_labels, dtype=float)
        k2o_arr = cycle_arr + rul_arr
        valid = (~np.isnan(rul_arr)) & (rul_arr >= 0) & (k2o_arr > 0)
        stage_arr = np.full_like(k2o_arr, np.nan, dtype=float)
        stage_arr[valid] = cycle_arr[valid] / k2o_arr[valid]
        self.train_stage = stage_arr

    def _filter_by_stage(
        self,
        train_indices: List[int],
        stage_hat: float,
        delta: float = None,
        min_samples: int = None,
    ) -> List[int]:
        if delta is None:
            delta = self.stage_delta
        if min_samples is None:
            min_samples = self.min_stage_samples

        idx = np.asarray(train_indices, dtype=int)
        if idx.size == 0:
            return train_indices

        st = self.train_stage[idx]
        mask = (~np.isnan(st)) & (np.abs(st - float(stage_hat)) <= float(delta))
        filtered = idx[mask]

        if int(filtered.size) < int(min_samples):
            return train_indices
        return filtered.tolist()

    def _estimate_stage_batch(
        self,
        X_train: np.ndarray,
        y_rul_train: np.ndarray,
        X_test: np.ndarray,
        test_cycle_numbers: List[int],
    ) -> np.ndarray:
        y_arr = np.asarray(y_rul_train, dtype=float)
        valid_mask = ~np.isnan(y_arr)
        X_train_valid = X_train[valid_mask]
        y_valid = y_arr[valid_mask]

        if X_train_valid.shape[0] < 10:
            return np.full(len(test_cycle_numbers), 0.5, dtype=float)

        try:
            y_scaled = self.scaler.transform(y_valid.reshape(-1, 1)).flatten()
            tabpfn_reg = TabPFNRegressor(
                device=self.device,
                ignore_pretraining_limits=True,
                n_estimators=1,
                memory_saving_mode=True,
            )
            tabpfn_reg.fit(X_train_valid, y_scaled)
            rul_pred_scaled = tabpfn_reg.predict(X_test)
            rul_hat = self.scaler.inverse_transform(rul_pred_scaled.reshape(-1, 1)).flatten()

            t = np.asarray(test_cycle_numbers, dtype=float)
            denom = t + np.maximum(rul_hat, 1.0)
            stage_hat = np.divide(t, denom, out=np.full_like(t, 0.5, dtype=float), where=denom > 0)
            stage_hat = np.clip(stage_hat, 0.0, 1.0)
            return stage_hat
        except Exception:
            return np.full(len(test_cycle_numbers), 0.5, dtype=float)
    
    def predict(
        self,
        test_data: Dict,
        task: str = 'both'
    ) -> Dict:
        test_sequences = test_data['sequences']
        test_battery_ids = test_data['battery_ids']
        test_cycle_numbers = test_data['cycle_numbers']
        
        test_X = extract_q_sequence(test_sequences)
        cd_df = load_cd_diff_features(self.dataset_name, self.cd_features_dir, self.feature_type)
        
        test_cd_features = extract_cd_features_from_df(
            test_battery_ids,
            cd_df
        )

        n_test = len(test_battery_ids)
        risk_predictions = np.zeros(n_test, dtype=int)
        risk_proba = np.zeros((n_test, 2), dtype=float)
        rul_predictions = np.zeros(n_test)
        similar_batteries_dict = {}
        similarities_dict = {}
        neighbor_stage_dist_dict = {}

        unique_test_batteries = sorted(set(test_battery_ids))

        for i, test_battery_id in enumerate(unique_test_batteries, 1):
            top_k_batteries, top_k_sims = retrieve_top_k_batteries_cd(
                test_battery_id,
                test_cd_features,
                self.train_cd_features,
                k=self.k
            )
            similar_batteries_dict[test_battery_id] = top_k_batteries
            similarities_dict[test_battery_id] = top_k_sims

            train_indices_k = []
            neighbor_cycles_list = []
            for battery_id in top_k_batteries:
                battery_mask = np.array([bid == battery_id for bid in self.train_battery_ids])
                battery_indices = np.where(battery_mask)[0]
                train_indices_k.extend(battery_indices)

                for idx in battery_indices:
                    cycle = self.train_cycle_numbers[idx]
                    rul = self.train_rul_labels[idx]
                    if not np.isnan(rul) and rul >= 0:
                        k2o_estimated = cycle + rul  # k2o = cycle + rul
                        if k2o_estimated > 0:
                            stage = cycle / k2o_estimated
                            neighbor_cycles_list.append(stage)
            
            if len(neighbor_cycles_list) > 0:
                stages = np.array(neighbor_cycles_list)
                stage_bins = [0, 0.2, 0.4, 0.6, 0.8, 1.01]
                stage_dist = np.histogram(stages, bins=stage_bins)[0]
                stage_dist_pct = (stage_dist / len(stages) * 100).tolist()
            else:
                stage_dist_pct = [0.0, 0.0, 0.0, 0.0, 0.0]
            neighbor_stage_dist_dict[test_battery_id] = stage_dist_pct
            
            X_train_k = self.train_X[train_indices_k]
            y_risk_train_k = self.train_risk_labels[train_indices_k]
            y_rul_train_k = self.train_rul_labels[train_indices_k]
            
            test_battery_mask = np.array([bid == test_battery_id for bid in test_battery_ids])
            test_battery_indices = np.where(test_battery_mask)[0]
            
            X_test_battery = test_X[test_battery_indices]

            test_cycles_battery = [test_cycle_numbers[idx] for idx in test_battery_indices]

            if task in ['risk', 'both']:
                if self.stage_aware:
                    stage_hat_batch = self._estimate_stage_batch(
                        X_train_k,
                        y_rul_train_k,
                        X_test_battery,
                        test_cycles_battery,
                    )

                    stage_bins = np.digitize(stage_hat_batch, bins=[0.2, 0.4, 0.6, 0.8])
                    risk_pred_battery = np.zeros(len(test_battery_indices), dtype=int)
                    risk_proba_battery = np.zeros((len(test_battery_indices), 2), dtype=float)

                    for bin_id in range(5):
                        bin_mask = stage_bins == bin_id
                        if not np.any(bin_mask):
                            continue

                        stage_center = float(np.mean(stage_hat_batch[bin_mask]))
                        train_indices_filtered = self._filter_by_stage(train_indices_k, stage_center)

                        X_train_filtered = self.train_X[train_indices_filtered]
                        y_risk_filtered = self.train_risk_labels[train_indices_filtered]

                        if len(np.unique(y_risk_filtered)) < 2:
                            X_train_filtered = X_train_k
                            y_risk_filtered = y_risk_train_k

                        tabpfn_clf = TabPFNClassifier(
                            device=self.device,
                            ignore_pretraining_limits=True,
                            n_estimators=1,
                            memory_saving_mode=False,
                        )
                        tabpfn_clf.fit(X_train_filtered, y_risk_filtered)

                        X_test_bin = X_test_battery[bin_mask]
                        risk_pred_bin = tabpfn_clf.predict(X_test_bin)
                        risk_proba_bin = tabpfn_clf.predict_proba(X_test_bin)
                        risk_pred_battery[bin_mask] = risk_pred_bin
                        risk_proba_battery[bin_mask] = risk_proba_bin

                    risk_predictions[test_battery_indices] = risk_pred_battery
                    risk_proba[test_battery_indices] = risk_proba_battery
                else:
                    tabpfn_clf = TabPFNClassifier(
                        device=self.device,
                        ignore_pretraining_limits=True,
                        n_estimators=1,
                        memory_saving_mode=False
                    )
                    tabpfn_clf.fit(X_train_k, y_risk_train_k)
                    risk_pred_battery = tabpfn_clf.predict(X_test_battery)
                    risk_proba_battery = tabpfn_clf.predict_proba(X_test_battery)
                    risk_predictions[test_battery_indices] = risk_pred_battery
                    risk_proba[test_battery_indices] = risk_proba_battery

            if task in ['rul', 'both']:
                y_rul_train_k_array = np.array(y_rul_train_k)
                train_valid_mask = ~np.isnan(y_rul_train_k_array)
                X_train_k_rul = X_train_k[train_valid_mask]
                y_rul_train_k_valid = y_rul_train_k_array[train_valid_mask]
                
                y_rul_train_k_scaled = self.scaler.transform(y_rul_train_k_valid.reshape(-1, 1)).flatten()
                
                if len(X_train_k_rul) > 0:
                    try:
                        feature_stds = np.std(X_train_k_rul, axis=0)
                        zero_var_count = np.sum(feature_stds < 1e-10)
                        
                        if zero_var_count > 0:
                            X_train_k_rul = diversify_features(X_train_k_rul, noise_level=1e-6)
                            X_test_battery = diversify_features(X_test_battery, noise_level=1e-6, seed=43)
                        
                        tabpfn_reg = TabPFNRegressor(
                            device=self.device,
                            ignore_pretraining_limits=True,
                            n_estimators=1,
                            memory_saving_mode=False
                        )
                        tabpfn_reg.fit(X_train_k_rul, y_rul_train_k_scaled)
                        
                        rul_pred_battery_scaled = tabpfn_reg.predict(X_test_battery)
                        rul_pred_battery = self.scaler.inverse_transform(
                            rul_pred_battery_scaled.reshape(-1, 1)
                        ).flatten()
                        
                        rul_predictions[test_battery_indices] = rul_pred_battery
                    
                    except Exception:
                        mean_rul = float(np.mean(y_rul_train_k_valid))
                        rul_pred_battery = np.full(len(X_test_battery), mean_rul)
                        rul_predictions[test_battery_indices] = rul_pred_battery
        
        return {
            'risk_pred': risk_predictions,
            'risk_proba': risk_proba,
            'rul_pred': rul_predictions,
            'similar_batteries_dict': similar_batteries_dict,
            'similarities_dict': similarities_dict,
            'neighbor_stage_dist_dict': neighbor_stage_dist_dict,
            'similar_batteries': similar_batteries_dict,
            'similarities': similarities_dict,
            'neighbor_stage_dist': neighbor_stage_dist_dict,
        }

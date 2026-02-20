# RA-TabPFN RUL - Fast Access Version

Retrieval-Augmented TabPFN for Battery Remaining Useful Life (RUL) Prediction.

This is the **fast access version** with preprocessed data. For the full pipeline (data preprocessing from scratch), see `RA-TabPFN_RUL2/`.

## Quick Start

### 1. Download Preprocessed Data

Download `data.zip` from [GitHub Release](https://github.com/Chenm2434/RA-TabPFN-RUL/releases/download/v1.0/data.zip) and extract to the `data/` directory:

```
data/
├── cache_unified/          # Required - cached train/test tensors
│   ├── MIT_train.pt
│   ├── MIT_test.pt
│   ├── XJTU_train.pt
│   └── XJTU_test.pt
└── cdfeature/              # Required - CD-diff retrieval features
    ├── cd_diff_features_MIT.csv
    └── cd_diff_features_XJTU.csv
```

> **Note:** `labels/*.csv` files are **not required** for running experiments. All label data (risk, k2o, cycle_numbers, battery_ids) is already embedded in the `.pt` cache files.

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run Experiments

```bash
# Run all models on both datasets (full val+test flow)
python scripts/run_experiments.py --datasets MIT XJTU --models RA-TabPFN MLP CNN Transformer --ks 3 5 7 10 15 --device cuda

# Two-stage execution: validation only (select best k)
python scripts/run_experiments.py --datasets MIT XJTU --models RA-TabPFN --val_only --device cuda

# Two-stage execution: test only (requires prior val results)
python scripts/run_experiments.py --datasets MIT XJTU --models RA-TabPFN --test_only --device cuda

# Run on CPU
python scripts/run_experiments.py --datasets MIT --models MLP --device cpu
```

Results will be saved to `output/comparison_results/`.

## Experiment Flow

The experiment follows a **two-stage validation/test** protocol:

1. **Validation phase**: Run RA-TabPFN (CDDiff + Random baseline) across all k values on a battery-level 10% validation split from the training set. Select best k by minimum validation MAE.
2. **Test phase**: Evaluate RA-TabPFN with the best k (using the full training set) and all DL models (MLP, CNN, Transformer) on the held-out test set. DL models use the battery-level validation split for early stopping.

## Required Data Files

| File | Description | Required |
|------|-------------|----------|
| `data/cache_unified/MIT_train.pt` | MIT training cache (sequences, risk, k2o, cycle_numbers, battery_ids) | **Yes** |
| `data/cache_unified/MIT_test.pt` | MIT test cache | **Yes** |
| `data/cache_unified/XJTU_train.pt` | XJTU training cache | **Yes** |
| `data/cache_unified/XJTU_test.pt` | XJTU test cache | **Yes** |
| `data/cdfeature/cd_diff_features_MIT.csv` | MIT CD-diff features (battery_id + feature columns) | **Yes** |
| `data/cdfeature/cd_diff_features_XJTU.csv` | XJTU CD-diff features | **Yes** |

## Directory Structure

```
RA-TabPFN_RUL_fa/
├── README.md
├── requirements.txt
├── config/
│   └── paths.py                # Path configuration (self-contained, no external deps)
├── core/
│   ├── ra_tabpfn_cd_diff.py    # RA-TabPFN with CD-diff retrieval
│   └── data_utils.py           # Data loading from .pt cache
├── models/
│   ├── mlp.py                  # MLP baseline
│   ├── cnn.py                  # CNN baseline
│   └── transformer.py          # Transformer baseline
├── utils/
│   └── helpers.py
├── scripts/
│   └── run_experiments.py      # Main entry point
├── data/                       # Preprocessed data (download required)
│   ├── cache_unified/          #   .pt files with embedded labels
│   └── cdfeature/              #   CD-diff feature CSVs
├── model/                      # DL checkpoints (auto-created)
│   └── checkpoints/
└── output/                     # Experiment results (auto-created)
    └── comparison_results/
```

## Command Line Options

```
python scripts/run_experiments.py [OPTIONS]

Options:
  --datasets        Datasets to evaluate (default: XJTU MIT)
  --models          Models to run (default: RA-TabPFN MLP CNN Transformer)
  --ks              K values for RA-TabPFN retrieval (default: 3 5 7 10 15)
  --device          Device to use: cuda or cpu (default: cuda)
  --val_only        RA-TabPFN: only run validation, skip test
  --test_only       RA-TabPFN: skip validation, read prior val results to select best k, then test
```

Fixed parameters (hardcoded): seed=42, epochs=100, batch_size=256, lr=1e-3, patience=10, val_battery_ratio=0.10, determinism=warn.

## Output Files

After running experiments, results are saved to `output/comparison_results/`:

| File | Description |
|------|-------------|
| `predictions_val.csv` | RA-TabPFN per-sample predictions on validation set (all k values) |
| `predictions_test.csv` | Per-sample predictions on test set (best-k RA-TabPFN + DL models) |
| `runs_summary.csv` | Aggregated metrics per run (MAE, risk accuracy, timing) |
| `best_k_by_val.json` | Best k per dataset selected by validation MAE |
| `predictions_{dataset}_{model}_*.csv` | Individual DL model prediction files |

DL model checkpoints are saved to `model/checkpoints/` (auto-created, independent from the main project).

## Requirements

- Python 3.9+
- PyTorch 2.0+
- TabPFN
- NumPy, Pandas, scikit-learn, scipy

## License

MIT License

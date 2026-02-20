#!/usr/bin/env python3
"""Path configuration for RA-TabPFN RUL (Fast Access Version).

All preprocessed data should be placed in the `data/` directory.
Download from GitHub Release: <RELEASE_URL>
"""

from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_PROJECT_DIR = _THIS_FILE.parent.parent

# Data directory (preprocessed data goes here)
DATA_DIR = _PROJECT_DIR / "data"

# Cache files (Step5 output)
CACHE_UNIFIED_DIR = DATA_DIR / "cache_unified"

# CD-diff feature files (Step4 output)
CDFEATURE_DIR = DATA_DIR / "cdfeature"

# Output directory
OUTPUT_DIR = _PROJECT_DIR / "output"
COMPARISON_RESULTS_DIR = OUTPUT_DIR / "comparison_results"

# Model checkpoints
MODEL_DIR = _PROJECT_DIR / "model"
CHECKPOINTS_DIR = MODEL_DIR / "checkpoints"

# Battery splits (same as full version)
MIT_TRAIN_BATTERIES = [
    'b1c1', 'b1c2', 'b1c3', 'b1c4', 'b1c6', 'b1c7', 'b1c9', 'b1c14',
    'b1c15', 'b1c16', 'b1c17', 'b1c18', 'b1c20', 'b1c21', 'b1c23', 'b1c24',
    'b1c26', 'b1c27', 'b1c28', 'b1c29', 'b1c31', 'b1c32', 'b1c33', 'b1c34',
    'b1c36', 'b1c37', 'b1c38', 'b1c39', 'b1c41', 'b1c42', 'b1c43', 'b1c44',
    'b2c0', 'b2c2', 'b2c3', 'b2c4', 'b2c6', 'b2c10', 'b2c11',
    'b2c13', 'b2c14', 'b2c17', 'b2c18', 'b2c19', 'b2c21', 'b2c22', 'b2c23',
    'b2c25', 'b2c26', 'b2c27', 'b2c28', 'b2c29', 'b2c31', 'b2c32', 'b2c33',
    'b2c35', 'b2c36', 'b2c37', 'b2c38', 'b2c39', 'b2c41', 'b2c42', 'b2c43',
    'b2c45', 'b2c46', 'b2c47', 'b3c0', 'b3c1', 'b3c4', 'b3c5', 'b3c6',
    'b3c8', 'b3c9', 'b3c10', 'b3c11', 'b3c12', 'b3c14', 'b3c15', 'b3c16',
    'b3c18', 'b3c19', 'b3c20', 'b3c21', 'b3c22', 'b3c25', 'b3c26', 'b3c27',
    'b3c29', 'b3c30', 'b3c31', 'b3c33', 'b3c34', 'b3c36', 'b3c40', 'b3c41',
    'b3c43', 'b3c44', 'b3c45'
]

MIT_TEST_BATTERIES = [
    'b1c0', 'b1c5', 'b1c11', 'b1c19', 'b1c25', 'b1c30', 'b1c35', 'b1c40',
    'b1c45', 'b2c1', 'b2c5', 'b2c12', 'b2c20', 'b2c24', 'b2c30', 'b2c34', 'b2c40',
    'b2c44', 'b3c3', 'b3c7', 'b3c13', 'b3c17', 'b3c24', 'b3c28', 'b3c35',
    'b3c42'
]

MIT_ALL_BATTERIES = MIT_TRAIN_BATTERIES + MIT_TEST_BATTERIES

XJTU_BATTERY_GROUPS = {
    '2C': list(range(1, 9)),
    '3C': list(range(1, 16)),
    'R2.5': list(range(1, 9)),
    'R3': list(range(1, 9)),
    'RW': list(range(1, 9)),
}

XJTU_TEST_NUMBERS = [4, 8, 14]

XJTU_TRAIN_BATTERIES = []
XJTU_TEST_BATTERIES = []

for group, numbers in XJTU_BATTERY_GROUPS.items():
    for num in numbers:
        battery_name = f'{group}_battery-{num}'
        if num in XJTU_TEST_NUMBERS and num <= max(numbers):
            XJTU_TEST_BATTERIES.append(battery_name)
        else:
            XJTU_TRAIN_BATTERIES.append(battery_name)

XJTU_ALL_BATTERIES = XJTU_TRAIN_BATTERIES + XJTU_TEST_BATTERIES

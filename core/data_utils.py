#!/usr/bin/env python3
"""Data loading and utility functions for RA-TabPFN."""

import numpy as np
import torch
from typing import Dict


def diversify_features(X: np.ndarray, noise_level: float = 1e-6, seed: int = 42) -> np.ndarray:
    """Add small noise to avoid zero-variance features."""
    np.random.seed(seed)
    feature_ranges = np.max(X, axis=0) - np.min(X, axis=0)
    feature_ranges = np.where(feature_ranges == 0, 1.0, feature_ranges)
    noise = np.random.normal(0, noise_level * feature_ranges, X.shape)
    return X + noise


def load_phase2_with_metadata(cache_file) -> Dict:
    try:
        data = torch.load(cache_file, map_location="cpu", weights_only=False)
    except TypeError:
        data = torch.load(cache_file, map_location="cpu")

    if isinstance(data['sequences'], torch.Tensor):
        data['sequences'] = data['sequences'].numpy()

    if 'risk' in data:
        if isinstance(data['risk'], torch.Tensor):
            data['risk_labels'] = data['risk'].numpy()
        else:
            data['risk_labels'] = data['risk']
    
    if 'k2o' in data:
        if isinstance(data['k2o'], torch.Tensor):
            k2o_values = data['k2o'].numpy()
        else:
            k2o_values = data['k2o']
    else:
        k2o_values = data['k2o_labels']
    
    cycle_numbers = np.array(data['cycle_numbers'])

    k2o_cycles = k2o_values

    rul_values = k2o_cycles - cycle_numbers
    rul_values = np.maximum(rul_values, 0)

    data['rul_labels'] = rul_values
    return data


def extract_q_sequence(sequences: np.ndarray) -> np.ndarray:
    return sequences[:, :, 1]


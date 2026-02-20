#!/usr/bin/env python3
"""MLP models for battery risk classification and RUL regression."""

import torch
import torch.nn as nn


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=155, hidden_dims=[256, 128, 64], num_classes=2, dropout=0.3):
        super(MLPClassifier, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout if i == 0 else dropout * 0.67))
            
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim=155, hidden_dims=[256, 128, 64], dropout=0.3):
        super(MLPRegressor, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, hidden_dim))

            if i < len(hidden_dims) - 1:
                layers.append(nn.BatchNorm1d(hidden_dim))

            layers.append(nn.ReLU())

            if i < len(hidden_dims) - 1:
                layers.append(nn.Dropout(dropout if i == 0 else dropout * 0.67))
            
            prev_dim = hidden_dim

        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

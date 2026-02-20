#!/usr/bin/env python3
"""1D CNN models for battery risk classification and RUL regression."""

import torch
import torch.nn as nn


class CNN1DClassifier(nn.Module):
    def __init__(self, input_dim=155, channels=[32, 64, 128], kernel_sizes=[3, 5, 7], 
                 fc_dim=64, num_classes=2, dropout=0.3):
        super(CNN1DClassifier, self).__init__()
        
        self.input_dim = input_dim

        conv_layers = []
        in_channels = 1
        
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            padding = kernel_size // 2
            
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
            
            in_channels = out_channels

        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        
        self.conv_network = nn.Sequential(*conv_layers)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, num_classes)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_network(x)
        return self.fc(x)


class CNN1DRegressor(nn.Module):
    def __init__(self, input_dim=155, channels=[32, 64, 128], kernel_sizes=[3, 5, 7], 
                 fc_dim=64, dropout=0.3):
        super(CNN1DRegressor, self).__init__()
        
        self.input_dim = input_dim

        conv_layers = []
        in_channels = 1
        
        for i, (out_channels, kernel_size) in enumerate(zip(channels, kernel_sizes)):
            padding = kernel_size // 2
            
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ])
            
            in_channels = out_channels

        conv_layers.append(nn.AdaptiveAvgPool1d(1))
        
        self.conv_network = nn.Sequential(*conv_layers)

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels[-1], fc_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_dim, 1)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_network(x)
        return self.fc(x)

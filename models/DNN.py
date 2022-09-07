import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List
from types import SimpleNamespace


class DNN(nn.Module):
    
    def __init__(self, x_time_length: int, x_num_features: int, y_num_features: int) -> None:
        
        super().__init__()
        self.hparams = SimpleNamespace(
            x_time_length = x_time_length, 
            x_num_features = x_num_features, 
            y_num_features = y_num_features,
        )
        self._create_network()
        self._init_params()
    
    
    def _create_network(self) -> None:
        
        dimensions = [self.hparams.x_time_length * self.hparams.x_num_features, 
                      1000, 1000, 1 * self.hparams.y_num_features] # 3 layers
        self.layers = []
        for i in range(len(dimensions) - 1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:
                self.layers.append(nn.BatchNorm1d(dimensions[i + 1]))
                self.layers.append(nn.ReLU(inplace = True))
        
        self.dnn = nn.Sequential(*self.layers)
    
    
    def _init_params(self) -> None:
        
        pass

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = x.view(-1, self.hparams.x_time_length * self.hparams.x_num_features) # Shape: (batch, time, features) --> (batch, time * features)
        out = self.dnn(x)
        out = out.view(-1, 1, self.hparams.y_num_features) # Shape: (batch, time * features) --> (batch, time, features)
        
        return out
    
        

if __name__ == '__main__':
    
    x_time_length = 128
    x_num_features = 38
    y_num_features = 9
    
    dnn = DNN(x_time_length, x_num_features, y_num_features)
    print(dnn)
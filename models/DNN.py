import os
import torch
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from typing import List


class DNN(pl.LightningModule):
    
    def __init__(self, x_time_length: int, x_num_features: int, y_num_features: int,
                 dimensions: List[int], enable_BN: bool = True):
        
        super().__init__()
        
        self.x_time_length = x_time_length
        self.x_num_features = x_num_features
        self.y_num_features = y_num_features
        
        self.layers = []
        for i in range(len(dimensions) - 1):
            self.layers.append(nn.Linear(dimensions[i], dimensions[i + 1]))
            if i < len(dimensions) - 2:
                if enable_BN:
                    self.layers.append(nn.BatchNorm1d(dimensions[i + 1]))
                self.layers.append(nn.ReLU(inplace = True))
        
        self.dnn = nn.Sequential(*self.layers)
        
        
    def forward(self, x):
        
        x = x.view(-1, self.x_time_length * self.x_num_features) # Shape: (batch, time, features) --> (batch, time * features)
        out = self.dnn(x)
        out = out.view(-1, 1, self.y_num_features) # Shape: (batch, time * features) --> (batch, time, features)
        
        return out
    
    
    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        
        return optimizer
    
    
    def training_step(self, train_batch, batch_idx):
    
        x, y = train_batch
        x, y = x.float(), y.float()
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log('train_loss', loss)
        
        return loss
    
    
    def validation_step(self, val_batch, batch_idx):
        
        x, y = val_batch
        x, y = x.float(), y.float()
        y_pred = self.forward(x)
        loss = F.mse_loss(y_pred, y)
        self.log('val_loss', loss)


if __name__ == '__main__':
    
    x_time_length = 128
    x_num_features = 38
    y_num_features = 9
    dimensions = [x_time_length * x_num_features, 1000, 1000, 1 * y_num_features]
    
    dnn = DNN(x_time_length, x_num_features, y_num_features, dimensions, enable_BN = True)
    print(dnn)
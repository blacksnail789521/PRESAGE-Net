import os
import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import List, Tuple
from types import SimpleNamespace
from torchmetrics import MeanAbsoluteError, SymmetricMeanAbsolutePercentageError

from .DNN import DNN


def create_model(model_name: str, model_hparams: dict) -> nn.Module:
    
    assert model_name in ['DNN', 'PRESAGE-Net']
    
    return globals()[model_name](**model_hparams)
    

class ModelModule(pl.LightningModule):
    
    def __init__(self, model_name: str, model_hparams: dict, optimizer_hparams: dict) -> None:
        
        super().__init__()
        
        # Exports the hyperparameters to a YAML file, and create 'self.hparams' namespace
        self.save_hyperparameters()
        
        # Create model
        self.model = create_model(model_name, model_hparams)
        
        # Create loss module and metric modules
        self.loss_module = nn.MSELoss()
        self.mae_module = MeanAbsoluteError()
        self.smape_module = SymmetricMeanAbsolutePercentageError()
        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return self.model(x)
        
    
    def configure_optimizers(self) -> Tuple[List[object], List[object]]:
        
        optimizer = torch.optim.Adam(self.parameters(), **self.hparams.optimizer_hparams)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1)
        
        return [optimizer], [scheduler]
    
    
    def log_loss_and_metric(self, y_pred, y, mode):
        
        loss = self.loss_module(y_pred, y)
        self.log(f'{mode}_loss', loss)
        self.log(f'{mode}_mae', self.mae_module(y_pred, y))
        self.log(f'{mode}_smape', self.smape_module(y_pred, y))
        
        return loss
    
    
    def training_step(self, batch, batch_idx):
    
        x, y = batch
        y_pred = self.model(x)
        loss = self.log_loss_and_metric(y_pred, y, 'train')
        
        return loss # Return tensor to call '.backward' on
    
    
    def validation_step(self, batch, batch_idx):
        
        x, y = batch
        y_pred = self.model(x)
        self.log_loss_and_metric(y_pred, y, 'val')
    
    
    def test_step(self, batch, batch_idx):
        
        x, y = batch
        y_pred = self.model(x)
        self.log_loss_and_metric(y_pred, y, 'test')
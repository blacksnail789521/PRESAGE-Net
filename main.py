import pandas as pd
import numpy as np
import os
import argparse
import random
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Tuple
from tqdm import tqdm

from datasets.dataset import MCMP_dataset
from models.DNN import DNN


class Main():
    
    def __init__(self) -> None:
        
        self.args = self.parse_args()
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() and self.args.device == 'cuda' 
            else 'cpu')
        
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)
    
    
    def parse_args(self) -> argparse.Namespace:
        
        parser = argparse.ArgumentParser(description = 'PRESAGE-Net', 
                                         formatter_class = argparse.ArgumentDefaultsHelpFormatter)
        
        parser.add_argument('--batch_size', help = 'batch size', type = int, default = 128)
        parser.add_argument('--epochs', help = 'training epochs', type = int, default = 1)
        parser.add_argument('--accelerator', help = 'accelerator for Trainer in pytorch-lightning', 
                            type = str, choices = ['cpu', 'gpu'], default = 'cpu')
        parser.add_argument('--seed', help = 'random seed', type = int, default = 0)
        parser.add_argument('--model', help = 'model', type = str, 
                            choices = ['all', 'DNN', 'PRESAGE-NET'], default = 'DNN')
        parser.add_argument('--x_time_length', help = 'input length in terms of time', type = int, default = 128)
        parser.add_argument('--shuffle', help = 'shuffle for Dataloader', type = bool, default = True)
        parser.add_argument('--lr', help = 'learning rate', type = float, default = 1.e-3)
        parser.add_argument('--gamma', help = 'learning rate step gamma', type = float, default = 0.999)
        
        args = parser.parse_args()
        
        return args
        
    
    def get_dataloader(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        
        train_ds = MCMP_dataset(self.args.x_time_length, 'train')
        train_loader = DataLoader(train_ds, batch_size = self.args.batch_size, 
                                  shuffle = self.args.shuffle)
        
        val_ds = MCMP_dataset(self.args.x_time_length, 'val')
        val_loader = DataLoader(val_ds, batch_size = self.args.batch_size, 
                                shuffle = self.args.shuffle)
        
        test_ds = MCMP_dataset(self.args.x_time_length, 'test')
        test_loader = DataLoader(test_ds, batch_size = self.args.batch_size, 
                                 shuffle = False)
        
        return train_loader, val_loader, test_loader
    
    
    def get_num_features(self) -> Tuple[int, int]:
        
        train_ds = MCMP_dataset(self.args.x_time_length, 'train')
        x_num_features = len(train_ds.x_cols)
        y_num_features = len(train_ds.y_cols)
        
        return x_num_features, y_num_features
    
    
    def train(self) -> float:
        
        self.model.train() # set to 'train' mode
        for index, (x, y) in enumerate(tqdm(self.train_loader, desc = 'train')):
            x, y = x.to(self.device).float(), y.to(self.device).float()
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = torch.nn.MSELoss()(y, y_pred)
            loss.backward()
            self.optimizer.step()
            
        mse = loss.item()
        print(f'[Train] MSE: {mse:.2%}')
        
        return mse
    
    
    def test(self) -> float:
        
        self.model.train() # set to 'train' mode
        for index, (x, y) in enumerate(tqdm(self.train_loader, desc = 'train')):
            x, y = x.to(self.device).float(), y.to(self.device).float()
            self.optimizer.zero_grad()
            y_pred = self.model(x)
            loss = torch.nn.MSELoss()(y, y_pred)
            loss.backward()
            self.optimizer.step()
            
        mse = loss.item()
        print(f'[Train] MSE: {mse:.2%}')
        
        return mse
    
    
    def run(self) -> None:
        
        # Load data
        self.train_loader, self.val_loader, self.test_loader = self.get_dataloader()
        self.x_num_features, self.y_num_features = self.get_num_features()
        
        # Load model
        if self.args.model == 'DNN':
            
            dimensions = [self.args.x_time_length * self.x_num_features, 
                          1000, 1000, 1 * self.y_num_features]
            self.model = DNN(self.args.x_time_length, self.x_num_features, self.y_num_features, 
                             dimensions, enable_BN = True)
        print(self.model)
        
        trainer = pl.Trainer(
            # accelerator = self.args.accelerator, 
            max_epochs = self.args.epochs,
        )
        trainer.fit(model = self.model, train_dataloaders = self.train_loader)
        
        
        
        
        
        # # Define optimizer, scheduler
        # self.optimizer = torch.optim.Adam(self.model.parameters(), 
        #                                   lr = self.args.lr, betas = (0.5, 0.99))
        # self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = 1, 
        #                                                  gamma = self.args.gamma)
        
        # # Train + test
        # for epoch in range(1, self.args.epochs + 1):
        #     print('-----------------------------------------------------------------')
        #     print(f'Epoch {epoch}:')
        #     train_loss = self.train()
        #     self.scheduler.step() # Update learning_rate every epoch, not every batch
        
        # Test


if __name__ == '__main__':
    
    main = Main()
    result = main.run()
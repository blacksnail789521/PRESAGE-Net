import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import os
import argparse
import random
import torch
from torch.utils.data import DataLoader
from typing import Tuple
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from datasets.dataset import MCMP_dataset
from models.ModelModule import ModelModule


def get_dataloader(args: argparse.Namespace) -> Tuple[DataLoader, DataLoader, DataLoader]:
    
    train_ds = MCMP_dataset(args.x_time_length, 'train')
    train_loader = DataLoader(train_ds, batch_size = args.batch_size, 
                              num_workers = args.num_workers, shuffle = args.shuffle)
    
    val_ds = MCMP_dataset(args.x_time_length, 'val')
    val_loader = DataLoader(val_ds, batch_size = args.batch_size, 
                            num_workers = args.num_workers, shuffle = False)
    
    test_ds = MCMP_dataset(args.x_time_length, 'test')
    test_loader = DataLoader(test_ds, batch_size = args.batch_size, 
                             num_workers = args.num_workers, shuffle = False)
    
    return train_loader, val_loader, test_loader


def get_num_features(x_time_length: int) -> Tuple[int, int]:
    
    train_ds = MCMP_dataset(x_time_length, 'train')
    x_num_features = len(train_ds.x_cols)
    y_num_features = len(train_ds.y_cols)
    
    return x_num_features, y_num_features


def parse_args() -> argparse.Namespace:
    
    parser = argparse.ArgumentParser(description = 'PRESAGE-Net', 
                                     formatter_class = argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--batch_size', help = 'batch size', type = int, default = 128)
    parser.add_argument('--epochs', help = 'training epochs', type = int, default = 100)
    parser.add_argument('--accelerator', help = 'accelerator for Trainer in pytorch-lightning', 
                        type = str, choices = ['cpu', 'gpu'], default = 'gpu')
    parser.add_argument('--devices', help = 'list od device-id for Trainer in pytorch-lightning', 
                        nargs='+', type = int, default = [0])
    parser.add_argument('--num_workers', help = 'num_workers for data_loader', type = int, default = 0)
    parser.add_argument('--seed', help = 'random seed', type = int, default = 0)
    parser.add_argument('--model_name', help = 'model name', type = str, 
                        choices = ['all', 'DNN', 'PRESAGE-NET'], default = 'DNN')
    parser.add_argument('--x_time_length', help = 'input length in terms of time', type = int, default = 128)
    parser.add_argument('--shuffle', help = 'shuffle for train_loader', type = bool, default = True)
    parser.add_argument('--lr', help = 'learning rate', type = float, default = 1.e-5)
    parser.add_argument('--gamma', help = 'learning rate step gamma', type = float, default = 0.999)
    
    args = parser.parse_args()
    
    return args


if __name__ == '__main__':
    
    args = parse_args()
    pl.seed_everything(args.seed)
    
    # Load data
    train_loader, val_loader, test_loader = get_dataloader(args)
    x_num_features, y_num_features = get_num_features(args.x_time_length)
    
    # Load model
    if args.model_name == 'DNN':
        
        model = ModelModule(
            'DNN', 
            model_hparams = {
                'x_time_length': args.x_time_length, 
                'x_num_features': x_num_features, 
                'y_num_features': y_num_features,
            },
            optimizer_hparams = {
                "lr": args.lr,
                # "weight_decay": 1e-4,
            },
        )
    print(model)
    
    # Train model with pytorch-lightning
    trainer = pl.Trainer(
        default_root_dir = os.path.join('lightning_logs', args.model_name), 
        accelerator = args.accelerator, 
        devices = args.devices,
        max_epochs = args.epochs,
        callbacks = [
            EarlyStopping(monitor = 'val_loss', patience = 3),
            LearningRateMonitor('epoch'),
            ModelCheckpoint(save_weights_only = True, monitor = 'val_loss', mode = 'min'),
        ],
    )
    trainer.fit(model, train_loader, val_loader)
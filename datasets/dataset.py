import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import re
from typing import Tuple


class MCMP_dataset(Dataset):
    
    def __init__(self, x_time_length: int = 128, mode: str = 'train') -> None:
        
        self.x_time_length = x_time_length
        self.mode = mode
        
        assert self.mode in ['train', 'validate', 'test']
        self.root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.df = pd.read_csv(os.path.join(self.root_path, 'data', f'{mode}.csv'), index_col = 'time_stamp')
        
        # Set x_all and y_all (use numpy instead of df to speedup)
        self.x_cols = [ col for col in self.df.columns if not re.search('Stage1.*.Actual', col) ]
        self.y_cols = [ col for col in self.df.columns if re.search('Stage1.*.Actual', col) ]
        
        self.x_all = self.df[self.x_cols].to_numpy()
        self.y_all = self.df[self.y_cols].to_numpy()
        
        
    def __len__(self) -> int:
        
        return len(self.df) - self.x_time_length
        
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        x = self.x_all[ index : index + self.x_time_length ]
        y = self.y_all[ index + self.x_time_length : index + self.x_time_length + 1]
        
        return x, y


if __name__ == '__main__':
    
    '''---------------------------------------------------------------'''
    x_time_length = 5
    # x_time_length = 128
    
    mode = 'train'
    # mode = 'validate'
    # mode = 'test'
    
    batch_size = 4
    
    # shuffle = True
    shuffle = False
    '''---------------------------------------------------------------'''
    
    ds = MCMP_dataset(x_time_length, mode)
    ds_loader = DataLoader(ds, batch_size = batch_size, shuffle = shuffle)
    ds_iterator = iter(ds_loader)
    
    x_batch, y_batch = next(ds_iterator)
    x_batch_numpy, y_batch_numpy = x_batch.numpy(), y_batch.numpy()
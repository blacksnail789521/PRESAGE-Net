import torch
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import re



if __name__ == '__main__':
    
    mode = 'train'
    # mode = 'validate'
    # mode = 'test'
    
    x_time_length = 5
    
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    df = pd.read_csv(os.path.join(root_path, 'data', f'{mode}.csv'), index_col = 'time_stamp')
    
    n = len(df) - x_time_length + 1
    
    x_cols = [ col for col in df.columns if not re.search('Stage1.*.Actual', col) ]
    y_cols = [ col for col in df.columns if re.search('Stage1.*.Actual', col) ]
    
    x_all = df[x_cols].to_numpy()
    y_all = df[y_cols].to_numpy()
    
    
    index = 0 
    x = x_all[ index : index + x_time_length ]
    y = y_all[ index + x_time_length : index + x_time_length + 1]
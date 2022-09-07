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
from models.ModelModule import ModelModule


def test_DNN() -> None:

    x_time_length = 128
    x_num_features = 38
    y_num_features = 9
    
    dnn = ModelModule(
        'DNN', 
        model_hparams = {'x_time_length': x_time_length, 
                         'x_num_features': x_num_features, 
                         'y_num_features': y_num_features},
    )
    print(dnn)


if __name__ == '__main__':
    
    test_DNN()
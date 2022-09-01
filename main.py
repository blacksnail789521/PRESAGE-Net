import pandas as pd
import numpy as np
import os
import argparse
import random
import torch

from datasets.load_data import read_MCMP_dataset


class Main():
    
    def __init__(self, args):
        
        self.args = args
    
    
    def get_dataloader(self):
        
        time_series_df, measurements = read_MCMP_dataset()
        
        return time_series_df, measurements
    
    
    def run(self):
        
        # Load data
        # train_loader, val_loader, test_loader = self.get_dataloader()
        return self.get_dataloader()
        
        # Load model
        if self.args.model == 'DNN':
            raise Exception("XD")
        
        # Train
        
        # Test
        
        


if __name__ == '__main__':
    
    # Get args
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', help = 'batch size', type = int, default = 128)
    parser.add_argument('--epoch', help = 'train epoch', type = int, default = 100)
    parser.add_argument('--device', help = 'cuda / cpu', type = str, choices = ['cuda', 'cpu'], default = 'cuda')
    parser.add_argument('--random_seed', help = 'random seed', type = int, default = 0)
    parser.add_argument('--val_ratio', help = 'val ratio', type = float, default = 0.1)
    parser.add_argument('--model', help = 'model', type = str, 
                        choices = ['all', 'DNN', 'PRESAGE-NET'], default = 'DNN')
    args = parser.parse_args()
    
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True
    # os.environ['PYTHONHASHSEED'] = str(args.random_seed)


    main = Main(args)
    result = main.run()
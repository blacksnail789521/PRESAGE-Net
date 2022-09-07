import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from typing import Tuple


def process_MCMP_raw_data(
    sample_rate: int = 1,
    split_ratio: Tuple[float, float, float] = (0.7, 0.15, 0.15),
) -> pd.DataFrame:
    
    root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    df = pd.read_csv(os.path.join(root_path, 'data', 'raw_data', 'continuous_factory_process.csv'))
    
    # Only keep features that are time, ambient, machine1, machine2, machine3, stage1's measurement (non-setpoint)
    keyword_list = ['time_stamp', 'Ambient', 'Machine1', 'Machine2', 'Machine3', 'Stage1.*.Actual']
    df = df.loc[ : , np.any([ df.columns.str.contains(keyword, regex = True) 
                                          for keyword in keyword_list ], axis = 0) ]
    
    # Set time_stamp as index
    df.set_index(pd.to_datetime(df.pop('time_stamp')), inplace = True)
    
    # Make sure to have unified sample rate
    df = df.resample(f'{sample_rate}S').mean().interpolate()
    assert df.isnull().values.any() == False, 'There are some nan values.'
    
    # Remove columns with too many zero entries.
    print(f'Before: {df.shape[1]}')
    df = df.loc[:, df.eq(0).mean() < 0.3]
    print(f'After: {df.shape[1]}')
    
    # Normalize data (only for x, not y)
    x_cols = [ col for col in df.columns if 'Measurement' not in col ]
    y_cols = [ col for col in df.columns if 'Measurement' in col ]
    df_x, df_y = df[x_cols], df[y_cols]
    df_x = (df_x - df_x.mean()) / df_x.std()
    df = pd.concat([df_x, df_y], axis = 1)
    
    # Split into train, val, test
    assert np.sum(split_ratio) == 1, 'sum of split_ratio should be equal to one'
    split_point = (split_ratio[0], split_ratio[0] + split_ratio[1])
    train_df = df[ : int(len(df) * split_point[0]) ]
    val_df = df[ int(len(df) * split_point[0]) : int(len(df) * split_point[1]) ]
    test_df = df[ int(len(df) * split_point[1]) : ]
    
    # Save to csv respectively
    train_df.to_csv(os.path.join(root_path, 'data', 'train.csv'))
    val_df.to_csv(os.path.join(root_path, 'data', 'val.csv'))
    test_df.to_csv(os.path.join(root_path, 'data', 'test.csv'))
    
    return train_df, val_df, test_df


if __name__ == '__main__':
    
    train_df, val_df, test_df = process_MCMP_raw_data()
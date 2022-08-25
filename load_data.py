import pandas as pd
import numpy as np
import os


def read_MCMP_dataset(sample_rate: int = 1):
    
    df = pd.read_csv(os.path.join('data', 'continuous_factory_process.csv'))
    
    # Only keep features that are time, ambient, machine1, machine2, machine3, stage1's measurement (setpoint)
    keyword_list = ['time_stamp', 'Ambient', 'Machine1', 'Machine2', 'Machine3']
    time_series_df = df.loc[ : , np.any([ df.columns.str.contains(keyword) 
                                          for keyword in keyword_list ], axis = 0) ]
    measurements = df.loc[ : , df.columns.str.contains('Stage1.*Setpoint', regex = True) ].iloc[0].to_dict()
    
    # Set time_stamp as index
    time_series_df.set_index(pd.to_datetime(df.pop('time_stamp')), inplace = True)
    
    # Make sure to have unified sample rate
    time_series_df = time_series_df.resample(f'{sample_rate}S').mean()
    
    return time_series_df, measurements


if __name__ == '__main__':
    
    time_series_df, measurements = read_MCMP_dataset()
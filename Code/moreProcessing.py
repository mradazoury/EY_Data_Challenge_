#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 17 11:58:10 2019

@author: mradazoury
"""

import pandas as pd
import numpy as np
from multiprocessing import Pool

processed = pd.read_csv('data/processed_agged.csv')

df = processed.copy()
_dft = df[['hash','x_entry','y_entry','changed']][(~df['x_exit'].isnull()) & (df['time_difference']!=0)]
df_split = np.array_split(df, 20)
pool = Pool(4)

df['percentage_change_square'] = 0

def prev_perc_move_full(df):
    print('Started split')
    def prev_perc_move(row):
        c = 1000
        x = row['x_entry']
        y = row['y_entry']
        temp_df = _dft[(_dft['x_entry'] <= x + c) & (_dft['x_entry'] >= x - c) & (_dft['y_entry'] >= y - c) 
                      & (_dft['y_entry'] >= y - c)]
        if len(temp_df) == 0:
            temp = -1
        else:
            temp= temp_df['changed'].sum() / len(temp_df)
        row['percentage_change_square'] = temp
        return row 
    df = df.apply(prev_perc_move, axis=1)
    print('Ended split')
    return df

df = pd.concat(pool.map(prev_perc_move_full, df_split))
pool.close()
pool.join()

df.to_csv('data/final_processed.csv', index=False)
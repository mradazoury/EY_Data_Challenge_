#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:11:33 2019

@author: mradazoury
"""
import pandas as pd
import numpy as np
from utils import *


## Load raw CSVs
train_data = pd.read_csv("data/data_train.csv")
train_data['test'] = 0
test_data = pd.read_csv("data/data_test.csv")
test_data['test'] = 1

## Concatenate train and test for processing
rawDataSet = pd.concat([train_data, test_data], axis=0)

## Add columns whether entry points is in town
rawDataSet['entry_in_town'] = np.where((rawDataSet.x_entry <= 3770901.5068) & (rawDataSet.x_entry >= 3750901.5068) & (rawDataSet.y_entry >= - 19268905.6133 )
                                & (rawDataSet.y_entry <= - 19208905.6133 ),1,0)

## Add columns whether entry points is in town
rawDataSet['exit_in_town'] = np.where((rawDataSet.x_exit <= 3770901.5068) & (rawDataSet.x_exit >= 3750901.5068) & (rawDataSet.y_exit >= - 19268905.6133 )
                                & (rawDataSet.y_exit <= - 19208905.6133 ),1,0)

### Convert/calculate times
rawDataSet = get_time_columns(rawDataSet)

## Calculate distance traveled 
rawDataSet = distance_traveled(rawDataSet)

## Calculate whether trajectory moved device into town
rawDataSet = moved_into_town(rawDataSet)

## Calculate whether trajectory moved device out of town
rawDataSet = moved_outof_town(rawDataSet)

## Column whether data point changed in/out city
rawDataSet['changed'] =  rawDataSet['moved_into_town'] + rawDataSet['moved_outof_town']

## Calculate average velocity from distance and time
rawDataSet = calculated_velocity(rawDataSet)

## Calculate distance from city_center
rawDataSet = distance_from_city_center(rawDataSet)

## Replace all the true test with NULL for new columns created above
columnsNotKnownForTest = ['distance_traveled', 'exit_in_town', 'moved_into_town', 'moved_outof_town', 'calculated_velocity']
for col in columnsNotKnownForTest:
    rawDataSet.loc[np.isnan(rawDataSet.y_exit), col] = np.nan 

rawDataSet.to_csv('data/processed.csv', index=False)


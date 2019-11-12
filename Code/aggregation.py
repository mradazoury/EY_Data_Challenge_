#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 19:36:44 2019

@author: caterinaselman
"""

from utils import *

processed = pd.read_csv('data/processed.csv')

processed = processed.drop(columns='Unnamed: 0', axis=1)

processed['hash'] = processed['hash'].astype(np.string_)

### Remove any rows where time difference ==0
#processed = processed[processed['time_difference'] != 0]
############

processed = processed.sort_values(['hash'])

processed['trajectory_num'] = 0


processed2 = processed.copy()
processed2['trajectory_id_count'] = processed2['trajectory_id']

aggFunction = {
            'trajectory_id_count': ['count'],
            'hash': ['last']
            }


aggregated = processed2.groupby(['hash']).agg(aggFunction)
aggregated.columns = aggregated.columns.droplevel(level=1)
## try  to add the trajectory_num based on aggregations done
realIndex = 0
for deviceIndex in aggregated.index:
    numberTraj = aggregated.get_value(deviceIndex,'trajectory_id_count')
    upToIndex = realIndex + numberTraj
    for index in range(realIndex,upToIndex):
        indexVal = processed.index[index]
        processed.set_value(indexVal, 'trajectory_num',index - realIndex + 1)
    realIndex = upToIndex
    
### Save to csv
processed.to_csv('processedWTrajNum.csv', index=False)


processed['x_exit'] = processed['x_exit'].replace(np.nan, 0)
processed['y_exit'] = processed['y_exit'].replace(np.nan, 0)

#-find max # of trajectories per device —> max
maxTrajectories = processed['trajectory_num'].max()

target = processed['exit_in_town']

aggFunction = {
        'hash': { 'hash': 'last' },
        'trajectory_id': {'trajectory_id': 'last' },
        'vmax': { 'vmax': 'mean' },
        'vmin': { 'vmin': 'mean'  },
        'vmean': { 'vmean': 'mean' },
        'x_entry': { 'x_entry': 'last'},
        'y_entry': { 'y_entry': 'last'},
        'x_exit': { 'x_exit': 'last'},
        'y_exit': { 'y_exit': 'last'},
        'test': { 'test': 'last'},
        'entry_in_town': { 'entry_in_town': 'last', 'sum_entry_in_town':'sum' },
        'exit_in_town': {'sum_exit_in_town':'sum' },
        'distance_traveled': { 'distance_traveled': 'mean'},
        'moved_into_town': { 'sum_moved_into_town': 'sum'},
        'moved_outof_town': { 'sum_moved_outof_town': 'sum'},
        'time_difference': { 'time_difference': 'last', 'avg_time_difference':'mean'},
        'calculated_velocity': { 'calculated_velocity': 'mean'},
        'time_entry_seconds': { 'time_entry_seconds': 'last'},
        'time_exit_seconds': { 'time_exit_seconds': 'last'},
        'distance_from_city_center': { 'distance_from_city_center': 'last'},
        'trajectory_num': {'trajectory_num': 'last'},
        'changed': {'changed': 'last'}
        }
#####
aggedDF = pd.DataFrame()

#-for i from 1 to max
#for i in range(1:maxTrajectories+1):
for i in range(1,maxTrajectories+1):
    # Find devices that have at least i trajectories
    relevantDevices = aggregated[aggregated['trajectory_id_count'] >= i]['hash']
    # Filter dataset on devices that have at last i trajectories
    filtered = processed[processed['hash'].isin(relevantDevices)]
    # Filter dataset on trajectories up to i
    filtered = filtered[filtered['trajectory_num']<=i].copy()
    ## train.loc[train['Pclass'] == 1, 'Cabin'] = 1
    ## 'distance_traveled', 'exit_in_town', 'moved_into_town',
    ## 'moved_outof_town', 'calculated_velocity'
    filtered.loc[filtered['trajectory_num'] == i, ['distance_traveled', 'exit_in_town', 'moved_into_town',
                 'moved_outof_town', 'calculated_velocity']] = np.nan
    # Group by device and do aggregation functions
    aggregatedRows = filtered.groupby(['hash']).agg(aggFunction)

    # Write to CSV
    CSVName = 'agged' + str(i) + '.csv'
    aggregatedRows.to_csv(CSVName, index=False)
    
    # Append
    aggedDF = aggedDF.append(aggregatedRows)
    
aggedDF.columns = aggedDF.columns.droplevel(level=0)

aggedDF['exit_in_town'] = np.where((aggedDF.x_exit <= 3770901.5068) & (aggedDF.x_exit >= 3750901.5068) & (aggedDF.y_exit >= - 19268905.6133 )
                                & (aggedDF.y_exit <= - 19208905.6133 ),1,0)


aggedDF.to_csv('data/processed_agged.csv', index=False)
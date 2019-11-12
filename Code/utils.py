#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 18:17:20 2019

@author: caterinaselman
"""

import pandas as pd
import numpy as np
import math as m
import matplotlib as plt
import seaborn as sns
import sklearn as skl
import warnings
import statsmodels.api as sm
import math
import datetime
from dateutil import parser

from multiprocessing import Pool
from sklearn import datasets, linear_model
from sklearn.linear_model import LogisticRegression, LinearRegression 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.model_selection import cross_val_score, cross_val_predict, ShuffleSplit, validation_curve, cross_validate
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelBinarizer, RobustScaler, LabelEncoder, scale, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_curve, roc_auc_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn import datasets
from sklearn.feature_selection import RFE,SelectFromModel
from sklearn.svm import SVC

from sklearn.ensemble import RandomForestClassifier , RandomForestRegressor ,GradientBoostingRegressor
from sklearn.datasets import make_classification

from xgboost import XGBClassifier 

from scipy.stats import skew, boxcox_normmax
from scipy.special import boxcox1p

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import math as m
import requests


def get_time_columns(dataset):
    dataset['time_entry_seconds'] = 0
    dataset['time_exit_seconds'] = 0
    dataset['time_difference'] = 0
    now = datetime.datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    def one_convert_time_numerical(row):
        row['time_entry_seconds'] = (datetime.datetime.combine(datetime.date.today(), parser.parse(row['time_entry']).time()) - midnight).seconds
        row['time_exit_seconds'] = (datetime.datetime.combine(datetime.date.today(), parser.parse(row['time_exit']).time()) - midnight).seconds
        row['time_difference'] = row['time_exit_seconds']  - row['time_entry_seconds']
        return row
    dataset = dataset.apply(one_convert_time_numerical, axis=1)
    print('Calculated times in seconds and time difference')
    return dataset

def distance_traveled(dataset):
    dataset['distance_traveled']=0
    def one_distance(row):
        row['distance_traveled']= m.sqrt((row['x_entry'] - row['x_exit'])**2 + (row['y_entry'] - row['y_exit'])**2)
        return row 
    dataset = dataset.apply(one_distance,axis=1)
    print('distance between entry and exit calculated')
    return dataset

def moved_into_town(dataset):
    dataset['moved_into_town']=0
    def one_moved_into_town(row):
        temp = (row.entry_in_town == 0 and row.exit_in_town == 1)
        row['moved_into_town']= temp
        return row 
    dataset = dataset.apply(one_moved_into_town,axis=1)
    print('Calculated where trajectory moved device into town')
    return dataset

def moved_outof_town(dataset):
    dataset['moved_outof_town']=0
    def one_moved_outof_town(row):
        temp = (row.entry_in_town == 1 and row.exit_in_town == 0)
        row['moved_outof_town']= temp
        return row 
    dataset = dataset.apply(one_moved_outof_town,axis=1)
    print('Calculated where trajectory moved device out of town')
    return dataset


def calculated_velocity(dataset):
    dataset['calculated_velocity'] = dataset.distance_traveled/dataset.time_difference
    return dataset


def distance_from_city_center(dataset):
    dataset['distance_from_city_center']=0
    def one_distance(row):
        row['distance_from_city_center']= m.sqrt((row['x_entry'] - 3765901.5068)**2 + (row['y_entry'] - -19238905.6133)**2)
        return row 
    dataset = dataset.apply(one_distance,axis=1)
    print('distance from city center')
    return dataset

def calculate_angle(dataset):
    dataset['angle'] = 0
    def one_angle(row):
        rise = row.y_exit - row.y_entry
        run = row.x_exit - row.x_entry   
        degree = math.degrees(math.atan2(rise, run))
        row['angle'] = degree
        return row
    dataset = dataset.apply(one_angle,axis=1)
    print('Calculated angle of trajectory')
    return dataset

def calculate_angle_fromcenter(dataset):
    dataset['angle_from_center'] = 0
    centerX = 3765901.5068
    centerY = -19238905.6133
    def one_angle(row):
        intown = row.entry_in_town
        if intown:
            rise = row.y_entry - centerY
            run = row.x_entry - centerX
        else:
            rise = centerY - row.y_entry
            run = centerX - row.x_entry
        degree = math.degrees(math.atan2(rise, run))
        row['angle_from_center'] = degree
        return row
    dataset = dataset.apply(one_angle,axis=1)
    print('Calculated angle of in relation to city centeer')
    return dataset

def bin_angles(dataset):
    bins = [-180, -157.5, -112.5, -67.5, -22.5, 22.5, 67.5, 112.5, 157.5, 180]
    dataset['angle_bin'] = pd.cut(dataset.angle,bins, labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8'])
    dataset['angle_bin'] = np.where(dataset.angle_bin == '0', '8', dataset.angle_bin)
    return dataset
    

def test_score_newone( x,y,test ):
    
    ### PLease specify the name!!!
    K = KFold(5, random_state  = 6666)
    
    #### Random forest with params from a gridsearch
    RFC = RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
            max_depth=85, max_features='auto', max_leaf_nodes=None,
            min_impurity_decrease=0.0, min_impurity_split=None,
            min_samples_leaf=2, min_samples_split=5,
            min_weight_fraction_leaf=0.0, n_estimators=230, n_jobs=-1,
            oob_score=False, random_state=6666, verbose=0,
            warm_start=False)
                      
    ###Scores from cross val 
    scores = cross_val_score(RFC, x, y,scoring='f1', cv=K)
    print('The average of the cross validation with Random Forest:{}'.format(scores.mean(), scores.std() * 2))
    
    #### Refitting on the whole data 
    RFC.fit(x , y)
    
    ### Predict on our test                 
    predictions = RFC.predict(test)
    
    return scores , predictions , RFC



def intown_intersections(dataset):
    dataset['intersects_intown'] = 0
    def one_intown_intersections(row):
        row['intersects_intown'] = intersects(row['x_entry'], row['y_entry'], row['calculated_distance_last_traj'], 3765901.5068, -19238905.6133, 10000, 60000)
        return row 
    dataset = dataset.apply(one_intown_intersections,axis=1)
    print('Calculated where trajectory intersects town border')
    return dataset

def intersects(circleX, circleY, radius, rectX, rectY, width, height):
    circleDistanceX = abs(circleX - rectX)
    circleDistanceY = abs(circleY - rectY)

    if (circleDistanceX > (width/2 + radius)): return False
    if (circleDistanceY > (height/2 + radius)): return False

    if (circleDistanceX <= (width/2)): return True
    if (circleDistanceY <= (height/2)): return True

    cornerDistance_sq = (circleDistanceX - width/2)**2 + (circleDistanceY- height/2)**2

    return (cornerDistance_sq <= (radius**2))

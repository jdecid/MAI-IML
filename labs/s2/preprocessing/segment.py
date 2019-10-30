#!/usr/bin/env python
# coding: utf-8

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.dataset import read_dataset

import os


def preprocess():
    # Read data
    dataset = read_dataset('segment', dataset_path=os.path.join('..', 'datasets'))
    data = dataset['data']

    df = pd.DataFrame(data)
    df = df.applymap(lambda x: x.decode('utf-8') if isinstance(x, bytes) else x)
    y = df['class'].copy()
    #y = y.applymap(lambda x: x.decode('utf-8'))
    X = df.drop(columns=['class', 'region-pixel-count'])
    # X.head()

    # Normalize
    scaler = MinMaxScaler()
    X[:] = scaler.fit_transform(X)

    X_numerical_as_categorical = X.copy()

    for i in range(X.shape[1]):
        X_numerical_as_categorical.iloc[:, i] = pd.qcut(x=X.iloc[:, i], q=5, duplicates='drop')

    # Write data
    with open(os.path.join('..', 'datasets', 'segment_clean.csv'), mode='w') as f:
        X.to_csv(f, index=False)
    with open(os.path.join('..', 'datasets', 'segment_clean_y.csv'), mode='w') as f:
        y.to_csv(f, index=False)
    with open(os.path.join('..', 'datasets', 'segment_clean_cat.csv'), mode='w') as f:
        X_numerical_as_categorical.to_csv(f, index=False)

    return 'segment_clean.csv', 'segment_clean_cat.csv', 'segment_clean_y.csv'

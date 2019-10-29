#!/usr/bin/env python
# coding: utf-8

# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from utils import read_dataset
import os
import pandas as pd


def preprocess():
    DATASET = 'connect-4'

    data = read_dataset(DATASET)
    df = pd.DataFrame(data['data'])

    # since we are still doing unsupervised methods (clustering), we will ignore labels y
    X = df.loc[:, df.columns != 'class']
    X = X.applymap(lambda x: x.decode('utf-8'))  # encode values as unicode strings instead of bytes
    # For all vars in X, the domain is ['b', 'o', 'x']
    # However, we will check it programatically.
    # Also, even if the dataset is supposed to have no missing values, we will check it as well, just in case.
    X_categories = set([])
    for index, row in X.iterrows():
        for col_val in row:
            X_categories.add(col_val)

    # {'b', 'o', 'x'}, so the domain is confirmed, Also, no missing values,
    # because otherwise would have None or others
    # Recall that: 'x' means that we have a cell with a disk belonging to player 'x',
    # 'o' means that we have a cell with a disk belonging to player 'o', and 'b' means that
    # the cell is empty (blank).

    # Instead of one hot encoding, we will apply label encoder with [-1, 0, 1]. The reason why we will do it
    # this way is that 'x' and 'o' are antagonists, and 'b' is the neutral value. So, there is some kind of natural
    # order. This way, we can avoid the one hot encoding, which would increase the number of columns.
    # Since all the variables have the same domain, we should be consistent with the encoding. For us, 'x'
    # will always be encoded as '-1' and 'o' will always be encoded as '1'.
    # X_encoded = X.apply(LabelEncoder().fit_transform)
    # LabelEncoder works alphabetically and with range [0,n_classes-1],
    # so 'b' will be encoded as 2, 'o' as 1, and 'x' as 0, which is not the intended outcome for us.
    # It has no additional parameters, so we will apply our own encoder:
    def recode(x):
        recode_map = {'x': -1, 'b': 0, 'o': 1}
        return recode_map[x]

    X_encoded = X.applymap(recode)


    # save the cleaned/encoded X as a CSV for later
    X.to_csv(os.path.join('datasets', 'connect-4-clean.csv'), index=False)
    X_encoded.to_csv(os.path.join('datasets', 'connect-4-clean-enc.csv'), index=False)
    return 'connect-4-clean.csv', 'connect-4-clean-enc.csv'



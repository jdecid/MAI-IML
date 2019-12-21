import os
import argparse
from typing import Tuple, List

import numpy as np

from preprocessing.adult import preprocess
from utils.dataset import read_dataset


def read_data(name: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    folds = []
    for i in range(10):
        train_data = read_dataset(name=f'{name}.fold.00000{i}.train', dataset_path=os.path.join('datasets', name))
        validation_data = read_dataset(name=f'{name}.fold.00000{i}.test', dataset_path=os.path.join('datasets', name))

        (X_train, y_train), (X_val, y_val) = preprocess(train_data, validation_data)
        folds.append({
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val
        })

    return folds


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all Clustering project for MAI-IML')
    parser.add_argument('output_path', type=str, default='output', help='Output path for the logs')
    parser.add_argument('--seed', type=int, help='Seed for random behavior reproducibility')

    parser.add_argument('--algorithm', type=str, help='Select algorithm to run',
                        choices=['agglomerative', 'kmeans', 'kmodes', 'kprototypes', 'fcm'])
    parser.add_argument('--dataset', type=str, help='Select dataset to use',
                        choices=['adult'])

    args = parser.parse_args()

    read_data(args.dataset)

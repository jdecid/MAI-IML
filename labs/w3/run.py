import argparse
from typing import Tuple, List

import numpy as np

from algorithms.KIBLAlgorithm import KIBLAlgorithm
from algorithms.reduction_KIBL_algorithm import reduction_KIBL_algorithm, REDUCTION_METHODS
from utils.dataset import read_dataset


def read_data(name: str) -> List[Tuple[np.ndarray, np.ndarray]]:
    folds = []
    for i in range(10):
        train_data = read_dataset(name=f'{name}.fold.00000{i}.train')
        validation_data = read_dataset(name=f'{name}.fold.00000{i}.test')

        folds.append((train_data, validation_data))

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

    X = np.zeros((0, 10))

    for reduction_method in REDUCTION_METHODS:
        U = reduction_KIBL_algorithm(X, reduction_method)
        knn = KIBLAlgorithm(U)

import argparse
import os
from typing import List

from tqdm import tqdm

from algorithms.KIBLAlgorithm import KIBLAlgorithm, VOTING_POLICIES, RETENTION_POLICIES
from preprocessing.adult import preprocess
from utils.dataset import read_dataset


def read_data(name: str) -> List[dict]:
    folds = []
    for i in tqdm(range(10), desc=f'Reading {name} dataset', ncols=150):
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

    folds = read_data(args.dataset)
    for fold in folds:
        for k in [1, 3, 5, 7]:
            for voting_policy in VOTING_POLICIES:
                for retention_policy in RETENTION_POLICIES:
                    alg = KIBLAlgorithm(K=k)
                    alg.fit(fold['X_train'], fold['y_train'])
                    results = alg.k_neighbours(fold['X_val'])

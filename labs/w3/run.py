import argparse
import os
from time import time
from typing import List

from tqdm import tqdm

from algorithms.KIBLAlgorithm import KIBLAlgorithm, VOTING_POLICIES, RETENTION_POLICIES
from preprocessing.adult import preprocess
from utils.dataset import read_dataset

K_VALUES = [1, 3, 5, 7]
R_VALUES = [2]


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


def run_knn(folds):
    t = tqdm(total=len(K_VALUES) * len(VOTING_POLICIES) * len(RETENTION_POLICIES) * len(R_VALUES),
             desc='KNN', ncols=150)

    results = []
    for k in K_VALUES:
        for r in R_VALUES:
            for voting_policy in VOTING_POLICIES:
                for retention_policy in RETENTION_POLICIES:

                    fold_results = []
                    for fold in tqdm(folds, desc='Folds', ncols=150, position=1):
                        time_start = time()

                        alg = KIBLAlgorithm(K=k)
                        alg.fit(fold['X_train'], fold['y_train'])

                        corrects = 0
                        val_data = list(zip(fold['X_val'], fold['y_val']))

                        t_val = tqdm(total=len(val_data), desc='Validation data', ncols=150, position=0, leave=True)
                        for X, y in val_data:
                            prediction = alg.k_neighbours(X, y)
                            if prediction == y:
                                corrects += 1

                            t_val.update()
                        t_val.clear()

                        fold_results.append({
                            'accuracy': corrects / len(val_data),
                            'time': time() - time_start
                        })

                    results.append({
                        'k': k,
                        'vp': voting_policy,
                        'rp': retention_policy,
                        'results': fold_results
                    })

                    t.update()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all Clustering project for MAI-IML')
    parser.add_argument('output_path', type=str, default='output', help='Output path for the logs')
    parser.add_argument('--seed', type=int, help='Seed for random behavior reproducibility')

    parser.add_argument('--algorithm', type=str, help='Select algorithm to run',
                        choices=['agglomerative', 'kmeans', 'kmodes', 'kprototypes', 'fcm'])
    parser.add_argument('--dataset', type=str, help='Select dataset to use',
                        choices=['adult'])

    args = parser.parse_args()

    data = read_data(args.dataset)
    run_knn(folds=data)

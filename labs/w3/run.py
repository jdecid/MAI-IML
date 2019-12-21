import argparse
import json
import multiprocessing
import os
from time import time
from typing import List

from tqdm import tqdm

from algorithms.KIBLAlgorithm import KIBLAlgorithm, VOTING_POLICIES, RETENTION_POLICIES
from preprocessing.adult import preprocess
from utils.dataset import read_dataset

K_VALUES = [1, 3]  # , 5, 7]
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


def run_knn_fold(fold, k, r):
    time_start = time()

    alg = KIBLAlgorithm(K=k, r=r)
    alg.fit(fold['X_train'], fold['y_train'])

    val_data = list(zip(fold['X_val'], fold['y_val']))
    t_val = tqdm(total=len(val_data), desc='Validation data', ncols=150, position=1, leave=True)

    corrects = 0
    for X, y in val_data:
        prediction = alg.k_neighbours(X, y)
        if prediction == y:
            corrects += 1

        t_val.update()
    t_val.clear()

    return {
        'accuracy': corrects / len(val_data),
        'time': time() - time_start
    }


def run_knn(folds, par=False):
    t = tqdm(total=len(K_VALUES) * len(VOTING_POLICIES) * len(RETENTION_POLICIES) * len(R_VALUES),
             desc='KNN', ncols=150)

    cores = min(multiprocessing.cpu_count(), 10)
    pool = multiprocessing.Pool(cores)

    results = []
    for k in K_VALUES:
        for r in R_VALUES:
            for voting_policy in VOTING_POLICIES:
                for retention_policy in RETENTION_POLICIES:
                    if par:
                        fold_results = pool.map(lambda x: run_knn_fold(x, k, r), folds)
                    else:
                        fold_results = list(map(lambda x: run_knn_fold(x, k, r), folds))

                    results.append({
                        'k': k,
                        'vp': voting_policy,
                        'rp': retention_policy,
                        'results': fold_results
                    })

                t.update()

    results_json = json.dumps(results)
    with open('results.json', mode='w') as f:
        f.write(results_json)


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

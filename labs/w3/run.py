import argparse
import json
import multiprocessing
import os
from multiprocessing.pool import ThreadPool
from threading import Lock
from time import time
from typing import List

import numpy as np
from tqdm import tqdm

from algorithms.KIBLAlgorithm import KIBLAlgorithm, VOTING_POLICIES, RETENTION_POLICIES
from preprocessing.hypothyroid import preprocess as preprocess_hypothyroid
from preprocessing.pen_based import preprocess as preprocess_penn
from utils.dataset import read_dataset

OUTPUT_PATH = 'output'

K_VALUES = [1, 3, 5, 7]
R_VALUES = [1, 2, 3]


def read_data(name: str) -> List[dict]:
    folds = []
    preprocess = preprocess_hypothyroid if name == 'hypothyroid' else preprocess_penn

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


def run_knn_fold(fold, k, r, seed, i=None, lock=None):
    np.random.seed(seed)

    time_start = time()

    alg = KIBLAlgorithm(K=k, r=r)
    alg.fit(fold['X_train'], fold['y_train'])
    val_data = list(zip(fold['X_val'], fold['y_val']))

    if lock is not None:
        with lock:
            t_val = tqdm(total=len(val_data), desc=f'Fold {i:2}', ncols=150, position=i)
    else:
        t_val = tqdm(total=len(val_data), desc=f'Fold {i:2}', ncols=150)

    corrects = 0
    for X, y in val_data:
        prediction = alg.k_neighbours(X, y)
        if prediction == y:
            corrects += 1

        if lock is not None:
            with lock:
                t_val.update()
        else:
            t_val.update()

    if lock is not None:
        with lock:
            t_val.close()
    else:
        t_val.close()

    return i, {
        'accuracy': corrects / len(val_data),
        'time': time() - time_start
    }


def run_kIBL(folds, name, seed, par):
    i_experiment = 0
    n_experiments = len(K_VALUES) * len(VOTING_POLICIES) * len(RETENTION_POLICIES) * len(R_VALUES)

    results = []
    for k in K_VALUES:
        for r in R_VALUES:
            for voting_policy in VOTING_POLICIES:
                for retention_policy in RETENTION_POLICIES:
                    i_experiment += 1
                    print('-' * 150)
                    print(f'> Running experiment ({i_experiment}/{n_experiments}): '
                          f'K={k}, r={r}, VP={voting_policy} and RP={retention_policy}' + ' ' * 100)

                    if par:
                        cores = min(multiprocessing.cpu_count(), 5)
                        pool = ThreadPool(cores)
                        lock = Lock()

                        fold_results = {}
                        for i, fold in enumerate(folds):
                            pool.apply_async(run_knn_fold, args=(fold, k, r, seed, i, lock),
                                             callback=lambda x: fold_results.update({x[0]: x[1]}))

                        pool.close()
                        pool.join()

                        fold_results = [fold_results[f] for f in range(len(folds))]
                    else:
                        fold_results = list(map(lambda x: run_knn_fold(x[1], k, r, seed, x[0]), enumerate(folds)))

                    results.append({
                        'k': k,
                        'vp': voting_policy,
                        'rp': retention_policy,
                        'results': fold_results
                    })

    results_json = json.dumps(results)
    with open(os.path.join(OUTPUT_PATH, name + '_results.json'), mode='w') as f:
        f.write(results_json)


def run_stat_select_kIBL(kIBL_json_path, seed):
    pass


def run_reduction_kIBL(folds, kIBL_params, seed):
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Work 3 experiments for MAI-IML')
    parser.add_argument('--seed', type=int, help='Seed for random behavior reproducibility')

    parser.add_argument('--algorithm', type=str, help='Select algorithm to run',
                        choices=['kIBL', 'stat', 'reduction'])
    parser.add_argument('--dataset', type=str, help='Select dataset to use',
                        choices=['hypothyroid', 'pen-based'])
    parser.add_argument('--results_kIBL', type=str, help='JSON with saved kIBL results')
    parser.add_argument('--par', type=str, help='Whether to take advantage of multiprocessing',
                        const=True, default=False, nargs='?')

    args = parser.parse_args()

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    data = read_data(args.dataset)
    if args.algorithm == 'kIBL':
        run_kIBL(folds=data, name=args.dataset, seed=args.seed, par=args.par)
    elif args.algorithm == 'stat':
        run_stat_select_kIBL(args.results_kIBL, seed=args.seed)
    else:
        kIBL_params = {}
        run_reduction_kIBL(folds=data, kIBL_params=kIBL_params, seed=args.seed)

import argparse
import logging
import os
from datetime import datetime
from typing import List, Dict

import pandas as pd

from algorithms.kmeans import KMeans
from algorithms.kprototypes import KPrototypes
from preprocessing import adult, connect_4, segment
from utils import evaluate
from utils.optimize import optimize

from algorithms.agglomerative import agglomerative_clustering


def run_agglomerative(paths: List[Dict[str, str]], args=dict):
    logging.info('Running Agglomerative experiments')

    for path in paths:
        X = pd.read_csv(os.path.join('datasets', path['X']))
        Y = pd.read_csv(os.path.join('datasets', path['Y']), header=None)

        # Instead of optimizing the number of clusters, this time we are going to test other parameter as suggested
        # in the assignment. In particular, we are going to experiment with different affinities and linkages
        n_classes = len(Y[Y.columns[0]].unique())
        results = agglomerative_clustering(X=X, K=n_classes, fig_save_path=args.output_path)

        for result in results:
            print(f'Results with affinity {result["affinity"]} and linkage {result["linkage"]}')
            # Supervised evaluation (we are using k = # classes)
            res = evaluate.evaluate_supervised(labels_true=Y.values.flatten(), labels_pred=result['prediction'])
            print(res)
            # Unsupervised
            res = evaluate.evaluate_unsupervised(X=X, labels=result['prediction'])
            print(res)


def run_kmeans(paths: List[Dict[str, str]], args=dict):
    logging.info('Running K-Means experiments')

    for path in paths:
        X = pd.read_csv(os.path.join('datasets', path['X']))
        Y = pd.read_csv(os.path.join('datasets', path['Y']), header=None)

        # Optimization of K

        alg_params = {'name': path['name'], 'vis_dims': 2, 'fig_save_path': args.output_path}
        results = optimize(X=X.values,
                           algorithm=KMeans,
                           algorithm_params=alg_params,
                           metric='calinski_harabasz_score',
                           metric_params={'X': X.values},
                           k_values=list(range(2, 10)),
                           goal='minimize')

        # With best k: unsupervised (supervised generally not possible unless best_k = n_classes
        res = evaluate.evaluate_unsupervised(X=X, labels=results[0]['prediction'])
        print(res)

        # With k = n_classes
        n_classes = len(Y[Y.columns[0]].unique())
        real_k = list(filter(lambda r: r['k'] == n_classes, results))[0]
        res = evaluate.evaluate_supervised(labels_true=Y.values.flatten(), labels_pred=real_k['prediction'])
        print(res)


def run_kprototypes(paths: List[Dict[str, str]], args=dict):
    logging.info('Running K-Prototypes experiments')

    for path in paths:
        X = pd.read_csv(os.path.join('datasets', path['X']))
        Y = pd.read_csv(os.path.join('datasets', path['Y']), header=None)

        # Optimization of K

        alg_params = {'name': path['name'], 'vis_dims': 2, 'fig_save_path': args.output_path}
        results = optimize(X=X.values,
                           algorithm=KPrototypes,
                           algorithm_params=alg_params,
                           metric='calinski_harabasz_score',
                           metric_params={'X': X.values},
                           k_values=list(range(2, 10)),
                           goal='minimize')

        # With best k: unsupervised (supervised generally not possible unless best_k = n_classes
        res = evaluate.evaluate_unsupervised(X=X, labels=results[0]['prediction'])
        print(res)

        # With k = n_classes
        n_classes = len(Y[Y.columns[0]].unique())
        real_k = list(filter(lambda r: r['k'] == n_classes, results))[0]
        res = evaluate.evaluate_supervised(labels_true=Y.values.flatten(), labels_pred=real_k['prediction'])
        print(res)


def main(args):
    """Runs EVERYTHING (preprocessing, clustering, evaluation,...), saves images, logs, results etc. for the report"""
    print('Preprocessing...')
    file_segment_num, file_segment_cat, file_segment_y = segment.preprocess()
    file_adult_num, file_adult_cat, file_adult_mix, file_adult_y = adult.preprocess()
    file_connect_4, file_connect_4_num, file_connect_4_y = connect_4.preprocess()

    run_agglomerative(paths=[
        {'name': 'segment', 'X': file_segment_num, 'Y': file_segment_y},
        # {'name': 'adult', 'X': file_adult_num, 'Y': file_adult_y},
        # {'name': 'connect_4', 'X': file_connect_4_num, 'Y': file_connect_4_y}
    ], args=args)

    run_kmeans(paths=[
        {'name': 'segment', 'X': file_segment_num, 'Y': file_segment_y},
        {'name': 'adult', 'X': file_adult_num, 'Y': file_adult_y},
        {'name': 'connect_4', 'X': file_connect_4_num, 'Y': file_connect_4_y}
    ], args=args)

    run_kprototypes(paths=[
        {'name': 'segment', 'X': file_segment_num, 'Y': file_segment_y},
        {'name': 'adult', 'X': file_adult_mix, 'Y': file_adult_y},
        {'name': 'connect_4', 'X': file_connect_4, 'Y': file_connect_4_y}
    ], args=args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all Clustering project for MAI-IML')
    parser.add_argument('output_path', type=str, default='output', help='Output path for the logs')
    parser.add_argument('--seed', type=int, help='Seed for random behavior reproducibility')

    args = parser.parse_args()

    # Use timestamp as log file name
    current_time = datetime.now()
    log_name = str(current_time.date()) + '_' + str(current_time.timetz())[:-7]
    log_folder = os.path.join(args.output_path, log_name)

    os.mkdir(log_folder, mode=0o777)
    logging.basicConfig(filename=os.path.join(log_folder, 'log.txt'), level=logging.DEBUG)

    # Disable INFO and DEBUG logging for Matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    args.output_path = log_folder
    main(args)

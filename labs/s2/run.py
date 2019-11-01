import argparse
import logging
import os
from datetime import datetime
from typing import List, Dict

import pandas as pd

from algorithms.kmeans import KMeans
from preprocessing import adult, connect_4, segment
from utils import evaluate
from utils.optimize import optimize


def run_kmeans(paths: List[Dict[str, str]], args=dict):
    logging.info('Running KMeans experiments')

    for path in paths:
        X = pd.read_csv(os.path.join('datasets', path['X']))
        Y = pd.read_csv(os.path.join('datasets', path['Y']))

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


def main(args):
    """Runs EVERYTHING (preprocessing, clustering, evaluation,...), saves images, logs, results etc. for the report"""
    print('Preprocessing...')
    # file_c4_cat, file_c4_num = connect_4.preprocess()
    file_segment_num, file_segment_cat, file_segment_y = segment.preprocess()
    # file_adult, file_adult_cat, file_adult_num = adult.preprocess()

    print('Applying hierarchical clustering...')
    # TODO

    run_kmeans(paths=[
        {'name': 'segment', 'X': file_segment_num, 'Y': file_segment_y}
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

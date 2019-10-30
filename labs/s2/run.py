import argparse
import logging
import os

import pandas as pd

from algorithms.kmeans import KMeans
from preprocessing import adult
from utils import evaluate
from utils.optimize import optimize

from datetime import datetime


def main(args):
    """Runs EVERYTHING (preprocessing, clustering, evaluation,...), saves images, logs, results etc. for the report"""
    print('Preprocessing...')
    # file_c4, file_c4_enc = connect_4.preprocess()
    file_adult, file_adult_enc, file_adult_y = adult.preprocess()
    # file_segment = segment.preprocess()

    print('Applying hierarchical clustering...')
    # TODO

    # ------------------------------------------------------ #

    print('KMeans')

    X = pd.read_csv(os.path.join('datasets', file_adult_enc))
    y = pd.read_csv(os.path.join('datasets', file_adult_y), header=None).values.flatten()
    n_classes = len(set(list(y)))

    # kmeans = KMeans(K=2, vis_dims=2)
    # kmeans.fit_predict(X.values)

    # Optimization of K

    result = optimize(X=X.values,
                      algorithm=KMeans,
                      algorithm_params={},
                      metric='calinski_harabasz_score',
                      metric_params={'X': X.values},
                      k_values=[2, 3, 4, 5],
                      goal='minimize')

    # Evaluate
    # With best k: unsupervised (supervised generally not possible unless best_k = n_classes
    result_best_k = result[0]
    print(evaluate.evaluate_unsupervised(X=X, labels=result_best_k['prediction']))
    # With k = n_classes
    for res in result:
        if res['k'] == n_classes:
            print()
            print(evaluate.evaluate_supervised(labels_true=y, labels_pred=res['prediction']))
            break

    # ------------------------------------------------------ #


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all Clustering project for MAI-IML')
    parser.add_argument('output_path', type=str, default='output', help='Output path for the logs')
    parser.add_argument('--seed', type=int, help='Seed for random behavior reproducibility')

    args = parser.parse_args()

    # Use timestamp as log file name
    current_time = datetime.now()
    log_file = str(current_time.date()) + '_' + str(current_time.timetz())[:-7]
    logging.basicConfig(filename=os.path.join(args.output_path, f'{log_file}.log'), level=logging.DEBUG)

    # Disable INFO and DEBUG logging for Matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    main(args)

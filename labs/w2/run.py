import argparse
import logging
import os
from datetime import datetime
from typing import List, Dict

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA

from algorithms.pca import PCA as IML_PCA
from preprocessing import adult, connect_4, segment


def run_pca(paths: List[Dict[str, str]], n_components: List[int], params):
    for path in paths:
        X = pd.read_csv(os.path.join('datasets', path['X'])).values
        # Y = pd.read_csv(os.path.join('datasets', path['Y']), header=None)

        for n in n_components:
            f, ax = plt.subplots(1, 3, figsize=(10, 3))
            f.tight_layout()

            # Our PCA
            pca = IML_PCA(n_components=n, name=path['name'], fig_save_path=params.output_path)
            results_pca = pca.fit_transform(X)

            ax[0].set_title('Custom PCA')
            ax[0].scatter(results_pca[:, 0], results_pca[:, 1], c='darkred', s=10, alpha=0.5)

            # PCA
            pca = PCA(n_components=n)
            results_pca = pca.fit_transform(X)

            ax[1].set_title('SKLearn PCA')
            ax[1].scatter(results_pca[:, 0], results_pca[:, 1], c='darkblue', s=10, alpha=0.5)

            # Incremental PCA
            ipca = IncrementalPCA(n_components=n)
            results_ipca = ipca.fit_transform(X)

            ax[2].set_title('SKLearn Incremental PCA')
            ax[2].scatter(results_ipca[:, 0], results_ipca[:, 1], c='teal', s=10, alpha=0.5)

            f.savefig(os.path.join(params.output_path, f'pca_comparative_{path["name"]}.png'))
            plt.close(f)


def main(params):
    datasets = []
    if params.dataset == 'adult' or params.dataset is None:
        file_adult_num, file_adult_cat, file_adult_mix, file_adult_y = adult.preprocess()
        datasets += [
            {'name': 'adult', 'X': file_adult_num, 'Y': file_adult_y, 'type': 'num'},
            {'name': 'adult', 'X': file_adult_cat, 'Y': file_adult_y, 'type': 'cat'},
            {'name': 'adult', 'X': file_adult_mix, 'Y': file_adult_y, 'type': 'mix'},
        ]

    if params.dataset == 'connect-4' or params.dataset is None:
        file_connect_4_cat, file_connect_4_num, file_connect_4_y = connect_4.preprocess()
        datasets += [
            {'name': 'connect-4', 'X': file_connect_4_num, 'Y': file_connect_4_y, 'type': 'num'},
            {'name': 'connect-4', 'X': file_connect_4_cat, 'Y': file_connect_4_y, 'type': 'cat'},
            {'name': 'connect-4', 'X': file_connect_4_cat, 'Y': file_connect_4_y, 'type': 'mix'}
        ]

    if params.dataset == 'segment' or params.dataset is None:
        file_segment_num, file_segment_cat, file_segment_y = segment.preprocess()
        datasets += [
            {'name': 'segment', 'X': file_segment_num, 'Y': file_segment_y, 'type': 'num'},
            {'name': 'segment', 'X': file_segment_cat, 'Y': file_segment_y, 'type': 'cat'},
            {'name': 'segment', 'X': file_segment_num, 'Y': file_segment_y, 'type': 'mix'},
        ]

    num_paths = list(filter(lambda d: d['type'] == 'num', datasets))
    run_pca(paths=num_paths, n_components=[2], params=params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all PCA + SOM project of MAI-IML')
    parser.add_argument('output_path', type=str, default='output', help='Output path for the logs')
    parser.add_argument('--seed', type=int, help='Seed for random behavior reproducibility')

    parser.add_argument('--algorithm', type=str, help='Select algorithm to run',
                        choices=['PCA', 'SOM'])
    parser.add_argument('--dataset', type=str, help='Select dataset to use',
                        choices=['adult', 'connect-4', 'segment'])

    args = parser.parse_args()

    print('+' + '-' * 45 + '+')
    print(f'| > Results will be stored in {args.output_path.upper()}')
    print(f'| > Seed: {args.seed}')
    print(f'| > Algorithms: {args.algorithm if args.algorithm is not None else "ALL"}')
    print(f'| > Datasets: {args.dataset if args.dataset is not None else "ALL"}')
    print('+' + '-' * 45 + '+')

    # Use timestamp as log file name
    current_time = datetime.now()
    log_name = str(current_time.date()) + '_' + (str(current_time.timetz())[:-7]).replace(':', '-')
    log_folder = os.path.join(args.output_path, log_name)

    os.makedirs(log_folder, mode=0o777)
    logging.basicConfig(filename=os.path.join(log_folder, 'log.txt'), level=logging.DEBUG)

    # Disable INFO and DEBUG logging for Matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    args.output_path = log_folder
    main(params=args)

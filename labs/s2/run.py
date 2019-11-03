import argparse
import logging
import os
import termtables as tt
from datetime import datetime
from typing import List, Dict

import pandas as pd
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from algorithms.agglomerative import agglomerative_clustering
from algorithms.fcm import FuzzyCMeans
from algorithms.kmeans import KMeans
from algorithms.kmodes import KModes
from algorithms.kprototypes import KPrototypes
from preprocessing import adult, connect_4, segment
from utils import evaluate
from utils.optimize import optimize

metrics = {
    'calinski_harabasz_score': calinski_harabasz_score,
    'davies_bouldin_score': davies_bouldin_score,
    'silhouette_score': silhouette_score
}


# TODO: prints -> logs, or tables, or something, PRETTIFY somehow

def optimize_dict_to_table(results):
    return tt.to_string(
        list(map(lambda x: [x['k'], f'{x["score"]:.6f}'], results)),
        header=['K', 'Score'],
        style=tt.styles.thin_double,
        padding=(0, 1),
        alignment='cr'
    )


def eval_dict_to_table(res):
    table = 'metric score'
    for metric, result in res.items():
        if metric != 'contingency_matrix':
            table += f'{metric} {result}\n'

    if 'contingency_matrix' in res:
        table += '\n contingency_matrix\n'
        table += str(res['contingency_matrix']) + '\n'
    return table


def run_agglomerative(paths: List[Dict[str, str]], params):
    # TODO: SUBSET of connect-4, otherwise memory error! Also, save this subset because the professor wants to inspect t
    message = 'Running Agglomerative experiments'
    print(message + '...')
    logging.info(message)
    results_to_save = 'Agglomerative experiments results\n'
    results_to_save += 'Except the number of clusters, affinity and linkage, the other parameters are the default ones.'
    results_to_save += 'We will set # clusters to # classes'
    for path in paths:
        results_to_save += f'{path["name"]} dataset\n'
        X = pd.read_csv(os.path.join('datasets', path['X']))
        Y = pd.read_csv(os.path.join('datasets', path['Y']), header=None)

        # Instead of optimizing the number of clusters, this time we are going to test other parameter as suggested
        # in the assignment. In particular, we are going to experiment with different affinities and linkages
        n_classes = len(Y[Y.columns[0]].unique())
        results = agglomerative_clustering(X=X, K=n_classes, name=path['name'], fig_save_path=params.output_path)

        for result in results:
            results_to_save += f'Results with affinity {result["affinity"]} and linkage {result["linkage"]}\n'
            # Supervised evaluation (we are using k = # classes)
            res = evaluate.evaluate_supervised(labels_true=Y.values.flatten(), labels_pred=result['prediction'])
            results_to_save += f'Supervised evaluation: {eval_dict_to_table(res)}\n'
            # Unsupervised
            res = evaluate.evaluate_unsupervised(X=X, labels=result['prediction'])
            results_to_save += f'Unsupervised evaluation: {eval_dict_to_table(res)}\n'
    with open(os.path.join(params.output_path, 'results.txt'), 'a') as f:
        f.write(results_to_save)


def run_kmeans(paths: List[Dict[str, str]], params):
    message = 'Running K-Means experiments'
    print(message + '...')
    logging.info(message)

    results_to_save = 'K-Means experiments results\n'
    results_to_save += 'Except K, the other parameters are the default ones (eg. euclidean distance)\n'
    for path in paths:
        results_to_save += f'{path["name"]} dataset\n'
        X = pd.read_csv(os.path.join('datasets', path['X']))
        Y = pd.read_csv(os.path.join('datasets', path['Y']), header=None)

        # Optimization of K

        alg_params = {'name': path['name'], 'vis_dims': 2, 'fig_save_path': params.output_path}
        results = optimize(X=X.values,
                           algorithm=KMeans,
                           algorithm_params=alg_params,
                           metric='calinski_harabasz_score',
                           metric_params={'X': X.values},
                           k_values=list(range(2, 10)),
                           goal='minimize')

        results_to_save += f'Optimization of K with calinski_harabasz_score:\n{optimize_dict_to_table(results)}\n'

        # With best k: unsupervised (supervised generally not possible unless best_k = n_classes
        res = evaluate.evaluate_unsupervised(X=X, labels=results[0]['prediction'])
        results_to_save += f'Unsupervised evaluation of the clustering with the best K ({results[0]["k"]}):\n'
        results_to_save += f'{eval_dict_to_table(res)}\n'

        # With k = n_classes
        n_classes = len(Y[Y.columns[0]].unique())
        real_k = list(filter(lambda r: r['k'] == n_classes, results))[0]
        res = evaluate.evaluate_supervised(labels_true=Y.values.flatten(), labels_pred=real_k['prediction'])
        results_to_save += f'Unsupervised evaluation of the clustering with K = # classes({real_k["k"]}):\n'
        results_to_save += f'{eval_dict_to_table(res)}\n'

    with open(os.path.join(params.output_path, 'results.txt'), 'a') as f:
        f.write(results_to_save)


# TODO: isn't run_kmodes extremely slow?
def run_kmodes(paths: List[Dict[str, str]], params):
    message = 'Running KModes experiments'
    print(message + '...')
    logging.info(message)

    results_to_save = 'K-Modes experiments results\n'
    for path in paths:
        results_to_save += f'{path["name"]} dataset\n'
        X = pd.read_csv(os.path.join('datasets', path['X']))
        Y = pd.read_csv(os.path.join('datasets', path['Y']), header=None)

        # Optimization of K

        alg_params = {'name': path['name'], 'fig_save_path': params.output_path}
        alg = KModes(K=1, **alg_params)
        precomputed_distances = alg.compute_point_wise_distances(X.values)
        results = optimize(X=X.values,
                           algorithm=KModes,
                           algorithm_params=alg_params,
                           metric='silhouette_score',
                           metric_params={'metric': 'precomputed'},
                           k_values=[2],  # list(range(2, 10)),
                           goal='minimize',
                           precomputed_distances=precomputed_distances)
        results_to_save += f'Optimization of K with silhouette_score:\n{optimize_dict_to_table(results)}\n'
        # With best k: unsupervised (supervised generally not possible unless best_k = n_classes
        # Only silhouette_score
        results_to_save += f'silhouette_score with best K ({results[0]["k"]}) = {results[0]["score"]}\n'

        # With k = n_classes
        n_classes = len(Y[Y.columns[0]].unique())
        real_k = list(filter(lambda r: r['k'] == n_classes, results))[0]
        res = evaluate.evaluate_supervised(labels_true=Y.values.flatten(), labels_pred=real_k['prediction'][0])
        results_to_save += f'Supervised evaluation of the clustering with K = # classes({real_k}):\n'
        results_to_save += f'{eval_dict_to_table(res)}\n'

    with open(os.path.join(params.output_path, 'results.txt'), 'a') as f:
        f.write(results_to_save)


def run_kprototypes(paths: List[Dict[str, str]], params):
    message = 'Running K-Prototypes experiments'
    print(message + '...')
    logging.info(message)

    results_to_save = 'K-Prototypes experiments results\n'
    for path in paths:
        results_to_save += f'{path["name"]} dataset\n'
        X = pd.read_csv(os.path.join('datasets', path['X']))
        Y = pd.read_csv(os.path.join('datasets', path['Y']), header=None)

        # Optimization of K

        alg_params = {'name': path['name'], 'fig_save_path': params.output_path}
        alg = KPrototypes(K=1, **alg_params)
        precomputed_distances = alg.compute_point_wise_distances(X.values)
        results = optimize(X=X.values,
                           algorithm=KPrototypes,
                           algorithm_params=alg_params,
                           metric='silhouette_score',
                           metric_params={'metric': 'precomputed'},
                           k_values=[2],  # list(range(2, 10)),
                           goal='minimize',
                           precomputed_distances=precomputed_distances)
        results_to_save += f'Optimization of K with silhouette_score:\n{optimize_dict_to_table(results)}\n'
        # With best k: unsupervised (supervised generally not possible unless best_k = n_classes
        # Only silhouette_score

        results_to_save += f'silhouette_score with best K ({results[0]["k"]}) = {results[0]["score"]}\n'

        # With k = n_classes
        n_classes = len(Y[Y.columns[0]].unique())
        real_k = list(filter(lambda r: r['k'] == n_classes, results))[0]
        res = evaluate.evaluate_supervised(labels_true=Y.values.flatten(), labels_pred=real_k['prediction'][0])
        results_to_save += f'Supervised evaluation of the clustering with K = # classes({real_k}):\n'
        results_to_save += f'{eval_dict_to_table(res)}\n'


def run_fcm(paths: List[Dict[str, str]], params):
    message = 'Running Fuzzy C-Means experiments'
    print(message + '...')
    logging.info(message)

    for path in paths:
        X = pd.read_csv(os.path.join('datasets', path['X']))
        Y = pd.read_csv(os.path.join('datasets', path['Y']), header=None)

        # Optimization of C

        alg_params = {'name': path['name'], 'vis_dims': 2, 'fig_save_path': params.output_path}
        results = optimize(X=X.values,
                           algorithm=FuzzyCMeans,
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
        ]

    if params.dataset == 'segment' or params.dataset is None:
        file_segment_num, file_segment_cat, file_segment_y = segment.preprocess()
        datasets += [
            {'name': 'segment', 'X': file_segment_num, 'Y': file_segment_y, 'type': 'num'},
            {'name': 'segment', 'X': file_segment_cat, 'Y': file_segment_y, 'type': 'cat'},
        ]

    num_paths = list(filter(lambda d: d['type'] == 'num', datasets))
    cat_paths = list(filter(lambda d: d['type'] == 'cat', datasets))

    if params.algorithm == 'agglomerative' or params.algorithm is None:
        run_agglomerative(paths=num_paths, params=params)

    if params.algorithm == 'kmeans' or params.algorithm is None:
        run_kmeans(paths=num_paths, params=params)

    if params.algorithm == 'kmodes' or params.algorithm is None:
        run_kmodes(paths=cat_paths, params=params)

    if params.algorithm == 'kprototypes' or params.algorithm is None:
        run_kprototypes(paths=datasets, params=params)

    if params.algorithm == 'fcm' or params.algorithm is None:
        run_fcm(paths=num_paths, params=params)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run all Clustering project for MAI-IML')
    parser.add_argument('output_path', type=str, default='output', help='Output path for the logs')
    parser.add_argument('--seed', type=int, help='Seed for random behavior reproducibility')

    parser.add_argument('--algorithm', type=str, help='Select algorithm to run',
                        choices=['agglomerative', 'kmeans', 'kmodes', 'kprototypes', 'fcm'])
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
    log_name = str(current_time.date()) + '_' + str(current_time.timetz())[:-7]
    log_folder = os.path.join(args.output_path, log_name)

    os.mkdir(log_folder, mode=0o777)
    logging.basicConfig(filename=os.path.join(log_folder, 'log.txt'), level=logging.DEBUG)

    # Disable INFO and DEBUG logging for Matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    args.output_path = log_folder
    main(args)

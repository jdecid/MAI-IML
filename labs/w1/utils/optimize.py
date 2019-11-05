import os
import pickle
import logging
import matplotlib.pyplot as plt
from typing import Type, List

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from tqdm import tqdm

from algorithms.kmeans import KMeans
from utils.evaluate import partition_entropy, normalized_partition_coefficient, xie_beni

metrics = {
    'calinski_harabasz_score': calinski_harabasz_score,
    'davies_bouldin_score': davies_bouldin_score,
    'silhouette_score': silhouette_score,
    'normalized_partition_coefficient': normalized_partition_coefficient,
    'partition_entropy': partition_entropy,
    'xie_beni': xie_beni
}


def plot_k_metrics(data, alg_name: str, alg_params: dict, metric: str):
    plt.figure()

    plt.title(f'Evolution of {metrics[metric].__name__} for K')
    plt.xlabel('K')
    plt.ylabel('Score')

    plt.plot(
        list(map(lambda x: x['k'], data)),
        list(map(lambda x: x['score'], data))
    )

    if alg_params['fig_save_path'] is None:
        plt.show()
    else:
        directory = os.path.join(alg_params['fig_save_path'], alg_name)
        if not os.path.exists(directory):
            os.mkdir(directory)
        plt.savefig(os.path.join(directory, f'evolution_K_{alg_params["name"]}.png'))
    plt.close()


def store_predictions(predictions, alg_name, dataset_name, k, fig_save_path):
    if fig_save_path is not None:
        directory = os.path.join(fig_save_path, alg_name)
        if not os.path.exists(directory):
            os.mkdir(directory)

        with open(os.path.join(directory, f'prediction_{dataset_name}_K{k}.pkl'), mode='wb') as f:
            pickle.dump(predictions, f)


def optimize(X: np.ndarray,
             algorithm: Type[KMeans], algorithm_params: dict, metric: str, metric_params: dict, k_values: List[int],
             goal: str, precomputed_distances: np.ndarray = None) -> List[dict]:
    """
    Optimize K value for the same data, algorithm and metric.
    :param X: 2D data matrix of size (#observations, #features).
    :param algorithm: Algorithm class to instantiate.
    :param algorithm_params: Extra parameters for the algorithm class.
    :param metric: Metric function used to evaluate.
    :param metric_params: Extra parameters for the metric function.
    :param k_values: List of `K` values to test.
    :param goal: `maximize` or `minimize` the metric.
    :return: List sorted from best to worst K value also containing metric score and predictions obtained.
    """
    assert goal in ['maximize', 'minimize']

    logging.info(f'Optimizing {algorithm.__name__} with {metric} for K in {k_values}')

    executions = []

    if precomputed_distances is not None:
        metric_params['X'] = precomputed_distances

    for k in tqdm(k_values, ncols=100):
        logging.info(f'Optimizing K = {k}')

        alg = algorithm(K=k, **algorithm_params)

        if metric in ['normalized_partition_coefficient', 'partition_entropy', 'xie_beni']:
            prediction, v = alg.fit_predict(X)
            crisp_prediction = alg.crisp_predict(X)

            if metric == 'xie_beni':
                score = metrics[metric](X=X, u=prediction, centroids=v)
            else:
                score = metrics[metric](u=prediction)

            executions.append({
                'k': k,
                'score': score,
                'prediction': crisp_prediction,
                'fuzzy_prediction': prediction,
                'centroids': v
            })
        else:
            prediction = alg.fit_predict(X)
            metric_params['labels'] = prediction
            score = metrics[metric](**metric_params)

            executions.append({
                'k': k,
                'score': score,
                'prediction': prediction
            })

        store_predictions(prediction, algorithm.__name__, alg.name, k, algorithm_params['fig_save_path'])

    plot_k_metrics(executions, algorithm.__name__, algorithm_params, metric)

    return sorted(executions,
                  key=lambda x: x['score'],
                  reverse=goal == 'maximize')

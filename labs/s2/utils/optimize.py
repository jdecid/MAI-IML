import os
import logging
import matplotlib.pyplot as plt
from typing import Type, List

import numpy as np
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score

from algorithms.kmeans import KMeans

metrics = {
    'calinski_harabasz_score': calinski_harabasz_score,
    'davies_bouldin_score': davies_bouldin_score,
    'silhouette_score': silhouette_score
}


def plot_k_metrics(data, alg_name: str, alg_params: dict, metric: str):
    plt.figure()

    plt.title('Scores for different K')
    plt.xlabel('K')
    plt.ylabel(metrics[metric].__name__)

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
        plt.savefig(os.path.join(directory, f'{alg_name}_K_evolution.png'))
    plt.close()


def optimize(X: np.ndarray,
             algorithm: Type[KMeans], algorithm_params: dict,
             metric: str, metric_params: dict, k_values: List[int], goal: str,
             precomputed_distances: bool = False) -> List[dict]:
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
    for k in k_values:
        logging.info(f'Optimizing K = {k}')

        alg = algorithm(K=k, **algorithm_params)

        if precomputed_distances:
            prediction, distances = alg.fit_predict(X)
            metric_params['X'] = distances
        else:
            prediction = alg.fit_predict(X)

        metric_params['labels'] = prediction
        score = metrics[metric](**metric_params)

        executions.append({
            'k': k,
            'score': score,
            'prediction': prediction
        })

    plot_k_metrics(executions, algorithm.__name__, algorithm_params, metric)

    return sorted(executions,
                  key=lambda x: x['score'],
                  reverse=goal == 'maximize')

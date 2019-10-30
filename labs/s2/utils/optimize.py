import logging
from typing import Type, List

import numpy as np
from sklearn.metrics import calinski_harabasz_score

from algorithms.kmeans import KMeans

metrics = {
    'calinski_harabasz_score': calinski_harabasz_score
}


def optimize(X: np.ndarray,
             algorithm: Type[KMeans], algorithm_params: dict,
             metric: str, metric_params: dict,
             k_values: List[int], goal: str) -> List[dict]:
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
        prediction = alg.fit_predict(X)

        metric_params['labels'] = prediction
        score = metrics[metric](**metric_params)

        executions.append({
            'k': k,
            'score': score,
            'prediction': prediction
        })

    return sorted(executions,
                  key=lambda x: x['score'],
                  reverse=goal == 'maximize')

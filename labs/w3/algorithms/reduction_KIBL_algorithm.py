import logging
from typing import Tuple, List, Set

import numpy as np

from algorithms.KIBLAlgorithm import KIBLAlgorithm
from algorithms.reduction.cnn import __cnn_reduction
from algorithms.reduction.drop import __drop1_reduction
from algorithms.reduction.ib3 import __ib3_reduction
from algorithms.reduction.renn import __renn_reduction

REDUCTION_METHODS = ['CNN', 'RENN', 'IB3', 'DROP1', 'DROP2']


def reduction_KIBL_algorithm(config: dict, X: np.ndarray, y: np.ndarray, reduction_method: str, seed: int):
    if reduction_method not in REDUCTION_METHODS:
        raise ValueError(f'Unknown reduction method {reduction_KIBL_algorithm()}')

    X = X.copy()
    y = y.copy()
    alg = KIBLAlgorithm(**config)
    np.random.seed(seed)

    if reduction_method == 'CNN':
        return __cnn_reduction(alg, X, y)
    elif reduction_method == 'RENN':
        return __renn_reduction(alg, X, y)
    elif reduction_method == 'IB3':
        return __ib3_reduction(alg, X, y)
    elif reduction_method == 'DROP1':
        return __drop1_reduction(alg, X, y)
    elif reduction_method == 'DROP2':
        return __drop1_reduction(alg, X, y, v2=True)

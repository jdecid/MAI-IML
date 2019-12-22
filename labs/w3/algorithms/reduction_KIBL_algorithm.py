from typing import Tuple

import numpy as np

from algorithms.KIBLAlgorithm import KIBLAlgorithm

REDUCTION_METHODS = ['CNN', 'RENN', 'IB3', 'DROP2', 'DROP3']


def __cnn_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = X.copy()
    unique_labels = set(y)

    U, V = [], []

    for label in unique_labels:
        i = np.random.choice(np.argwhere(y == label).reshape(-1))
        U.append(X[i, :])
        V.append(y[i])

    U = np.array(U)
    V = np.array(V)

    while True:
        for i in range(X.shape[1]):
            knn.fit(np.array(U), np.array(V))
            pred = knn.k_neighbours(X[i, :], y[i])
            if pred != y[i]:
                break
        else:
            U = np.vstack((U, X[i, :]))
            V = np.concatenate((V, y[i]))
            X = np.delete(X, i, axis=0)
            continue
        break

    return np.stack(U), np.array(V)


def __renn_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    U = X.copy()
    V = y.copy()

    while True:
        for i in range(U.shape[1]):
            knn.fit(U, V)
            pred = knn.k_neighbours(U[i, :], V[i])
            if pred != y[i]:
                break
        else:
            U = np.delete(U, i, axis=0)
            V = np.delete(V, i, axis=0)
            continue
        break

    return np.stack(U), np.array(V)


def reduction_KIBL_algorithm(config: dict, X: np.ndarray, y: np.ndarray, reduction_method: str, seed: int):
    if reduction_method not in REDUCTION_METHODS:
        raise ValueError(f'Unknown reduction method {reduction_KIBL_algorithm()}')

    np.random.seed(seed)

    alg = KIBLAlgorithm(**config)

    if reduction_method == 'CNN':
        return __cnn_reduction(alg, X, y)
    elif reduction_method == 'RENN':
        return __renn_reduction(alg, X, y)

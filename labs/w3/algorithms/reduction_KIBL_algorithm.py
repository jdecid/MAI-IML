from typing import Tuple

import numpy as np

from algorithms.KIBLAlgorithm import KIBLAlgorithm

REDUCTION_METHODS = ['CNN', 'RENN', 'IB3'] + [f'DROP{i}' for i in range(1, 6)]


def __cnn_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    X = X.copy()
    unique_labels = set(y)

    U, V = [], []
    for label in unique_labels:
        index = np.random.choice(np.argwhere(y == label))
        U.append(X[index, :])
        U.append(y[index])

    while True:
        for i in range(X.shape[1]):
            knn.fit(U, V)
            pred = knn.k_neighbours(X[i, :], y[i])
            if pred != y[i]:
                break
        else:
            U.append(X[i, :])
            V.append(y[i])
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


def reduction_KIBL_algorithm(X: np.ndarray, reduction_method: str):
    if reduction_method not in REDUCTION_METHODS:
        raise ValueError(f'Unknown reduction method {reduction_KIBL_algorithm()}')

    if reduction_method == 'CNN':
        __cnn_reduction(X)

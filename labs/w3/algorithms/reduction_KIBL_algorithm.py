from typing import Tuple

import numpy as np

from algorithms.KIBLAlgorithm import KIBLAlgorithm

REDUCTION_METHODS = ['CNN', 'RENN', 'IB3', 'DROP2', 'DROP3']


def __cnn_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    U, V = [], []

    # Random initialization with one random element for each different class
    for label in set(y):
        i = np.random.choice(np.argwhere(y == label).reshape(-1))
        # Add to new sets
        U.append(X[i, :])
        V.append(y[i])
        # Remove from new sets
        X = np.delete(X, i, axis=0)
        y = np.delete(y, i)

    U = np.array(U)
    V = np.array(V)
    knn.fit(U, V)

    while True:
        indices_to_remove = []
        for i in range(X.shape[0]):
            pred = knn.k_neighbours(X[i, :], y[i])
            if pred != y[i]:
                indices_to_remove.append(i)

        if len(indices_to_remove):
            # Update U and V if prediction is wrong and fit the model
            U = np.vstack((U, X[indices_to_remove, :]))
            V = np.concatenate((V, y[indices_to_remove]))
            knn.fit(U, V)

            # Remove instance from X
            X = np.delete(X, indices_to_remove, axis=0)
            y = np.delete(y, indices_to_remove)
        else:
            break

    return U, V


def __renn_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    knn.fit(X, y)

    while True:
        indices_to_remove = []
        for i in range(X.shape[0]):
            pred = knn.k_neighbours(X[i, :], y[i])
            if pred != y[i]:
                indices_to_remove.append(i)

        if len(indices_to_remove):
            # Remove instance from X and fit the model
            X = np.delete(X, indices_to_remove, axis=0)
            y = np.delete(y, indices_to_remove)
            knn.fit(X, y)
        else:
            break

    return X, y


def __ib3_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    U = X.copy()
    V = y.copy()

    for i in range(X.shape[0]):
        pass

    return U, V


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

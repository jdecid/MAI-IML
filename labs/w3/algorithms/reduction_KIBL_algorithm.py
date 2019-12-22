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
    def is_acceptable(p, z, n):
        left = p + z * z / 2 * n
        right = z * np.sqrt(p * (1 - p) / n + z * z / 4 * n * n)
        bottom = 1 + z ** 2 / n
        return (left + right) / bottom, (left - right) / bottom

    S = []

    for i in range(X.shape[0]):
        pass
    return U, V


def __drop1_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = list(range(X.shape[0]))
    S = T.copy()
    knn_ = knn.fit(X, y)
    knn.K += 1
    kplus1nn = knn.fit(X, y)
    associates = {}
    for P in S:
        neighbours = kplus1nn.k_neighbours(X[P], only_winner=False)
        for ne in neighbours:
            if ne in associates:
                associates[ne].add(P)
            else:
                associates[ne] = {P}

    for P in S:
        A = associates[P]
        knn_with = knn_.fit(X[S], y[S])
        S_without = S.copy()
        del S_without[S_without.index(P)]
        knn_without = knn_.fit(X[S_without], y[S_without])
        with_ = 0
        without = 0
        for a in A:
            pred_knn_with = knn_with.k_neighbours(X[a])
            if pred_knn_with == y[a]:
                with_ += 1
            pred_knn_without = knn_without.k_neighbours(X[a])
            if pred_knn_without == y[a]:
                without += 1
        if with_ > without:
            del S[S.index(P)]
            for a in A:
                associates[a].remove(P)
                kplus1nn = knn.fit(X[S], y[S])
                a_neighbours = kplus1nn.k_neighbours(X[a], only_winner=False)
                for ne in a_neighbours:
                    if ne not in associates[a]:
                        associates[a].add(ne)
                P_neighbours = kplus1nn.k_neighbours(X[P], only_winner=False)
                for ne in P_neighbours:
                    associates[ne].remove(P)

    return X[S], y[S]


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

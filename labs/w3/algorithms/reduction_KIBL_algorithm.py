from typing import Tuple, List, Set

import numpy as np
import logging

from algorithms.KIBLAlgorithm import KIBLAlgorithm
from algorithms.reduction.cnn import __cnn_reduction

REDUCTION_METHODS = ['CNN', 'RENN', 'IB3', 'DROP1', 'DROP2']


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
    def calculate_boundaries(p, z, n):
        left = p + z * z / 2 * n
        right = z * np.sqrt(p * (1 - p) / n + z * z / 4 * n * n)
        bottom = 1 + z ** 2 / n
        return (left + right) / bottom, (left - right) / bottom

    def is_acceptable(p, z, n, to_accept: bool):
        upper, lower = calculate_boundaries(p, z, n)
        if to_accept:
            pass
        else:
            pass

    def is_significantly_poor():
        return

    # Initialize S empty (We put the first value of X).
    S = np.empty(shape=(1, X.shape[1]))
    S[0, :] = X[0, :]
    V = np.array([y[0]])
    classification_record = {}

    # For each instance in X (Skip the already added first one).
    for t_idx in range(1, X.shape[0]):

        a_idx = None
        a_distance = np.inf
        for s_idx in range(S.shape[0]):
            if is_acceptable(p=np.nan, z=0.9, n=np.nan, to_accept=True):
                s_distance = KIBLAlgorithm.distance_function(X[t_idx, :], S[s_idx, :], r=knn.r)
                if s_distance < a_distance:
                    a_idx = s_idx
                    a_distance = s_distance

        if a_idx is None:
            a_idx = np.random.randint(0, S.shape[0])

        if V[a_idx] != y[t_idx]:
            S = np.vstack((S, X[t_idx, :]))

        indices_to_remove = []
        for s_idx in range(S.shape[0]):
            s_distance = KIBLAlgorithm.distance_function(X[t_idx, :], S[s_idx, :], r=knn.r)
            if s_distance <= a_distance:
                # TODO: Update class record
                if is_significantly_poor():
                    indices_to_remove.append(s_idx)
        S = np.delete(S, indices_to_remove, axis=0)

    indices_to_remove = []
    for s_idx in range(S.shape[0]):
        if not is_acceptable(p=np.nan, z=0.7, n=np.nan, to_accept=False):
            indices_to_remove.append(s_idx)
    S = np.delete(S, indices_to_remove, axis=0)

    return S, y


def __drop1_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray, v2=False) -> Tuple[np.ndarray, np.ndarray]:
    S = X.copy()
    V = y.copy()

    associates: List[Set[int]] = [set() for _ in range(S.shape[0])]
    neighbours = [[] for _ in range(S.shape[0])]

    knn_1 = KIBLAlgorithm(K=knn.K + 1, voting_policy=knn.voting_policy, retention_policy=knn.retention_policy, r=knn.r)
    knn_1.fit(S, V)

    for p_idx in range(S.shape[0]):
        # Find the k + 1 nearest neighbors of p in S.
        neighbours[p_idx] = knn_1.k_neighbours(S[p_idx, :], V[p_idx], only_winner=False)
        logging.debug(f'Instance {p_idx} neighbours -> {neighbours[p_idx]}')

        # Add p to each of its neighbors’ lists of associates.
        for n_idx in neighbours[p_idx]:
            associates[n_idx].add(p_idx)
            logging.debug(f'- Associates of {n_idx} -> {associates[n_idx]}')

    knn_1 = KIBLAlgorithm(K=1, voting_policy=knn_1.voting_policy, retention_policy=knn_1.retention_policy, r=knn_1.r)
    knn_1.fit(S, V)

    p_idx = 0
    p_idx_original = 0
    while p_idx < S.shape[0]:
        # Num. of associates of p classified correctly with p as a neighbour.
        knn.fit(S, V)
        d_with = sum(map(lambda x: y[x] == knn.k_neighbours(S[x], V[x]), associates[p_idx]))
        # Num. of associates of p classified correctly without p as a neighbour.
        knn.fit(np.delete(S, p_idx, axis=0), np.delete(V, p_idx))
        d_without = sum(map(lambda x: y[x] == knn.k_neighbours(S[x], V[x]), associates[p_idx]))

        logging.debug(f'For instance {p_idx}: with={d_with} and without={d_without}')

        if d_without > d_with:
            S = np.delete(S, p_idx, axis=0)
            V = np.delete(V, p_idx, axis=0)
            knn_1.fit(S, V)

            for a_idx in associates[p_idx_original] - {p_idx_original}:
                # Remove p from a’s list of nearest neighbors.
                associates[a_idx] -= {p_idx_original}

                # Find a new nearest neighbor for A.
                a_nn = knn_1.k_neighbours(S[a_idx, :], V[a_idx], only_winner=False)

                # Add A to its new neighbor’s list of associates
                associates[a_nn[0]] = a_idx

            # For each neighbor of p, remove p from its lists of associates.
            for n_idx in neighbours[p_idx_original]:
                associates[n_idx] -= {p_idx_original}
        else:
            p_idx += 1
        p_idx_original += 1

    return S, V



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

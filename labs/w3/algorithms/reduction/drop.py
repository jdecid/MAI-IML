import logging
from typing import Tuple, List, Set

import numpy as np

from algorithms.KIBLAlgorithm import KIBLAlgorithm

from copy import deepcopy


def __drop1_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray, v2=False) -> Tuple[np.ndarray, np.ndarray]:
    S = list(range(X.shape[0]))

    associates: List[Set[int]] = [set() for _ in range(X.shape[0])]
    # neighbours = [[] for _ in range(S.shape[0])]

    knn_1 = KIBLAlgorithm(K=knn.K + 1, voting_policy=knn.voting_policy, retention_policy=knn.retention_policy, r=knn.r)
    knn_1.fit(X, y)

    for p_idx in range(X.shape[0]):
        # Find the k + 1 nearest neighbors of p in S.
        associates[p_idx] = set(knn_1.k_neighbours(X[p_idx, :], y[p_idx], only_winner=False))
        logging.debug(f'Instance {p_idx} neighbours -> {associates[p_idx]}')

        # Add p to each of its neighbors’ lists of associates.
        for n_idx in associates[p_idx]:
            associates[n_idx].add(p_idx)
            logging.debug(f'- Associates of {n_idx} -> {associates[n_idx]}')

    knn_1 = KIBLAlgorithm(K=1, voting_policy=knn_1.voting_policy, retention_policy=knn_1.retention_policy, r=knn_1.r)
    knn_1.fit(X, y)

    p_idx = 0
    p_idx_original = 0
    while p_idx < len(S):
        # Num. of associates of p classified correctly with p as a neighbour.
        knn.fit(X[S], y[S])
        try:
            d_with = sum(map(lambda x: y[x] == knn.k_neighbours(X[x], y[x]), associates[p_idx]))
        except:
            p_idx += 1
            continue
        # Num. of associates of p classified correctly without p as a neighbour.
        # knn.fit(np.delete(S, p_idx, axis=0), np.delete(V, p_idx))
        S_ = S.copy()
        del S_[p_idx]
        knn.fit(X[S_], y[S_])
        try:
            d_without = sum(map(lambda x: y[x] == knn.k_neighbours(X[x], y[x]), associates[p_idx]))
        except:
            p_idx += 1
            continue

        logging.debug(f'For instance {p_idx}: with={d_with} and without={d_without}')

        if d_without >= d_with:
            S = S_
            if v2:
                knn_1.fit(X, y)

            for a_idx in associates[p_idx_original] - {p_idx_original}:
                # Remove p from a’s list of nearest neighbors.
                associates[a_idx] -= {p_idx_original}

                # Find a new nearest neighbor for A.
                a_nn = knn_1.k_neighbours(X[a_idx, :], y[a_idx], only_winner=False)

                # Add A to its new neighbor’s list of associates
                associates[a_nn[0]].add(a_idx)

            # For each neighbor of p, remove p from its lists of associates.
            associates_ = deepcopy(associates)
            for n_idx in associates[p_idx_original]:
                try:
                    associates_[n_idx].remove(p_idx_original)
                except:
                    continue
            associates = associates_
        else:
            p_idx += 1
        p_idx_original += 1

    return X[S], y[S]

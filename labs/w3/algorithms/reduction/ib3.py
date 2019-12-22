from typing import Tuple

import numpy as np

from algorithms.KIBLAlgorithm import KIBLAlgorithm


def __ib3_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def calculate_boundaries(p, z, n):
        left = p + z * z / 2 * n
        right = z * np.sqrt(p * (1 - p) / n + z * z / 4 * n * n)
        bottom = 1 + z ** 2 / n
        return (left + right) / bottom, (left - right) / bottom

    def is_acceptable(p_a, n_a, p_f, n_f, z):
        upper_acc, lower_acc = calculate_boundaries(p_a, z, n_a)
        upper_freq, lower_freq = calculate_boundaries(p_f, z, n_f)
        return lower_acc > upper_freq

    # Initialize S empty (We put the first value of X).
    S = np.empty(shape=(1, X.shape[1]))
    S[0, :] = X[0, :]
    V = np.array([y[0]])
    c_rec = [{} for _ in range(X.shape[0])]

    # For each instance in X (Skip the already added first one).
    for t_idx in range(1, X.shape[0]):
        a_idx = None
        a_distance = np.inf
        for s_idx in range(S.shape[0]):
            # For accuracy:
            # - p_a =
            # - n_a =
            # - z_a = 0.9
            p_a = c_rec[t_idx].get('p_a', 0)
            n_a = c_rec[t_idx].get('n_a', 0)
            if n_a > 0:
                p_a /= n_a

            # For frequency:
            # - p_f = #equal_classes / t_idx
            # - n_f = t_idx
            # - z_f = 0.9
            if is_acceptable(p_a=p_a, n_a=n_a,
                             p_f=sum(map(lambda x: x.get() == y[t_idx], range(t_idx))),
                             n_f=t_idx,
                             z=0.9):

                s_distance = KIBLAlgorithm.distance_function(X[t_idx, :], S[s_idx, :], r=knn.r)
                if s_distance < a_distance:
                    a_idx = s_idx
                    a_distance = s_distance

        if a_idx is None:
            a_idx = np.random.randint(0, S.shape[0])

        if V[a_idx] != y[t_idx]:
            S = np.vstack((S, X[t_idx, :]))

        s_idx = 0
        while s_idx < S.shape[0]:
            s_distance = KIBLAlgorithm.distance_function(X[t_idx, :], S[s_idx, :], r=knn.r)
            if s_distance <= a_distance:
                c_rec[s_idx] = {}
                if not is_acceptable(np.nan, 0.7, np.nan):
                    S = np.delete(S, s_idx, axis=0)
            else:
                s_idx += 1

    s_idx = 0
    while s_idx < S.shape[0]:
        if not is_acceptable(p=np.nan, z=0.7, n=np.nan, to_accept=False):
            S = np.delete(S, s_idx, axis=0)

    return S, y

from typing import Tuple

import numpy as np
from tqdm import tqdm

from algorithms.KIBLAlgorithm import KIBLAlgorithm


def __ib3_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def calculate_boundaries(p, z, n):
        p /= n
        left = p + z * z / 2 * n
        right = z * np.sqrt(p * (1 - p) / n + z * z / 4 * n * n)
        bottom = 1 + z ** 2 / n
        return (left - right) / bottom, (left + right) / bottom

    def is_acceptable(p_a, n_a, p_f, n_f):
        lower_acc, _ = calculate_boundaries(p_a, 0.9, n_a)
        _, upper_freq = calculate_boundaries(p_f, 0.9, n_f)
        return lower_acc > upper_freq

    def is_significantly_poor(p_a, n_a, p_f, n_f):
        _, upper_acc = calculate_boundaries(p_a, 0.7, n_a)
        lower_freq, _ = calculate_boundaries(p_f, 0.7, n_f)
        return lower_freq > upper_acc

    # Initialize S empty (We put the first value of X).
    S = [0]

    c_rec = [{} for _ in range(X.shape[0])]
    c_rec[0]['c'] = 1
    c_rec[0]['acc'] = 1

    # For each instance in X (Skip the already added first one).
    for t_idx in tqdm(range(1, X.shape[0]), desc='Running IB3', ncols=150):
        a_idx = None
        a_distance = np.inf
        distances = KIBLAlgorithm.distance_function(X[t_idx, :], X[S, :], r=knn.r).tolist()
        for s_idx in S:
            if is_acceptable(p_a=c_rec[s_idx]['acc'],
                             n_a=c_rec[s_idx]['c'],
                             p_f=sum(y[S] == y[s_idx]),
                             n_f=t_idx):
                if distances[s_idx] < a_distance:
                    a_idx = s_idx
                    a_distance = distances[s_idx]

        # If no acceptable instances in S, a random_instance(S)
        if a_idx is None:
            a_idx = np.random.randint(0, len(S))
            a_distance = KIBLAlgorithm.distance_function(X[t_idx, :], X[a_idx, :],
                                                         r=knn.r, axis=0)

        # Add t -> S if classes of both are different.
        if y[a_idx] != y[t_idx]:
            S.append(t_idx)
            distances += [0]

        for d_idx, s_idx in enumerate(S):
            d = KIBLAlgorithm.distance_function(X[t_idx, :], X[s_idx, :], r=knn.r, axis=0)
            if d < a_distance:
                if s_idx == t_idx:
                    c_rec[s_idx]['c'] = 1
                    c_rec[s_idx]['acc'] = 1
                else:
                    c_rec[s_idx]['c'] += 1
                    if y[s_idx] == y[a_idx]:
                        c_rec[s_idx]['acc'] += 1

                if is_significantly_poor(p_a=c_rec[s_idx]['acc'],
                                         n_a=c_rec[s_idx]['c'],
                                         p_f=sum(y[S] == y[s_idx]),
                                         n_f=t_idx):
                    S.remove(s_idx)
            else:
                c_rec[s_idx]['c'] = 1
                c_rec[s_idx]['acc'] = 1

    for s_idx in S:
        if not is_acceptable(p_a=c_rec[s_idx]['acc'],
                             n_a=c_rec[s_idx]['c'],
                             p_f=sum(y[S] == y[s_idx]),
                             n_f=t_idx):
            S.remove(s_idx)

    return X[S, :], y[S]

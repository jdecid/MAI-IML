from typing import Tuple

import numpy as np
from tqdm import tqdm

from algorithms.KIBLAlgorithm import KIBLAlgorithm


def __ib3_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    def calculate_boundaries(p, z, n):
        try:
            left = p + z * z / 2 * n
            right = z * np.sqrt(p * (1 - p) / n + z * z / 4 * n * n)
            bottom = 1 + z ** 2 / n
            return (left + right) / bottom, (left - right) / bottom
        except ZeroDivisionError:
            return 0, 0

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
    for t_idx in tqdm(range(1, X.shape[0]), desc='Running IB3', ncols=150):
        a_idx = None
        a_distance = np.inf
        for s_idx in range(S.shape[0]):
            # For accuracy:
            # - p_a =
            # - n_a =
            # - z_a = 0.9
            accuracy_attempts = c_rec[t_idx].get('p', 0)
            classification_attempts = c_rec[t_idx].get('n', 0)
            if classification_attempts > 0:
                accuracy_attempts /= classification_attempts

            # For frequency:
            # - p_f = #equal_classes / t_idx
            # - n_f = t_idx
            # - z_f = 0.9
            if is_acceptable(p_a=accuracy_attempts, n_a=classification_attempts,
                             p_f=sum(map(lambda x: c_rec[x].get('pred') == y[t_idx], range(t_idx))),
                             n_f=t_idx,
                             z=0.9):

                s_distance = KIBLAlgorithm.distance_function(X[t_idx, :], S[s_idx, :], r=knn.r, axis=0)
                if s_distance < a_distance:
                    a_idx = s_idx
                    a_distance = s_distance

        # If no acceptable instances in S, a random_instance(S)
        if a_idx is None:
            a_idx = np.random.randint(0, S.shape[0])

        # Add t -> S if classes of both are different.
        if V[a_idx] != y[t_idx]:
            S = np.vstack((S, X[t_idx, :]))
            V = np.concatenate((V, [y[t_idx]]))

        indices_to_delete = []
        for s_idx in range(S.shape[0]):
            s_distance = KIBLAlgorithm.distance_function(X[t_idx, :], S[s_idx, :], r=knn.r, axis=0)

            if s_distance <= a_distance:
                c_rec[s_idx]['pred'] = knn.k_neighbours(S, V)
                c_rec[s_idx]['p'] = c_rec[s_idx].get('p', 0) + 1 if c_rec[s_idx]['pred'] == y[t_idx] else 0
                c_rec[s_idx]['n'] = c_rec[s_idx].get('n', 0) + 1
                if not is_acceptable(p_a=c_rec[s_idx]['p'],
                                     n_a=c_rec[s_idx]['n'],
                                     p_f=sum(map(lambda x: c_rec[x].get('pred') == y[t_idx], range(t_idx))),
                                     n_f=t_idx,
                                     z=0.7):
                    indices_to_delete.append(s_idx)

        S = np.delete(S, indices_to_delete, axis=0)
        V = np.delete(V, indices_to_delete)

    indices_to_delete = []
    for s_idx in range(S.shape[0]):
        if not is_acceptable(p_a=c_rec[s_idx]['p'],
                             n_a=c_rec[s_idx]['n'],
                             p_f=sum(map(lambda x: c_rec[x].get('pred') == y[t_idx], range(t_idx))),
                             n_f=t_idx,
                             z=0.7):
            indices_to_delete.append(s_idx)
    S = np.delete(S, indices_to_delete, axis=0)

    return S, y

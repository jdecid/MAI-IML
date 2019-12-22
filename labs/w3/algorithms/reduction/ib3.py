from typing import Tuple

import numpy as np

from algorithms.KIBLAlgorithm import KIBLAlgorithm


def calculate_boundaries(p, z, n):
    left = p + z * z / 2 * n
    right = z * np.sqrt(p * (1 - p) / n + z * z / 4 * n * n)
    bottom = 1 + z ** 2 / n
    return (left + right) / bottom, (left - right) / bottom


def is_acceptable(p, z, n, to_accept: bool):
    upper, lower = calculate_boundaries(p, z, n)


def is_significantly_poor():
    return


def __get_nearest_acceptable_instance_a(X: np.ndarray, S: np.ndarray, t_idx: int, r: int):
    a_idx = None
    a_distance = np.inf
    for s_idx in range(S.shape[0]):
        if is_acceptable(p=np.nan, z=0.9, n=np.nan, to_accept=True):
            s_distance = KIBLAlgorithm.distance_function(X[t_idx, :], S[s_idx, :], r=r)
            if s_distance < a_distance:
                a_idx = s_idx
                a_distance = s_distance

    if a_idx is None:
        a_idx = np.random.randint(0, S.shape[0])

    return a_idx, a_distance

def remove_non_acceptable_instances():


def ib3_reduction(knn: KIBLAlgorithm, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Initialize S empty (We put the first value of X).
    S = np.empty(shape=(1, X.shape[1]))
    S[0, :] = X[0, :]
    V = np.array([y[0]])
    classification_record = {}

    # For each instance in X (Skip the already added first one).
    for t_idx in range(1, X.shape[0]):
        a_idx, a_distance = __get_nearest_acceptable_instance_a(X, S, t_idx, knn.r)
        if V[a_idx] != y[t_idx]:
            S = np.vstack((S, X[t_idx, :]))

        s_idx = 0
        while s_idx < S.shape[0]:
            s_distance = KIBLAlgorithm.distance_function(X[t_idx, :], S[s_idx, :], r=knn.r)
            if s_distance <= a_distance:
                classification_record[s_idx] = {}
                if not is_acceptable(np.nan, 0.7, np.nan):
                    S = np.delete(S, s_idx, axis=0)
            else:
                s_idx += 1

    s_idx = 0
    while s_idx < S.shape[0]:
        if not is_acceptable(p=np.nan, z=0.7, n=np.nan, to_accept=False):
            S = np.delete(S, s_idx, axis=0)

    return S, y

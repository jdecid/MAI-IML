from typing import Tuple

import numpy as np

from algorithms.KIBLAlgorithm import KIBLAlgorithm


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

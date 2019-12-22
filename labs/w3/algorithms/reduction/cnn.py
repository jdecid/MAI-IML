from typing import Tuple

import numpy as np

from algorithms.KIBLAlgorithm import KIBLAlgorithm


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
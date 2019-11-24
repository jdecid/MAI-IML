import numpy as np


def leave_one_out_cross_validation(X: np.ndarray, n_folds: int = 10) -> float:
    """

    """
    accuracies = []
    folds_indices = np.array_split(np.random.permutation(X.shape[0]), n_folds)
    for t in range(n_folds):
        X_train = X[np.concatenate([indices for idx, indices in enumerate(folds_indices) if t != idx])]
        X_test = X[folds_indices[t]]

        accuracy = run_model(X_train, X_test)
        accuracies.append(accuracy)

    return sum(accuracies) / n_folds

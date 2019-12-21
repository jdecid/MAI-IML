import numpy as np

REDUCTION_METHODS = ['CNN', 'SNN' 'ENN', 'RENN'] + [f'DROP{i}' for i in range(1, 6)]


def __cnn_reduction(X: np.ndarray) -> np.ndarray:
    X_copy = X.copy()
    # TODO: Initial U initialization
    U = []
    while True:
        nearest_x_idx = None
        for i in range(X_copy.shape[1]):
            if X_copy[i] is not None:
                pass
                # TODO: Implement

        if nearest_x_idx is not None:
            U.append(X_copy[nearest_x_idx])
            X_copy[nearest_x_idx] = None
        else:
            break

    return np.stack(U)


def reduction_KIBL_algorithm(X: np.ndarray, reduction_method: str):
    if reduction_method not in REDUCTION_METHODS:
        raise ValueError(f'Unknown reduction method {reduction_KIBL_algorithm()}')

    if reduction_method == 'CNN':
        __cnn_reduction(X)

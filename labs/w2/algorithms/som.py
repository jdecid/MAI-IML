import numpy as np
from neupy.algorithms import SOFM


class SOM:
    def __init__(self):
        self.model: 'SOFM' = None

    def fit(self, X: np.ndarray, epochs=100, shuffle=False, verbose=False):
        self.model = SOFM(
            n_inputs=X.shape[1],
            features_grid=0,
            learning_radius=0,
            reduce_radius_after=50,
            step=0.5,
            std=1,
            shuffle_data=shuffle,
            verbose=verbose
        )

        self.model.train(X_train=X, epochs=epochs)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)

    def fit_predict(self, X: np.ndarray, epochs=100, shuffle=False, verbose=False):
        self.fit(X, epochs=epochs, shuffle=shuffle, verbose=verbose)
        return self.predict(X)

    def _init_nodes(self):
        """

        :return:
        """

    def __xavier_glorot(self):
        pass

    def __he_et_al(self):
        pass

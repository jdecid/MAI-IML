import numpy as np


class FuzzyCMeans:
    def __init__(self, c: int):
        self.model = None

    def fit(self) -> None:
        pass

    def predict(self, X: np.ndarray):
        assert self.model is not None, 'Use .fit() method to feed the model with data before using it'

    def fit_predict(self):
        self.fit()
        return self.predict()

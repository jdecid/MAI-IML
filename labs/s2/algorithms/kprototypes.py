from scipy import stats

from algorithms.kmeans import KMeans
from utils.evaluate import *


class KPrototypes(KMeans):
    def __init__(self, cat_idx, gamma=1, **kwargs):
        self.gamma = gamma
        self.cat_idx = cat_idx
        self.mask = None

        super().__init__(**kwargs)

    def fit(self, X: np.ndarray, max_it=20):
        self.mask = np.zeros(X.shape[1], dtype=bool)
        self.mask[self.cat_idx] = True

        super().fit(X)

    def compute_point_wise_distances(self, X):
        self.mask = np.zeros(X.shape[1], dtype=bool)
        self.mask[self.cat_idx] = True
        return super().compute_point_wise_distances(X)

    def _compute_centroids(self):
        # Categorical
        for k in range(self.K):
            if self.nearest[k]:
                self.centroids[k, self.mask] = stats.mode(np.array(self.nearest[k])[:, self.mask]).mode[0]

        # Numerical
        self.centroids[:, ~self.mask] = np.array(
            list(map(lambda x: np.mean(np.array(x)[:, ~self.mask], axis=0), self.nearest)))

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute distance between two points using:
            - Categorical attributes: Kronecker's delta (1 if the values are equal, and 0 otherwise).
            - Numerical attributes: Distance according to the specified metric in the instantiation.
        :param a: 1D vector with all A attributes.
        :param b: 1D vector with all B attributes.
        :return: Numerical attributes distance + gamma_factor * Categorical attributes distance.
        """
        num_sum = 0
        if any(not e for e in self.mask):
            num_sum = np.linalg.norm(a[~self.mask] - b[~self.mask])
        cat_sum = 0
        if any(e for e in self.mask):
            cat_sum = sum(a[self.mask] != b[self.mask])
        return num_sum + self.gamma * cat_sum

import numpy as np
from scipy.stats import mode

from algorithms.kmeans import KMeans


class KModes(KMeans):
    def _compute_centroids(self):
        for k in range(self.K):
            if len(self.nearest[k]) > 0:
                self.centroids[k, :] = mode(self.nearest[k])[0][0]

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute distance between two points using Kronecker's delta (1 if the values are equal, and 0 otherwise)
        :param a: 1D vector with all A attributes.
        :param b: 1D vector with all B attributes.
        :return: Distance (dissimilarity) between a and b.
        """
        return sum(a != b)

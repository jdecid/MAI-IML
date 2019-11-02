from scipy import stats

from algorithms.kmeans import KMeans
from utils.evaluate import *


class KModes(KMeans):
    def _compute_centroids(self):
        for k in range(self.K):
            if self.nearest[k]:
                self.centroids[k, :] = stats.mode(self.nearest[k]).mode[0]

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute distance between two points using Kronecker's delta (1 if the values are equal, and 0 otherwise)
        :param a: 1D vector with all A attributes.
        :param b: 1D vector with all B attributes.
        :return: Distance (dissimilarity) between a and b.
        """
        return sum(a != b)

import numpy as np
import pandas as pd

from scipy import stats
from algorithms.kmeans import KMeans


class KModes(KMeans):
    def _init_centroids(self):
        idx = np.random.choice(range(self.K), size=self.K, replace=False)
        self.centroids = self.X[idx]

    def _compute_centroids(self, nearest):
        for k in range(self.K):
            if nearest[k]:
                self.centroids[k, :] = stats.mode(nearest[k]).mode[0]

    @staticmethod
    def _distance(a, b, **kwargs):
        # delta kronecker (0 if ==, else 1)
        return np.sum(a != b)


dataset = pd.read_csv('../datasets/connect-4-clean.csv')
dataset.describe()
X = dataset.iloc[:30000, :].values

kmodes = KModes(3)
kmodes.fit(X)
print(kmodes.predict(X[:20, :]))

import numpy as np
import pandas as pd

from scipy import stats
from algorithms.kmeans import KMeans
import collections

class KModes(KMeans):
    def __init__(self, K: int, seed=1):
        """

        :param K: Number of Clusters
        :param seed: Fixed seed to allow reproducibility.
        """
        super().__init__(K, seed=seed)

    def _init_centroids(self):
        idx = np.random.choice(range(0, self.X.shape[0]), size=self.K, replace=False)
        print(idx)
        self.centroids = self.X[idx]

    def _compute_centroids(self, nearest):
        for k in range(self.K):
            if nearest[k]:
                self.centroids[k, :] = stats.mode(nearest[k]).mode[0]

    def _distance(self, a, b):
        # delta kronecker (0 if ==, else 1)
        return np.sum(a != b)


if __name__ == '__main__':
    dataset = pd.read_csv('../datasets/car.data')
    X = dataset.iloc[:,:6]

    kmodes = KModes(K=4,seed=1)
    res = kmodes.fit_predict(X.values)
    with open('res.txt', 'w') as f:
        f.write(str(res))
    c = collections.Counter()
    c.update(res)
    print(c)
    Y = dataset.iloc[:,6]
    print(Y.value_counts())


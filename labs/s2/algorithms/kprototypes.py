import numpy as np
import pandas as pd

from scipy import stats
from algorithms.kmeans import KMeans
from sklearn.preprocessing import MinMaxScaler


class KPrototypes(KMeans):
    def __init__(self, K, cat_idx, gamma=1):
        super().__init__(K)
        self.gamma = gamma
        self.cat_idx = cat_idx
        self.mask = None

    def fit(self, X: np.ndarray, max_it=20):
        self.mask = np.zeros(X.shape[1], dtype=bool)
        self.mask[self.cat_idx] = True

        super().fit(X, max_it)

    def _init_centroids(self):
        idx = np.random.choice(range(self.K), size=self.K, replace=False)
        self.centroids = self.X[idx]

    def _compute_centroids(self, nearest):
        # Categorical
        for k in range(self.K):
            if nearest[k]:
                self.centroids[k, self.mask] = stats.mode(np.array(nearest[k])[:, self.mask]).mode[0]

        # Numerical
        self.centroids[:, ~self.mask] = np.array(list(map(lambda x: np.mean(np.array(x)[:, ~self.mask], axis=0), nearest)))

    def _distance(self, a, b):
        # delta kronecker (0 if ==, else 1)
        num_sum = np.linalg.norm(a[~self.mask] - b[~self.mask])
        cat_sum = np.sum(a[self.mask] != b[self.mask])
        return num_sum + self.gamma * cat_sum


if __name__ == '__main__':
    dataset = pd.read_csv('Mall_Customers.csv')
    dataset.describe()
    X = dataset.iloc[:, 1:].values

    scaler = MinMaxScaler()
    X[:, 1:] = scaler.fit_transform(X[:, 1:])

    kprototypes = KPrototypes(3, cat_idx=[0])
    kprototypes.fit_predict(X)

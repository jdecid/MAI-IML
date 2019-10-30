import collections

import pandas as pd

from scipy import stats
from algorithms.kmeans import KMeans
from utils.evaluate import *

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
        self.centroids[:, ~self.mask] = np.array(
            list(map(lambda x: np.mean(np.array(x)[:, ~self.mask], axis=0), nearest)))

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute distance between two points using:
            - Categorical attributes: Kronecker's delta (1 if the values are equal, and 0 otherwise).
            - Numerical attributes: Distance according to the specified metric in the instantiation.
        :param a: 1D vector with all A attributes.
        :param b: 1D vector with all B attributes.
        :return: Numerical attributes distance + gamma_factor * Categorical attributes distance.
        """
        num_sum = np.linalg.norm(a[~self.mask] - b[~self.mask])
        cat_sum = sum(a[self.mask] != b[self.mask])
        return num_sum + self.gamma * cat_sum


if __name__ == '__main__':
    ''''
    dataset = pd.read_csv('../datasets/cmc.data')
    X = dataset.iloc[:, :9]

    kprototypes = KPrototypes(K=3, cat_idx=[1, 2, 4, 5, 6, 7, 8])
    res = kprototypes.fit_predict(X.values)
    with open('res.txt', 'w') as f:
        f.write(str(res))
    c = collections.Counter()
    c.update(res)
    print(c)
    Y = dataset.iloc[:, 9]
    print(Y.value_counts())
    '''
    dataset = pd.read_csv('../tests/datasets/post-operative.data')
    print(len(dataset))
    dataset = dataset[~dataset.isin(["?"]).any(axis=1)]
    dataset.iloc[:, 7] = dataset.iloc[:, 7].apply(lambda x: int(x))
    dataset.iloc[:, 8] = dataset.iloc[:, 8].apply(lambda x: x.strip())
    print(len(dataset))
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:,-1]
    print(set(y))
    kprototypes = KPrototypes(K=3, cat_idx=[0,1,2,3,4,5,6])
    res = kprototypes.fit_predict(X.values)
    c = collections.Counter()
    c.update(res)
    print(c)
    print(evaluate_supervised(labels_true=y, labels_pred=res))



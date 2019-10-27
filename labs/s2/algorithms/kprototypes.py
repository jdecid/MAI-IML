import numpy as np
import pandas as pd

from scipy import stats
from algorithms.kmeans import KMeans
from sklearn.preprocessing import MinMaxScaler


class KPrototypes(KMeans):
    def __init__(self, K, cat_columns, gamma=1):
        print(cat_columns)
        super().__init__(K)
        self.dist_params['gamma'] = gamma
        self.dist_params['cat_columns'] = cat_columns
        print(self.dist_params)

    def _init_centroids(self):
        idx = np.random.choice(range(self.K), size=self.K, replace=False)
        self.centroids = self.X[idx]

    def _compute_centroids(self, nearest):
        # self.centroids = np.array(list(map(lambda x: np.mean(np.array(x), axis=0), nearest)))

        for k in range(self.K):
            if nearest[k][self.dist_params['cat_columns']]:
                self.centroids[k, self.dist_params['cat_columns']] = stats.mode(nearest[k][self.dist_params['cat_columns']]).mode[0]
        self.centroids[:, np.logical_not(self.dist_params['cat_columns'])] = \
            np.array(list(map(lambda x: np.mean(np.array(x), axis=0), nearest[np.logical_not(self.dist_params['cat_columns'])])))


    @staticmethod
    def _distance(a, b, **kwargs):
        # delta kronecker (0 if ==, else 1)
        num_sum = np.linalg.norm(a[np.logical_not(kwargs['cat_columns'])] - b[np.logical_not(kwargs['cat_columns'])])
        cat_sum = np.sum(a[kwargs['cat_columns']] != b[kwargs['cat_columns']])
        return num_sum + kwargs['gamma'] * cat_sum


if __name__ == '__main__':
    dataset = pd.read_csv('Mall_Customers.csv')
    dataset.describe()
    X = dataset.iloc[:, 1:].values

    scaler = MinMaxScaler()
    X[:, 1:] = scaler.fit_transform(X[:, 1:])

    kprototypes = KPrototypes(3, [1] + [0]*3)
    kprototypes.fit_predict(X)

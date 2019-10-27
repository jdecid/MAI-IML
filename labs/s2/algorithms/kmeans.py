import colorsys
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class KMeans:
    def __init__(self, K, seed=1, visualize=False):
        print('KMeans')
        self.K = K
        self.seed = seed
        self.visualize = visualize

        self.X = None
        self.centroids = None

        colors = [(x * 1.0 / K, x * 1.0 / K, 0.5) for x in range(K)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), colors))
        self.colors = [plt.cm.hsv(x / K) for x in range(K)]
        self.dist_params = {}

    def fit(self, X: np.ndarray, max_it=20):
        np.random.seed(self.seed)
        self.X = X
        self._init_centroids()

        previous_nearest_idx = None

        it = 0
        while it < max_it:
            print(f'Iteration {it}')

            # Calculate distance from each point to each cluster
            distances = np.zeros(shape=(self.K, X.shape[0]))
            for c_idx, centroid in enumerate(self.centroids):
                for p_idx, point in enumerate(X):
                    distances[c_idx, p_idx] = self._distance(centroid, point, **self.dist_params)

            # Get nearest points for each cluster
            nearest = [[] for _ in range(self.K)]
            nearest_idx = [[] for _ in range(self.K)]

            for p_idx, point in enumerate(X):
                min_index = int(np.argmin(distances[:, p_idx]))
                nearest[min_index].append(point)
                nearest_idx[min_index].append(p_idx)

            if self.visualize:
                self._display_iteration(X, nearest_idx)

            # Recalculate centroids as the mean of their nearest points
            self._compute_centroids(nearest)

            # Check convergence
            it += 1
            if previous_nearest_idx == nearest_idx:
                break
            else:
                previous_nearest_idx = nearest_idx

            time.sleep(1)

    def _compute_centroids(self, nearest):
        # TODO: Ugly AF
        self.centroids = np.array(list(map(lambda x: np.mean(np.array(x), axis=0), nearest)))

    def predict(self, X):
        # Calculate distance from each point to each cluster
        distances = np.zeros(shape=(self.K, X.shape[0]))
        for c_idx, centroid in enumerate(self.centroids):
            for p_idx, point in enumerate(X):
                distances[c_idx, p_idx] = self._distance(centroid, point, **self.dist_params)

        # Get nearest points for each cluster
        nearest = [[] for _ in range(self.K)]

        classes = []
        for p_idx, point in enumerate(X):
            cluster_idx = int(np.argmin(distances[:, p_idx]))
            classes.append(cluster_idx)

        return classes

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)

    def _init_centroids(self):
        self.centroids = np.random.random(size=(self.K, self.X.shape[1]))

    def _display_iteration(self, X, nearest_idx):
        plt.figure(figsize=(4, 4))
        for idx in range(self.K):
            plt.scatter(X[nearest_idx[idx], 0], X[nearest_idx[idx], 1],
                        c=[self.colors[idx]], s=10, alpha=0.5)
            plt.scatter(self.centroids[idx, 0], self.centroids[idx, 1],
                        c=[self.colors[idx]], s=150, marker='*', edgecolors='black')
        plt.show()

    @staticmethod
    def _distance(a, b, **kwargs):
        return np.linalg.norm(a - b)



if __name__ == '__main__':
    dataset = pd.read_csv('Mall_Customers.csv')
    dataset.describe()
    X = dataset.iloc[:, [3, 4]].values
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    kmeans = KMeans(3)
    kmeans.fit_transform(X)
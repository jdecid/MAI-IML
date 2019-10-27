import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler


class KMeans:
    def __init__(self, K, seed=1):
        self.K = K
        self.seed = seed

        self.centroids = None

    def fit(self, X: np.ndarray, max_it=20):
        np.random.seed(self.seed)
        self.centroids = np.random.random(size=(self.K, X.shape[1]))

        previous_nearest_idx = None

        it = 0
        while it < max_it:
            print(f'Iteration {it}')
            self.__display_iteration(X)

            # Calculate distance from each point to each cluster
            distances = np.zeros(shape=(self.K, X.shape[0]))
            for c_idx, centroid in enumerate(self.centroids):
                for p_idx, point in enumerate(X):
                    distances[c_idx, p_idx] = KMeans.__distance(centroid, point)

            # Get nearest points for each cluster
            nearest = [[] for _ in range(self.K)]
            nearest_idx = [set() for _ in range(self.K)]

            for p_idx, point in enumerate(X):
                min_index = int(np.argmin(distances[:, p_idx]))
                nearest[min_index].append(point)
                nearest_idx[min_index].add(p_idx)

            # Recalculate centroids as the mean of their nearest points
            # TODO: Ugly AF
            self.centroids = np.array(list(map(lambda x: np.mean(np.array(x), axis=0), nearest)))

            # Check convergence
            it += 1
            if previous_nearest_idx == nearest_idx:
                break
            else:
                previous_nearest_idx = nearest_idx

    def transform(self, X):
        # Calculate distance from each point to each cluster
        distances = np.zeros(shape=(self.K, X.shape[0]))
        for c_idx, centroid in enumerate(self.centroids):
            for p_idx, point in enumerate(X):
                distances[c_idx, p_idx] = KMeans.__distance(centroid, point)

        # Get nearest points for each cluster
        nearest = [[] for _ in range(self.K)]

        for p_idx, point in enumerate(X):
            min_index = int(np.argmin(distances[:, p_idx]))
            nearest[min_index].append(point)

        return nearest

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def __display_iteration(self, X):
        plt.figure(figsize=(4, 4))
        plt.scatter(X[:, 0], X[:, 1], c='blue')
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], c='red')
        plt.show()

    @staticmethod
    def __distance(a, b):
        return np.linalg.norm(a - b)


dataset = pd.read_csv('Mall_Customers.csv')
dataset.describe()
X = dataset.iloc[:, [3, 4]].values

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

kmeans = KMeans(5)
kmeans.fit_transform(X)

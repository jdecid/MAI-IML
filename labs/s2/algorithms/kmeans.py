from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from sklearn.decomposition import PCA


class KMeans:
    """

    """

    def __init__(self, K: int, metric='euclidean', vis_dims=0, seed=1):
        """

        :param K: Number of Clusters
        :param metric: Distance function
        :param vis_dims: Visualization level (0 no visualization, 2 for 2D and 3 for 3D).
        :param seed: Fixed seed to allow reproducibility.
        """
        if K < 1:
            raise ValueError('K must be a positive number > 0')

        if vis_dims not in [0, 2, 3]:
            raise ValueError('Visualize dimensions must be an integer in {0, 2, 3}')

        if metric not in ['euclidean', 'cityblock', 'cosine']:
            raise ValueError('Accepted metrics are `euclidean`, `cityblock` and `cosine` distances')

        self.K = K
        self.metric = metric
        self.vis_dims = vis_dims

        self.seed = seed

        self.X = None
        self.centroids = None

        self.colors = self._get_colors(K)

    def fit(self, X: np.ndarray, max_it=20):
        np.random.seed(self.seed)
        self.X = X
        self._init_centroids()

        previous_nearest_idx = None

        it = 0
        while True:
            distances = self._calculate_distances(X)
            _, nearest, nearest_idx = self._get_nearest(X, distances)

            self._display_iteration(X, nearest_idx)

            self._compute_centroids(nearest)

            # Check convergence

            it += 1
            if it >= max_it or previous_nearest_idx == nearest_idx:
                break
            else:
                previous_nearest_idx = nearest_idx

    def predict(self, X: np.ndarray) -> List[int]:
        """
        Assign clusters to a list of observations.
        :param X: 2D data array of size (rows, features).
        :return: Cluster indexes assigned to each row of X.
        """
        if self.centroids is None:
            raise Exception('Fit the model with some data before running a prediction')

        distances = self._calculate_distances(X)
        classes, _, _ = self._get_nearest(X, distances)

        return classes

    def fit_predict(self, X: np.ndarray, max_it=1000) -> List[int]:
        """
        Fit the model with provided data and return their assigned clusters.
        :param X: 2D data array of size (rows, features).
        :param max_it: Maximum number of iterations for the algorithm.
        :return: Cluster indexes assigned to each row of X.
        """
        self.fit(X, max_it)
        return self.predict(X)

    def _init_centroids(self):
        """Initialize centroids"""
        self.centroids = np.random.random(size=(self.K, self.X.shape[1]))

    def _calculate_distances(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate distance from each point to each cluster
        :param X: 2D vector with the set of points of size (#observations, #features).
        :return: Distance matrix of shape (K, #observations)
        """
        distances = np.zeros(shape=(self.K, X.shape[0]))

        for c_idx, centroid in enumerate(self.centroids):
            for p_idx, point in enumerate(X):
                distances[c_idx, p_idx] = self._distance(centroid, point)

        return distances

    def _get_nearest(self, X: np.ndarray, distances: np.ndarray) \
            -> Tuple[List[int], List[List[np.ndarray]], List[List[int]]]:
        """
        For each data instance
        :param X: 2D vector with the set of points of size (#observations, #features).
        :param distances: 2D vector of distances between centroids and observations.
        :return: Tuple containing:
            - Cluster indexes assigned to each observation.
            - List of nearest observations for each cluster.
            - List of nearest observations index for each cluster.
        """
        # Get nearest points for each cluster
        classes = []
        nearest = [[] for _ in range(self.K)]
        nearest_idx = [[] for _ in range(self.K)]

        for p_idx, point in enumerate(X):
            cluster_idx = int(np.argmin(distances[:, p_idx]))

            classes.append(cluster_idx)
            nearest[cluster_idx].append(point)
            nearest_idx[cluster_idx].append(p_idx)

        return classes, nearest, nearest_idx

    def _compute_centroids(self, nearest):
        # TODO: Ugly AF
        self.centroids = np.array(list(map(lambda x: np.mean(np.array(x), axis=0), nearest)))

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute distance between 2 elements using the specified metric. Check metrics in:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        :param a: 1D vector with all A attributes.
        :param b: 1D vector with all B attributes.
        :return: Distance between both vectors using the specified metric.
        """
        return distance.cdist(np.array([a]), np.array([b]), metric=self.metric)[0][0]

    def _display_iteration(self, X, nearest_idx):
        """
        Visualize a plot for the current iteration centroids and point assignments.
        :param X: 2D vector with the set of points of size (instances, features).
        :param nearest_idx: List of size (K, ?) assigning each point to its nearest cluster.
        """
        if self.vis_dims == 0:
            return

        points = X.copy()
        centroids = self.centroids.copy()

        data_components = points.shape[1]
        if data_components > self.vis_dims:
            pca = PCA(n_components=self.vis_dims)
            points = pca.fit_transform(points)
            centroids = pca.transform(centroids)

        f = plt.figure(figsize=(4, 4))

        # Visualization for 3D
        if self.vis_dims == 3:
            ax = Axes3D(f)
            for k in range(self.K):
                # Plot centroid k
                ax.scatter(xs=centroids[k, 0],
                           ys=centroids[k, 1],
                           zs=centroids[k, 2],
                           c=[self.colors[k]], s=150,
                           marker='*', edgecolors='black')

                # Plot points associated with cluster k
                ax.scatter(xs=points[nearest_idx[k], 0],
                           ys=points[nearest_idx[k], 1],
                           zs=points[nearest_idx[k], 2],
                           c=[self.colors[k]], s=10, alpha=0.5)

        # Visualization for 2D
        else:
            for k in range(self.K):
                # Plot centroid k
                plt.scatter(x=centroids[k, 0],
                            y=centroids[k, 1],
                            c=[self.colors[k]], s=150,
                            marker='*', edgecolors='black')

                # Plot points associated with cluster k
                plt.scatter(x=points[nearest_idx[k], 0],
                            y=points[nearest_idx[k], 1],
                            c=[self.colors[k]], s=10, alpha=0.5)

        plt.show()

    @staticmethod
    def _get_colors(n):
        """
        Sample RGBA colors from HSV matplotlib colormap.
        :param n: Number of colors to obtain.
        :return: List of n RGBA colors.
        """
        return [plt.cm.hsv(x / n) for x in range(n)]

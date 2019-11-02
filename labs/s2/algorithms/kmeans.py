import logging
import os
from typing import List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from sklearn.decomposition import PCA

from utils.plotting import get_colors


class KMeans:
    """

    """

    def __init__(self, K: int, name: str, max_it=1000, metric='euclidean', vis_dims=0, fig_save_path: str = None,
                 return_distances=False, seed=1):
        """

        :param K: Number of Clusters
        :param metric: Distance function
        :param max_it: Maximum number of iterations if hasn't reached convergence yet.
        :param vis_dims: Visualization level (0 no visualization, 2 for 2D and 3 for 3D).
        :param fig_save_path: Whether to save figure.
        :param return_distances: True if distances are returned in a tuple in predict method.
        :param seed: Fixed seed to allow reproducibility.
        """
        if K < 1:
            raise ValueError('K must be a positive number > 0')

        if vis_dims not in [0, 2, 3]:
            raise ValueError('Visualize dimensions must be an integer in {0, 2, 3}')

        if metric not in ['euclidean', 'cityblock', 'cosine']:
            raise ValueError('Accepted metrics are `euclidean`, `cityblock` and `cosine` distances')

        self.K = K
        self.name = name

        self.metric = metric

        self.vis_dims = vis_dims
        self.fig_save_path = fig_save_path

        self.return_distances = return_distances
        self.seed = seed

        self.it = 0
        self.max_it = max_it

        self.X = None
        self.centroids = None
        self.nearest = None

        self.colors = get_colors(K)

    def fit(self, X: np.ndarray):
        np.random.seed(self.seed)
        self.X = X
        self._init_centroids()

        # previous_nearest_idx = None
        previous_centroids = None

        while True:
            distances = self._calculate_distances(X)
            _, self.nearest, nearest_idx = self._get_nearest(X, distances)

            self._display_iteration(X, nearest_idx)

            self._compute_centroids()

            # Check convergence
            logging.info(f'{self.it:3}/{self.max_it} Loss: {self._loss()}')

            self.it += 1
            if self._check_convergence(previous_centroids):
                break
            else:
                previous_centroids = self.centroids

    def predict(self, X: np.ndarray) -> Union[List[int], Tuple[List[int], np.ndarray]]:
        """
        Assign clusters to a list of observations.
        :param X: 2D data array of size (rows, features).
        :return: Cluster indexes assigned to each row of X.
        """
        if self.centroids is None:
            raise Exception('Fit the model with some data before running a prediction')

        distances = self._calculate_distances(X)
        classes, _, _ = self._get_nearest(X, distances)

        return (classes, distances) if self.return_distances else classes

    def fit_predict(self, X: np.ndarray) -> List[int]:
        """
        Fit the model with provided data and return their assigned clusters.
        :param X: 2D data array of size (rows, features).
        :return: Cluster indexes assigned to each row of X.
        """
        self.fit(X)
        return self.predict(X)

    def _init_centroids(self):
        """Initialize centroids"""
        # self.centroids = np.random.random(size=(self.K, self.X.shape[1]))
        idx = np.random.choice(range(self.X.shape[0]), size=self.K, replace=False)
        self.centroids = self.X[idx, :]

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

    def _compute_centroids(self):
        for k in range(self.K):
            if len(self.nearest[k]) > 0:  # TODO: Review if necessary
                self.centroids[k, :] = np.mean(np.array(self.nearest[k]), axis=0)

    def _loss(self):
        # TODO: Avoid repeat, use previous calculated distances
        loss = 0
        for k in range(len(self.nearest)):
            loss += sum(list(map(lambda x: self._distance(x, self.centroids[k]), self.nearest[k])))
        return loss

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute distance between 2 elements using the specified metric. Check metrics in:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        :param a: 1D vector with all A attributes.
        :param b: 1D vector with all B attributes.
        :return: Distance between both vectors using the specified metric.
        """
        return distance.cdist(np.array([a]), np.array([b]), metric=self.metric)[0][0]

    def _check_convergence(self, previous_centroids):
        if self.it >= self.max_it:
            return True
        conv = previous_centroids == self.centroids
        if isinstance(conv, np.ndarray):
            return conv.all()
        return conv

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
                           marker='*', edgecolors='black', zorder=2)

                # Plot points associated with cluster k
                ax.scatter(xs=points[nearest_idx[k], 0],
                           ys=points[nearest_idx[k], 1],
                           zs=points[nearest_idx[k], 2],
                           c=[self.colors[k]], s=10, alpha=0.5, zorder=1)

        # Visualization for 2D
        else:
            for k in range(self.K):
                # Plot centroid k
                plt.scatter(x=centroids[k, 0],
                            y=centroids[k, 1],
                            c=[self.colors[k]], s=150,
                            marker='*', edgecolors='black', zorder=2)

                # Plot points associated with cluster k
                plt.scatter(x=points[nearest_idx[k], 0],
                            y=points[nearest_idx[k], 1],
                            c=[self.colors[k]], s=10, alpha=0.5, zorder=1)

        if self.fig_save_path is None:
            plt.show()
        else:
            directory = os.path.join(self.fig_save_path, self.__class__.__name__)
            if not os.path.exists(directory):
                os.mkdir(directory)
            plt.savefig(os.path.join(directory, f'{self.name}_K{self.K}_{self.it}.png'))
        plt.close()

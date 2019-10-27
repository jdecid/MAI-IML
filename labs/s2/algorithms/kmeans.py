import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler


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
        assert K > 0, 'K must be a positive number > 0'
        assert vis_dims in [0, 2, 3], 'visualize must be an integer in {0, 2, 3}'
        assert metric in ['euclidean', 'cityblock', 'cosine'], \
            'accepted metrics are euclidean, manhattan and cosine distances'

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
        while it < max_it:
            print(f'Iteration {it}')

            # Calculate distance from each point to each cluster
            distances = np.zeros(shape=(self.K, X.shape[0]))
            for c_idx, centroid in enumerate(self.centroids):
                for p_idx, point in enumerate(X):
                    distances[c_idx, p_idx] = self._distance(centroid, point)

            # Get nearest points for each cluster
            nearest = [[] for _ in range(self.K)]
            nearest_idx = [[] for _ in range(self.K)]

            for p_idx, point in enumerate(X):
                min_index = int(np.argmin(distances[:, p_idx]))
                nearest[min_index].append(point)
                nearest_idx[min_index].append(p_idx)

            if self.vis_dims:
                self._display_iteration(X, nearest_idx)

            # Recalculate centroids as the mean of their nearest points
            self._compute_centroids(nearest)

            # Check convergence
            it += 1
            if previous_nearest_idx == nearest_idx:
                break
            else:
                previous_nearest_idx = nearest_idx

    def predict(self, X):
        # Calculate distance from each point to each cluster
        distances = np.zeros(shape=(self.K, X.shape[0]))
        for c_idx, centroid in enumerate(self.centroids):
            for p_idx, point in enumerate(X):
                distances[c_idx, p_idx] = self._distance(centroid, point)

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

    def _compute_centroids(self, nearest):
        # TODO: Ugly AF
        self.centroids = np.array(list(map(lambda x: np.mean(np.array(x), axis=0), nearest)))

    def _display_iteration(self, X, nearest_idx):
        """
        Visualize a plot for the current iteration centroids and point assignments.
        :param X: 2D vector with the set of points of size (instances, features).
        :param nearest_idx: List of size (K, ?) assigning each point to its nearest cluster.
        """
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

    def _distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute distance between 2 elements using the specified metric. Check metrics in:
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html
        :param a: 1D vector with all A attributes.
        :param b: 1D vector with all B attributes.
        :return:
        """
        return distance.cdist(np.array([a]), np.array([b]), metric=self.metric)[0][0]

    @staticmethod
    def _get_colors(n):
        """
        Sample RGBA colors from HSV matplotlib colormap.
        :param n: Number of colors to obtain.
        :return: List of n RGBA colors.
        """
        return [plt.cm.hsv(x / n) for x in range(n)]


if __name__ == '__main__':
    dataset = pd.read_csv('Mall_Customers.csv')
    dataset.describe()
    dataset = dataset.iloc[:, [2, 4]].values

    scaler = MinMaxScaler()
    dataset = scaler.fit_transform(dataset)

    kmeans = KMeans(K=5, metric='cosine', vis_dims=2)
    kmeans.fit_predict(dataset)

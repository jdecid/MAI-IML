import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from algorithms.kmeans import KMeans


class FuzzyCMeans(KMeans):
    def __init__(self, C: int, m: int, vis_dims: int):
        """

        :param C: Number of Clusters
        :param m: Degree of fuzziness (1 - crisp)
        """
        super().__init__(K=C, vis_dims=vis_dims)
        self.m = m

    def _init_centroids(self):
        super()._init_centroids()
        self._init_u()

    def _init_u(self):
        u = np.random.random(size=(self.K, self.X.shape[0]))
        self.u = u / u.sum(axis=0)[None, :]

    def _update_u(self, X: np.ndarray):
        # TODO: Remove
        # self.centroids = np.random.random(size=(self.K, X.shape[1]))

        print(self.u, self.u.shape, '\n')

        # TODO: Vectorize
        for i in range(self.u.shape[0]):
            for k in range(self.u.shape[1]):
                uik = 0
                num = np.linalg.norm(X[k, :] - self.centroids[i])
                for j in range(self.u.shape[0]):
                    den = np.linalg.norm(X[k, :] - self.centroids[j])
                    uik += (num / den) ** (2 / (self.m - 1))
                uik ** -1
                self.u[i, k] = uik

        # TODO: Necessary?
        self.u = self.u / self.u.sum(axis=0)[None, :]

        print(self.u, self.u.shape, '\n')

    def _compute_centroids(self, X):
        # TODO: Remove
        # self.centroids = np.random.random(size=(self.K, X.shape[1]))

        # for i in range(self.K):
        #     u_pow_m = self.u[i, :] ** self.m
        #
        #     self.centroids[i] = (u_pow_m.dot(X)).sum() / u_pow_m.sum()

        # TODO: Test and review
        u_pow_m = self.u ** self.m
        num = u_pow_m @ X
        self.centroids = num / num.sum(axis=0)

        # print(self.centroids, self.centroids.shape, '\n')
        # print(self.centroids2, self.centroids2.shape, '\n')

    def _display_iteration(self, X, nearest_idx):
        if self.vis_dims == 0:
            return

        f, ax = plt.subplots(self.K, 1, figsize=(self.K, 6))

        colors = self._get_colors(4)
        ax[0].set_title('Membership Functions')

        for idx in range(self.K):
            ax[idx].plot(self.u[idx, :], c=colors[idx])
            ax[idx].tick_params(labelsize=6)
            ax[idx].set_ylim((0, 1))

        plt.show()


if __name__ == '__main__':
    dataset = pd.read_csv('../tests/datasets/Mall_Customers.csv')
    dataset.describe()
    dataset = dataset.iloc[:, [2, 3, 4]].values

    sc = MinMaxScaler()
    dataset = sc.fit_transform(dataset)

    fcm = FuzzyCMeans(C=4, m=2, vis_dims=2)
    # kmeans.fit_predict(dataset)
    fcm.X = dataset
    fcm._init_u()
    fcm._display_iteration(dataset, None)

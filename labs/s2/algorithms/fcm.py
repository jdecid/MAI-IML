import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from algorithms.kmeans import KMeans


class FuzzyCMeans(KMeans):
    def __init__(self, C: int, m: int, vis_dims: int, epsilon: float = 0.01):
        """

        :param C: Number of Clusters
        :param m: Degree of fuzziness (1 - crisp)
        """
        super().__init__(K=C, vis_dims=vis_dims)
        self.m = m
        self.epsilon = epsilon

    def _init_centroids(self):
        super()._init_centroids()
        self._init_u()

    def _init_u(self):
        u = np.random.random(size=(self.K, self.X.shape[0]))
        self.u = u / u.sum(axis=0)[None, :]

    def _compute_centroids(self, *args):
        """
        Update c-partition matrix U and centroids (centers of gravity) V.
        :param args: Ignored for inheritance interface design.
        """
        print('Loss BEFORE', self._loss())
        self._update_v()
        print('Loss AFTER update_v()', self._loss())
        self._update_u()
        print('Loss AFTER update_u()', self._loss())
        print('--------------------')

    def _update_v(self):
        """
        Update centroids (centers of gravity) V.
        v_k = ∑_i ((U_ki ^ m) * x_i) / ∑_i (U_ki ^ m)
        """

        u_pow_m = self.u ** self.m
        n_term = u_pow_m @ self.X
        d_term = u_pow_m.sum(axis=1, keepdims=True)
        self.centroids = n_term / d_term

    def _loss(self):
        res = 0
        for k in range(self.X.shape[0]):
            for i in range(self.K):
                res += (self.u[i,k]**self.m) * np.linalg.norm(self.X[k] - self.centroids[i])**2
        return res

    def _update_u(self):
        # TODO: Vectorize
        for i in range(self.u.shape[0]):
            for k in range(self.u.shape[1]):
                uik = 0
                num = np.linalg.norm(self.X[k, :] - self.centroids[i])
                for j in range(self.u.shape[0]):
                    den = np.linalg.norm(self.X[k, :] - self.centroids[j])
                    uik += (num / den) ** (2 / (self.m - 1))
                uik ** -1
                self.u[i, k] = uik

        # TODO: Necessary?
        self.u = self.u / self.u.sum(axis=0)[None, :]

    def _check_convergence(self, it, max_it, previous_centroids):
        if it >= max_it:
            return True
        if previous_centroids is not None:
            return np.linalg.norm(self.centroids - previous_centroids, ord=1) < self.epsilon
        return False

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
    fcm.fit_predict(dataset)
    # fcm.X = dataset
    # fcm._init_u()
    # fcm._display_iteration(dataset, None)

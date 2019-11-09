from typing import Union

import numpy as np
from sklearn.decomposition import PCA as SKPCA


class PCA:
    """Principal component analysis (PCA)

    Parameters
    ----------
    n_components : int, float, None
        Number of components to keep.
        - If parameter is an int `n_components` >= 1, it is number of selected components.
        - If parameter is a float 0 < `n_components` < 1, it selects the number of components that explain at least
            this a variance equivalent to this value.
        - If `n_components` is None, all components are kept: n_components == n_features.

    Attributes
    ----------
    components_: np.array, shape (`n_components_`, d_features)
        Principal axes in feature space of directions with maximum variance in the data.
        The components are sorted by `explained_variance_`.

    explained_variance_: np.array

    explained_variance_ratio_: np.array, shape (`n_components_`,)

    singular_values_: np.array, shape(`n_components`,)

    mean_: np.array, shape(d_features,)
        Mean vector for all features, estimated empirically from the data.

    n_components_: int
        Number of components selected to use:
        - Equivalent to parameter `n_components` if it is an int.
        - Estimated from the input data if it is a float.
        - Equivalent to d_features if not specified.
    """

    def __init__(self, n_components: Union[int, float, None]):
        # Parameters
        self._n_components = n_components

        # Attributes
        self.components_: np.ndarray = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None

    def fit(self, X: np.ndarray) -> 'PCA':
        """Fit the model with X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, d_features)
            Training data where
            - n_samples is the number of samples
            - d_features is the number of features

        Returns
        -------
        self : PCA object
            Returns the instance itself.
        """
        self.mean_ = np.mean(X, axis=0)

        phi_mat = (X - self.mean_).T

        cov_mat = phi_mat @ phi_mat.T

        singular_values, singular_vectors = np.linalg.eig(cov_mat)

        eig = list(zip(singular_values, singular_vectors))
        eig.sort(key=lambda x: x[0], reverse=True)
        singular_values, singular_vectors = zip(*eig)

        self.singular_values_ = np.array(singular_values)
        self.explained_variance_ = (self.singular_values_ ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()

        if type(self._n_components) == int:
            k = self._n_components
        elif type(self._n_components) == float:
            k = np.searchsorted(self.explained_variance_ratio_.cumsum(), self._n_components) + 1
        else:
            k = X.shape[1]

        self.components_ = np.stack(singular_vectors[:k], axis=0)
        self.n_components_ = k

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the dimensionality reduction on X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, d_features)
            Training data where
            - n_samples is the number of samples
            - d_features is the number of features

        Returns
        -------
        X_new : np.ndarray, shape (n_samples, self.n_components)
        """
        if self.components_ is None:
            raise Exception('Fit the model first before running a transformation')
        return (self.components_ @ (X - self.mean_).T).T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model with X and then apply the dimensionality reduction on X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, d_features)
            Training data where
            - n_samples is the number of samples
            - d_features is the number of features

        Returns
        -------
        X_new : np.ndarray, shape (n_samples, self.n_components)
        """
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('banana.dat')

    pca = PCA(n_components=0.50)
    X = np.random.random(size=(100, 10))
    Xp = pca.fit_transform(df.values)
    print(pca.n_components_, pca.explained_variance_ratio_)

    import matplotlib.pyplot as plt

    plt.scatter(Xp[:, 0], Xp[:, 1])
    plt.show()
    plt.close()

    Xp2 = SKPCA(n_components=2).fit_transform(df.values)

    plt.scatter(Xp2[:, 0], Xp2[:, 1])
    plt.show()
    plt.close()

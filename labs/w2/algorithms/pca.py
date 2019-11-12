import os
from typing import Union

import matplotlib.pyplot as plt
import numpy as np


class PCA:
    """Principal component analysis (PCA)

    Algorithm to transform multi-dimensional observations into a set of values of linearly uncorrelated variables called
    principal components. The first principal component has the largest possible variance
    (that is, accounts for as much of the variability in the data as possible), and each succeeding component in turn
    has the highest variance possible under the constraint that it is orthogonal to the preceding components.
    The resulting vectors (each being a linear combination of the variables and containing n observations)
    are an uncorrelated orthogonal basis set. PCA is sensitive to the relative scaling of the original variables.

    Parameters
    ----------
    n_components : int, float, None
        Number of components to keep.
        - If parameter is an int `n_components` >= 1, it is the number of selected components.
        - If parameter is a float 0 < `n_components` < 1, it selects the number of components that explain at least
            a variance equivalent to this value.
        - If `n_components` is None, all components are kept: n_components == n_features.

    name: str
        Identifiable name of the dataset used for plotting and printing purposes.

    solver: str
        Solving method for extracting eigenvalues. It may take three different values:
        - `eig`: Uses `np.linalg.eig` to compute the eigenvalues and right eigenvectors of the covariance matrix.
        - `hermitan`: Uses `np.linalg.eigh` which assumes the matrix is Hermitan (i.e. symmetric)
            By default takes only the lower triangular part of the matrix and assumes that the upper triangular part
            is defined by the symmetry of the matrix. This one is faster to compute.
        - `svd`: Uses `np.linalg.svd` which is exactly the same for Hermitan matrices. However it's faster and
            numerically more stable than calculate the eigenvalues and eigenvectors as it uses a Divide and Conquer
            approaches instead of a plain QR factorization, which are less stable.

    fig_save_path : str, None
        Output path to save all generated figures. If None, those would be shown instead of being saved.

    Attributes
    ----------
    components_ : np.array, shape (`n_components_`, d_features)
        Principal axes in feature space of directions with maximum variance in the data.
        The components are sorted by `explained_variance_`.

    explained_variance_ : np.array

    explained_variance_ratio_ : np.array, shape (`n_components_`,)

    singular_values_ : np.array, shape(`n_components`,)

    mean_: np.array, shape(d_features,)
        Mean vector for all features, estimated empirically from the data.

    n_components_ : int
        Number of components selected to use:
        - Equivalent to parameter `n_components` if it is an int.
        - Estimated from the input data if it is a float.
        - Equivalent to d_features if not specified.
    """

    def __init__(self, n_components: Union[int, float, None], name: str, solver='eig', fig_save_path: str = None):
        if solver not in ['eig', 'hermitan', 'svd']:
            raise ValueError('Solver must be "eig", "hermitan", "svd"')

        # Parameters
        self._n_components = n_components
        self._fig_save_path = fig_save_path
        self._name = name
        self._solver = solver

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
            Training data where:
            - n_samples is the number of rows (samples).
            - d_features is the number of columns (features).

        Returns
        -------
        self : PCA object
            Returns the instance itself.
        """
        self.mean_ = np.mean(X, axis=0)

        phi_mat = (X - self.mean_).T

        cov_mat = phi_mat @ phi_mat.T
        PCA.__save_cov_matrix(cov_mat, self._name, self._fig_save_path)

        if self._solver == 'eig':
            # Using Eigenvalues and Eigenvectors
            eig_values, eig_vectors = np.linalg.eig(cov_mat)
            eig_values, eig_vectors = PCA.__sort_eigen(eig_values, eig_vectors)
        elif self._solver == 'hermitan':
            # Using Eigenvalues and Eigenvectors assuming Hermitan matrix
            eig_values, eig_vectors = np.linalg.eigh(cov_mat)
            eig_values, eig_vectors = PCA.__sort_eigen(eig_values, eig_vectors)
        else:
            # Using Singular Value Decomposition
            _, eig_values, eig_vectors = np.linalg.svd(cov_mat, compute_uv=True)

        # PCA.__display_eig(singular_values, singular_vectors)

        self.explained_variance_ = (eig_values ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / self.explained_variance_.sum()

        if type(self._n_components) == int:
            k = self._n_components
        elif type(self._n_components) == float:
            k = np.searchsorted(self.explained_variance_ratio_.cumsum(), self._n_components) + 1
        else:
            k = X.shape[1]

        self.components_ = eig_vectors[:k, :]
        self.n_components_ = k

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply the dimensionality reduction on X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, d_features)
            Test data where:
            - n_samples is the number of rows (samples).
            - d_features is the number of columns (features).

        Returns
        -------
        X_transformed : np.ndarray, shape (n_samples, self.n_components)
        """
        if self.components_ is None:
            raise Exception('Fit the model first before running a transformation')
        return (self.components_ @ (X - self.mean_).T).T

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit the model with X and then apply the dimensionality reduction on X.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, d_features)
            Training data where:
            - n_samples is the number of rows (samples).
            - d_features is the number of columns (features).

        Returns
        -------
        X_transformed : np.ndarray, shape (n_samples, self.n_components)
        """
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct original data.
        TODO: proper docstring

        Parameters
        ----------
        X : np.ndarray,

        Returns
        -------
        X_transformed : np.ndarray,
        """
        if self.components_ is None:
            raise Exception('Fit the model first before running a transformation')
        return X @ self.components_ + self.mean_

    @staticmethod
    def __display_eig(values, vectors):
        for i in range(len(values)):
            print(f'{i}) {values[i]}: {vectors[i, :]}')

    @staticmethod
    def __save_cov_matrix(mat: np.ndarray, name: str, save_path: str):
        f = plt.figure()
        plt.matshow(mat, cmap=plt.get_cmap('coolwarm'))
        plt.title(f'Covariance matrix for {name} dataset', y=1.08)
        plt.colorbar()
        if save_path is None:
            plt.show()
        else:
            plt.savefig(os.path.join(save_path, f'cov_mat_{name}.png'))
        plt.close(f)

    @staticmethod
    def __sort_eigen(values, vectors):
        eig = list(zip(values, vectors.T))
        eig.sort(key=lambda x: x[0], reverse=True)
        values, vectors = zip(*eig)
        values, vectors = np.array(values), np.array(vectors)
        return values, vectors

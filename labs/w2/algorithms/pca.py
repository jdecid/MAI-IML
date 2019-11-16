import logging
from typing import Union, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils.plotting import mat_print


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

    Attributes
    ----------
    components_ : np.array, shape (`n_components`, `d_features`)
        Principal axes in feature space of directions with maximum variance in the data.
        The components are sorted from largest to smallest associated eigenvalue,
        which is equivalent to sort descending by explained variance.

    cov_mat_ : np.array, shape (`d_features`, `d_features`)
        Covariance matrix of the training data.

    explained_variance_ : np.array
        The amount of variance explained by each of the selected `n_components`.
        Equivalent to n_components largest eigenvalues of `cov_mat`.

    explained_variance_ratio_ : np.array, shape (`n_components`,)
        Percentage of the amount of variance explained by each of the selected `n_components`.
        If all the components are stored, this sum of all ratios is equal to 1.0.

    eigen_values_ : np.array, shape(`n_components`,)
        Singular values associated to each of the selected `n_components` (eigenvectors).

    mean_: np.array, shape(`d_features`,)
        Mean vector for all features, estimated empirically from the training data.

    n_components_ : int
        Number of components selected to use:
        - Equivalent to parameter `n_components` if it is an int.
        - Estimated from the input data if it is a float.
        - Equivalent to d_features if not specified.
    """

    def __init__(self, n_components: Union[int, float, None], name: str, solver='eig'):
        if solver not in ['eig', 'hermitan', 'svd']:
            raise ValueError('Solver must be "eig", "hermitan", "svd"')

        # Parameters
        self._n_components = n_components
        self._name = name
        self._solver = solver

        # Attributes
        self.components_: np.ndarray = None
        self.cov_mat_: np.ndarray = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.eigen_values_ = None
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

        self.cov_mat_ = phi_mat @ phi_mat.T

        if self._solver == 'eig':
            # Using Eigenvalues and Eigenvectors
            eig_values, eig_vectors = np.linalg.eig(self.cov_mat_)
            eig_values, eig_vectors = PCA.__sort_eigen(eig_values, eig_vectors)
        elif self._solver == 'hermitan':
            # Using Eigenvalues and Eigenvectors assuming Hermitan matrix
            eig_values, eig_vectors = np.linalg.eigh(self.cov_mat_)
            eig_values, eig_vectors = PCA.__sort_eigen(eig_values, eig_vectors)
        else:
            # Using Singular Value Decomposition
            _, eig_values, eig_vectors = np.linalg.svd(self.cov_mat_, compute_uv=True)

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
        self.eigen_values_ = eig_values[:k]
        self.n_components_ = k

        logging.info(f'Original Matrix:\n{mat_print(X)}')
        logging.info(f'Covariance Matrix:\n{mat_print(self.cov_mat_)}')
        logging.info(f'Eigenvalues:\n{mat_print(self.eigen_values_)}')
        logging.info(f'Eigenvectors:\n{mat_print(self.components_)}')
        logging.info(f'Explained Variance:\n{mat_print(self.explained_variance_ratio_)}')
        logging.info(f'Cumulative Explained Variance:\n{mat_print(np.cumsum(self.explained_variance_ratio_))}')

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

        X_transformed = (self.components_ @ (X - self.mean_).T).T
        logging.info(f'Transformed X:\n{mat_print(X_transformed)}')
        return X_transformed

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

    def inverse_transform(self, X_transformed: np.ndarray) -> np.ndarray:
        """Reconstruct X_original from X which would be its transformation.

        Parameters
        ----------
        X_transformed : np.ndarray, shape (`n_shape`, `n_components`)
            Transformed X from original data obtained with `transform` or `fit_transform`.

        Returns
        -------
        X_original : np.ndarray, shape (n_samples, d_features)
        """
        if self.components_ is None:
            raise Exception('Fit the model first before running a transformation')
        return X_transformed @ self.components_ + self.mean_

    def get_cov_matrix(self, dataset_name: str) -> Tuple[plt.Figure, plt.Axes]:
        """Create covariance matrix Matplotlib figure.

        Parameters
        ----------
        dataset_name : str
            Dataset name used in the figure title.

        Returns
        -------
        Cov_figure : plt.Axes

        """
        f, ax = plt.subplots(1, 1)
        im = ax.matshow(self.cov_mat_, cmap=plt.get_cmap('coolwarm'))
        ax.set_title(f'Covariance matrix for {dataset_name} dataset', y=1.08)
        plt.colorbar(im, ax=ax)
        return f, im

    @staticmethod
    def __display_eig(values, vectors):
        for i in range(len(values)):
            print(f'{i}) {values[i]}: {vectors[i, :]}')

    @staticmethod
    def __sort_eigen(values, vectors):
        eig = list(zip(values, vectors.T))
        eig.sort(key=lambda x: x[0], reverse=True)
        values, vectors = zip(*eig)
        values, vectors = np.array(values), np.array(vectors)
        return values, vectors

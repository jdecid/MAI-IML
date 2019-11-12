from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA

from algorithms.pca import PCA as IMLPCA


def compare_pca(X: np.ndarray, n_components: List[int]):
    for n in n_components:
        f, ax = plt.subplots(1, 3, figsize=(10, 3))
        f.tight_layout()

        # Our PCA
        pca = IMLPCA(n_components=n)
        results_pca = pca.fit_transform(X)

        ax[0].set_title('Custom PCA')
        ax[0].scatter(results_pca[:, 0], results_pca[:, 1],
                      c='darkred', s=10, alpha=0.5)

        # PCA
        pca = PCA(n_components=n)
        results_pca = pca.fit_transform(X)

        ax[1].set_title('SKLearn PCA')
        ax[1].scatter(results_pca[:, 0], results_pca[:, 1],
                      c='darkblue', s=10, alpha=0.5)

        # Incremental PCA
        ipca = IncrementalPCA(n_components=n)
        results_ipca = ipca.fit_transform(X)

        ax[2].set_title('SKLearn Incremental PCA')
        ax[2].scatter(results_ipca[:, 0], results_ipca[:, 1],
                      c='teal', s=10, alpha=0.5)

        plt.show()
        plt.close(f)


if __name__ == '__main__':
    df = pd.read_csv('datasets/segment_clean.csv')
    compare_pca(df.values, [2])

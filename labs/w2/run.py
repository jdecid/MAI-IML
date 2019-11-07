from typing import List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, IncrementalPCA

df = pd.read_csv('datasets/segment_clean.csv')


def compare_pca(X: np.ndarray, n_components: List[int]):
    for n in n_components:
        f, ax = plt.subplots(1, 3, figsize=(8, 2))

        # Our PCA
        # TODO
        ax[0].set_title('Custom PCA')
        ax[0].scatter([0.5], [0.5],
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

        ax[2].set_title('SKLearn PCA')
        ax[2].scatter(results_ipca[:, 0], results_ipca[:, 1],
                      c='teal', s=10, alpha=0.5)

        plt.show()
        plt.close(f)


compare_pca(df.values, [2, 3, 4, 5])

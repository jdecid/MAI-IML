import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from tqdm import tqdm

from utils.optimize import store_predictions
from utils.plotting import get_colors

AFFINITIES = ['euclidean', 'cosine']
LINKAGES = ['complete', 'average', 'single']


def agglomerative_clustering(X: np.ndarray, K: int, name: str, fig_save_path: str = None):
    """
    Run agglomerative clustering with six different combinations between:
        - Affinities: `euclidean` or `cosine`.
        - Linkages: `complete`, `average`, `single`.
    :param X: 2D input data matrix of shape (#instances, #features).
    :param K: Number of clusters.
    :param fig_save_path: Whether to save or plot figure
    :return: List of clustering predictions with the corresponding affinity and linkage methods.
    """
    pb = tqdm(total=len(AFFINITIES) * len(LINKAGES), ncols=100)

    results = []
    for affinity in AFFINITIES:
        for linkage in LINKAGES:
            clustering = AgglomerativeClustering(n_clusters=K,
                                                 affinity=affinity,
                                                 linkage=linkage)

            prediction = clustering.fit_predict(X)
            store_predictions(prediction, 'Agglomerative', f'{name}_{affinity}_{linkage}', K, fig_save_path)

            results.append({
                'affinity': affinity,
                'linkage': linkage,
                'prediction': prediction
            })

            pca = PCA(n_components=2)
            points = pca.fit_transform(X)

            colors = get_colors(K)

            plt.figure()
            plt.title(f'Agglomerative Clustering K={K}, affinity={affinity} and linkage={linkage}')

            for k in range(K):
                plt.scatter(x=points[np.where(prediction == k)][:, 0],
                            y=points[np.where(prediction == k)][:, 1],
                            c=[colors[k]], s=10, zorder=1)

            if fig_save_path is None:
                plt.show()
            else:
                log_directory = os.path.join(fig_save_path, 'Agglomerative')
                if not os.path.exists(log_directory):
                    os.mkdir(log_directory)
                plt.savefig(os.path.join(log_directory, f'agglomerative_{name}_{affinity}_{linkage}_{K}.png'))
            plt.close()

            pb.update()

    return results

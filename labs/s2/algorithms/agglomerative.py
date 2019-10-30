import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

AFFINITIES = ['euclidean', 'cosine']
LINKAGES = ['complete', 'average', 'single']


def _get_colors(n):
    """
    Sample RGBA colors from HSV matplotlib colormap.
    :param n: Number of colors to obtain.
    :return: List of n RGBA colors.
    """
    return [plt.cm.hsv(x / n) for x in range(n)]


def agglomerative_clustering(X: np.ndarray, K: int):
    for affinity in AFFINITIES:
        for linkage in LINKAGES:
            clustering = AgglomerativeClustering(n_clusters=K,
                                                 affinity=affinity,
                                                 linkage=linkage)

            prediction = clustering.fit_predict(X)

            pca = PCA(n_components=2)
            points = pca.fit_transform(X)

            colors = _get_colors(K)

            plt.figure()
            plt.title(f'Agglomerative Clustering with {affinity} affinity and {linkage} linkage')

            for k in range(K):
                plt.scatter(x=points[np.where(prediction == k)][:, 0],
                            y=points[np.where(prediction == k)][:, 1],
                            c=[colors[k]], s=10, zorder=1)

            plt.show()


if __name__ == '__main__':
    dataset = pd.read_csv('tests/datasets/iris.csv')

    dataset = dataset.iloc[:, :4].values

    sc = MinMaxScaler()
    dataset = sc.fit_transform(dataset)

    agglomerative_clustering(dataset, 3)

#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering
# Imports
from sklearn.preprocessing import MinMaxScaler

from utils.dataset import read_dataset


def preprocess():
    # Read data
    dataset = read_dataset('segment')
    data = dataset['data']

    df = pd.DataFrame(data)
    X = df.drop(columns=['class', 'region-pixel-count'])
    # X.head()

    # Normalize
    scaler = MinMaxScaler()
    X[:] = scaler.fit_transform(X)

    # X.head()

    # Agglomerative clustering
    linkages = ['complete', 'average', 'single']
    affinities = ['euclidean', 'cosine']

    cluster_models = []
    for linkage in linkages:
        for affinity in affinities:
            cluster_models.append(AgglomerativeClustering(affinity=affinity, linkage=linkage).fit(df))

    print(cluster_models)

    def plot_dendrogram(model, **kwargs):
        # Children of hierarchical clustering
        children = model.children_

        # Distances between each pair of children
        # Since we don't have this information, we can use a uniform one for plotting
        distance = np.arange(children.shape[0])

        # The number of observations contained in each cluster level
        no_of_observations = np.arange(2, children.shape[0] + 2)

        # Create linkage matrix and then plot the dendrogram
        linkage_matrix = np.column_stack([children, distance, no_of_observations]).astype(float)

        # Plot the corresponding dendrogram
        dendrogram(linkage_matrix, **kwargs)

    for model in cluster_models:
        plot_dendrogram(model, labels=model.labels_, truncate_mode='level', p=10)
    plt.show()

    # Write data
    with open('datasets/segment-clean.csv', mode='w') as f:
        X.to_csv(f, index=False)

    return 'segment-clean.csv'

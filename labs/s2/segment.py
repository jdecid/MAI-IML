#!/usr/bin/env python
# coding: utf-8

# Imports
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import AgglomerativeClustering

import pandas as pd
from utils import read_dataset


# Read data
dataset = read_dataset('segment')
data = dataset['data']

df = pd.DataFrame(data)
df = df.drop(columns=['class', 'region-pixel-count'])
#df.head()

# Normalize
scaler = MinMaxScaler()
df[:] = scaler.fit_transform(df)

#df.head()

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


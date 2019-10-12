from sklearn.cluster import AgglomerativeClustering

from .utils import read_dataset

AFFINITIES = ['euclidean', 'cosine']
LINKAGES = ['complete', 'average', 'single']

dataset = read_dataset()

for affinity in AFFINITIES:
    for linkage in LINKAGES:
        clustering = AgglomerativeClustering(affinity=affinity, linkage=linkage).fit(dataset)

from sklearn.metrics import adjusted_rand_score
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial import distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from algorithms.kmodes import KModes
#from algorithms.kmeans import KMeans
def evaluate(labels_true, labels_pred):
    return adjusted_rand_score(labels_true, labels_pred)

def main():
    dataset = pd.read_csv('datasets/car.data')
    X = dataset.iloc[:, :6]
    kmodes = KModes(K=4, seed=1)
    res = kmodes.fit_predict(X.values)
    with open('res.txt', 'w') as f:
        f.write(str(res))
    # c = collections.Counter()
    # c.update(res)
    # print(c)
    Y = dataset.iloc[:, 6]
    # print(Y.value_counts())
    print(evaluate(labels_true=Y, labels_pred=res))
    print('hola')


if __name__ == '__main__':
    main()

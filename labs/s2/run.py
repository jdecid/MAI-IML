import os
import logging
import pandas as pd

from algorithms.kmeans import KMeans
from algorithms.kmodes import KModes
from preprocessing import connect_4, adult, segment


def main():
    """
    Runs EVERYTHING (preprocessing, clustering, evaluation,...), saves images, logs, results etc for the report
    :return:
    """
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.DEBUG)

    print('Preprocessing...')
    # file_c4, file_c4_enc = connect_4.preprocess()
    # file_adult, file_adult_enc = adult.preprocess()
    file_segment = segment.preprocess()

    print('Applying hierarchical clustering...')
    # TODO

    print('KMeans')

    X = pd.read_csv(os.path.join('datasets', file_segment))

    kmeans = KMeans(K=3, vis_dims=2)
    kmeans.fit_predict(X.values)

    print('KModes')

    X = pd.read_csv(os.path.join('datasets', file_c4))

    kmeans = KModes(K=3)
    kmeans.fit_predict(X.values)


if __name__ == '__main__':
    main()

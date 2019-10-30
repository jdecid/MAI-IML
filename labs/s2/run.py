import os
import logging
import pandas as pd

from algorithms.kmeans import KMeans
from preprocessing import connect_4, adult


def main():
    """
    Runs EVERYTHING (preprocessing, clustering, evaluation,...), saves images, logs, results etc for the report
    :return:
    """
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.DEBUG)

    print('Preprocessing...')
    file_c4, file_c4_enc = connect_4.preprocess()
    # file_adult, file_adult_enc = adult.preprocess()


    print('Applying hierarchical clustering...')
    # TODO

    X = pd.read_csv(os.path.join('datasets', file_c4_enc))

    kmeans = KMeans(K=3, vis_dims=3)
    kmeans.fit_predict(X.values)


if __name__ == '__main__':
    main()

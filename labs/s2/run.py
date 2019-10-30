import os
import logging
import pandas as pd
from sklearn.metrics import calinski_harabasz_score

from algorithms.kmeans import KMeans
from algorithms.kmodes import KModes
from preprocessing import connect_4, adult, segment
from utils.optimize import optimize


def main():
    """
    Runs EVERYTHING (preprocessing, clustering, evaluation,...), saves images, logs, results etc for the report
    :return:
    """
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger().setLevel(logging.DEBUG)

    print('Preprocessing...')
    # file_c4, file_c4_enc = connect_4.preprocess()
    file_adult, file_adult_enc = adult.preprocess()
    # file_segment = segment.preprocess()

    print('Applying hierarchical clustering...')
    # TODO

    # ------------------------------------------------------ #

    print('KMeans')

    X = pd.read_csv(os.path.join('datasets', file_adult_enc))

    # kmeans = KMeans(K=2, vis_dims=2)
    # kmeans.fit_predict(X.values)

    # Optimization of K

    result = optimize(X=X.values,
                      algorithm=KMeans,
                      algorithm_params={},
                      metric=calinski_harabasz_score,
                      metric_params={'X': X.values},
                      k_values=[2, 3, 4, 5],
                      goal='minimize')

    print(result)

    # ------------------------------------------------------ #


if __name__ == '__main__':
    main()

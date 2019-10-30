import pickle
import unittest

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from algorithms.fcm import FuzzyCMeans


class KMeansTest(unittest.TestCase):
    def setUp(self):
        sc = MinMaxScaler()

        dataset = pd.read_csv('datasets/iris.csv')
        dataset = dataset.iloc[:, :4].values
        self.dataset = sc.fit_transform(dataset)

        self.seed = 42

    def test_fuzzy_c_means_predictions(self):
        with open('datasets/iris.pkl', mode='rb') as f:
            expected_predictions = pickle.load(f)['predictions']

        fcm = FuzzyCMeans(C=3, m=2, vis_dims=0, seed=self.seed)

        predictions = fcm.fit_predict(self.dataset)

        self.assertEqual(predictions, expected_predictions)

    def test_fuzzy_c_means_loss(self):
        with open('datasets/iris.pkl', mode='rb') as f:
            expected_loss = pickle.load(f)['loss']

        fcm = FuzzyCMeans(C=3, m=2, vis_dims=0, seed=self.seed)
        fcm.fit(self.dataset)
        loss = fcm._loss()

        self.assertEqual(loss, expected_loss)


if __name__ == '__main__':
    unittest.main()

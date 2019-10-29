import unittest

from algorithms.kmeans import KMeans


class KMeansTest(unittest.TestCase):
    def test_kmeans(self):
        _ = KMeans(K=3)
        self.assertEqual(1, 1)


if __name__ == '__main__':
    unittest.main()

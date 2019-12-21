from collections import Counter
from typing import List

import numpy as np
from scipy.spatial.distance import minkowski

from utils.exceptions import RetentionPolicyException, VotingPolicyException

VOTING_POLICIES = ['MVS', 'MP']
RETENTION_POLICIES = ['NR', 'AR', 'DF', 'DD']


class KIBLAlgorithm:
    def __init__(self, K: int, voting_policy: str = 'MVS', retention_policy: str = 'NR'):
        if voting_policy not in VOTING_POLICIES:
            raise VotingPolicyException()
        if retention_policy not in RETENTION_POLICIES:
            raise RetentionPolicyException()

        self.K = K
        self.voting_policy = voting_policy
        self.retention_policy = retention_policy

        self.X = None
        self.y = None

        self.classes = 0

    def fit(self, X, y):
        self.X = X
        self.y = y

        self.classes = len(set(y))

    def k_neighbours(self, X: np.ndarray, y: int = None) -> List[int]:
        similarity = list(map(lambda x: KIBLAlgorithm.__distance_function(x, X), self.X))
        k_nearest = similarity.sort()[:self.K]
        y_pred = self.__vote(k_nearest)

        self.__apply_retention_policy(X, y, y_pred, k_nearest)

        return y_pred

    def __vote(self, k_most_similar: List[int]):
        counter = Counter(k_most_similar)
        max_occurrence = max(counter.values())
        most_common = [k for k in counter.keys() if counter[k] == max_occurrence]
        if self.voting_policy == 'MVS':
            return most_common[np.random.randint(len(most_common))]
        else:
            if len(most_common) > 1:
                return self.__vote(k_most_similar[:-self.K])
            else:
                return most_common[0]

    def __apply_retention_policy(self, X: np.ndarray, y: int, y_pred: int, k_nearest: np.ndarray):
        if self.retention_policy == 'AR':
            self.X.append(X)
            self.y.append(y)

        elif self.retention_policy == 'DF':
            if y != y_pred:
                self.X.append(X)
                self.y.append(y)

        elif self.retention_policy == 'DD':
            counter = Counter(k_nearest)
            majority_cases = max(counter.values())
            d = (self.K - majority_cases) / (self.classes - 1) * majority_cases
            if d > 0.5:
                self.X.append(X)
                self.y.append(y)

    @staticmethod
    def __distance_function(u: np.ndarray, v: np.ndarray, r=2):
        return minkowski(u, v, r)

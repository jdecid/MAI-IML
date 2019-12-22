from collections import Counter
from typing import List, Tuple

import numpy as np

from utils.exceptions import RetentionPolicyException, VotingPolicyException

VOTING_POLICIES = ['MVS', 'MP']
RETENTION_POLICIES = ['NR', 'AR', 'DF', 'DD']


class KIBLAlgorithm:
    def __init__(self, K: int, voting_policy: str = 'MVS', retention_policy: str = 'NR', r=2):
        if voting_policy not in VOTING_POLICIES:
            raise VotingPolicyException()
        if retention_policy not in RETENTION_POLICIES:
            raise RetentionPolicyException()

        self.K = K
        self.r = r

        self.voting_policy = voting_policy
        self.retention_policy = retention_policy

        self.X = None
        self.y = None

        self.classes = 0

    def fit(self, X, y) -> 'KIBLAlgorithm':
        self.X = X
        self.y = y
        self.classes = len(set(y))
        return self

    def k_neighbours(self, X: np.ndarray, y: int = None) -> List[int]:
        distances = self.__distance_function(self.X, X, self.r)
        # distances = list(enumerate(distances))
        # distances = list(map(lambda x: KIBLAlgorithm.__distance_function(x, X), self.X))
        k_nearest_idx = np.argsort(distances)[:self.K]
        y_pred = self.__vote(self.y[k_nearest_idx])

        self.__apply_retention_policy(X, y, y_pred, self.y[k_nearest_idx])

        return y_pred

    def __vote(self, k_most_similar: List[Tuple[int]]):
        counter = Counter(k_most_similar)
        max_occurrence = max(counter.values())
        most_common = [k for k in counter.keys() if counter[k] == max_occurrence]
        if self.voting_policy == 'MVS':
            return most_common[np.random.randint(len(most_common))]
        else:
            if len(most_common) > 1:
                return self.__vote(k_most_similar[:-1])
            else:
                return most_common[0]

    def __apply_retention_policy(self, X: np.ndarray, y: int, y_pred: int, k_nearest: np.ndarray):
        if self.retention_policy == 'AR':
            self.X = np.vstack((self.X, X))
            self.y = np.concatenate((self.y, [y]))

        elif self.retention_policy == 'DF':
            if y != y_pred:
                self.X = np.vstack((self.X, X))
                self.y = np.concatenate((self.y, [y]))

        elif self.retention_policy == 'DD':
            counter = Counter(k_nearest)
            majority_cases = max(counter.values())
            d = (self.K - majority_cases) / (self.classes - 1) * majority_cases
            if d > 0.5:
                self.X = np.vstack((self.X, X))
                self.y = np.concatenate((self.y, [y]))

    @staticmethod
    def __distance_function(u: np.ndarray, v: np.ndarray, r: int):
        return np.linalg.norm(u - v, axis=1, ord=r)
        # return minkowski(u, v, r)

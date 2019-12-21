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

    def fit(self, X, y):
        self.X = X
        self.y = y

    def k_neighbours(self, current_instance: np.ndarray) -> List[int]:
        similarity = list(map(lambda x: KIBLAlgorithm.__distance_function(x, current_instance), self.X))
        k_nearest = similarity.sort()[:self.K]
        nearest = self.__vote(k_nearest)

        self.__apply_retention_policy(current_instance, nearest)

        return nearest

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

    def __apply_retention_policy(self, new_X: np.ndarray, new_y):
        if self.retention_policy == 'AR':
            self.X.append(new_X)
            self.y.append(new_y)

        elif self.retention_policy == 'DF':
            pass

        elif self.retention_policy == 'DD':
            pass

    @staticmethod
    def __distance_function(u: np.ndarray, v: np.ndarray, r=2):
        return minkowski(u, v, r)

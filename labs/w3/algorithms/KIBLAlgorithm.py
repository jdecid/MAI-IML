from collections import Counter
from typing import List

import numpy as np
from scipy.spatial.distance import minkowski

from utils.exceptions import RetentionPolicyException, VotingPolicyException


class KIBLAlgorithm:
    def __init__(self, voting_policy, retention_policy: str = 'NR'):
        if voting_policy not in ['MVS', 'MP']:
            raise VotingPolicyException()
        if retention_policy not in ['NR', 'AR', 'DF', 'DD']:
            raise RetentionPolicyException()

        self.voting_policy = voting_policy
        self.retention_policy = retention_policy
        self.X = None

    def get_most_similar(self, current_instance: np.ndarray, K: int) -> List[int]:
        similarity = list(map(lambda x: KIBLAlgorithm.distance_function(x, current_instance), self.X))
        similarity.sort()
        return similarity[:K]

    def vote(self, k_most_similar: List[int]):
        counter = Counter(k_most_similar)
        max_occurrence = max(counter.values())
        most_common = [k for k in counter.keys() if counter[k] == max_occurrence]
        if self.voting_policy == 'MVS':
            return most_common[np.random.randint(len(most_common))]
        else:
            if len(most_common) > 1:
                return self.vote(k_most_similar[:-1])
            else:
                return most_common[0]

    @staticmethod
    def distance_function(u: np.ndarray, v: np.ndarray, r=2):
        return minkowski(u, v, r)


KIBLAlgorithm('asdasd')

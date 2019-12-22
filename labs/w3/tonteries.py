import argparse
import json
import multiprocessing
import os
import pickle
from copy import deepcopy
from multiprocessing.pool import ThreadPool
from threading import Lock
from time import time
from typing import List

import numpy as np
from scipy.stats import wilcoxon, ttest_ind
from tqdm import tqdm

from algorithms.KIBLAlgorithm import KIBLAlgorithm, VOTING_POLICIES, RETENTION_POLICIES
from algorithms.reduction_KIBL_algorithm import reduction_KIBL_algorithm, REDUCTION_METHODS
from preprocessing.hypothyroid import preprocess as preprocess_hypothyroid
from preprocessing.pen_based import preprocess as preprocess_penn
from utils.dataset import read_dataset
from utils.exceptions import TestMethodException

K_VALUES = [1, 3, 5, 7]
R_VALUES = [1, 2, 3]


i = 0
print('index\tk\tr\tvoting_policy\tretention_policy')
for k in K_VALUES:
    for r in R_VALUES:
        for voting_policy in VOTING_POLICIES:
            for retention_policy in RETENTION_POLICIES:
                print(i, k, r, voting_policy, retention_policy)
                i = i + 1
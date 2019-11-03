from sklearn.metrics import *
import numpy as np


def get_metrics_from_mat(contingency_matrix):
    # TODO: add precision, recall, F1_score
    accuracy = np.trace(contingency_matrix) / np.sum(contingency_matrix)
    return dict(accuracy=accuracy)


def evaluate_supervised(labels_true, labels_pred):
    contingency_matrix = cluster.contingency_matrix(labels_true, labels_pred)
    metrics_from_contingency_matrix = get_metrics_from_mat(contingency_matrix)
    res = dict(adjusted_mutual_info_score=adjusted_mutual_info_score(labels_true, labels_pred, 'arithmetic'),
               adjusted_rand_score=adjusted_rand_score(labels_true, labels_pred),
               completeness_score=completeness_score(labels_true, labels_pred),
               contingency_matrix=contingency_matrix,
               fowlkes_mallows_score=fowlkes_mallows_score(labels_true, labels_pred),
               homogeneity_score=homogeneity_score(labels_true, labels_pred),
               v_measure_score=v_measure_score(labels_true, labels_pred), **metrics_from_contingency_matrix)
    return res


def evaluate_unsupervised(X, labels):
    res = dict(calinski_harabasz_score=calinski_harabasz_score(X, labels),
               davies_bouldin_score=davies_bouldin_score(X, labels),
               silhouette_score=silhouette_score(X, labels))
    return res


def normalized_partition_coefficient(u):
    """
    :param u: membership matrix
    :return: normalized (range: [0,1] where 1 is hard) partition coeficient
    Note: # clusters = u.shape[0], # rows = u.shape[1]
    """
    return (np.sum(u ** 2) / u.shape[1] - 1 / u.shape[0]) / (1 - 1 / u.shape[0])


def partition_entropy(u):
    """
    :param u: membership matrix, with shape (# clusters, # rows)
    :return: partition entropy
    """
    return -np.sum(u * np.log(u) / u.shape[1])


def xie_beni(X, u, centroids):
    # TODO: Fuzzy degree m
    min_val = np.inf
    for t in range(centroids.shape[0]):
        for s in range(centroids.shape[0]):
            if t != s:
                new_val = np.linalg.norm(centroids[t] - centroids[s]) * 2
                if new_val < min_val:
                    min_val = new_val
    den = X.shape[0] * min_val
    num = 0
    for i in range(X.shape[0]):
        for k in range(centroids.shape[0]):
            num += ((u[k, i] ** 2) * (np.linalg.norm(X[i, :] - centroids[k, :]) ** 2))
    return num / den


def evaluate_soft(X, u, v):
    res = dict(normalized_partition_coefficient=normalized_partition_coefficient(u),
               partition_entropy=partition_entropy(u),
               xie_beni=xie_beni(X, u, v))
    return res

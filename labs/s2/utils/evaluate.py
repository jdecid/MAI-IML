from sklearn.metrics import *
import numpy as np


def get_metrics_from_mat(contingency_matrix):
    # TODO: add precision, recall, F1_score
    accuracy = np.trace(contingency_matrix)/np.sum(contingency_matrix)
    return dict(accuracy=accuracy)


def evaluate_supervised(labels_true, labels_pred):
    contingency_matrix = cluster.contingency_matrix(labels_true, labels_pred)
    metrics_from_contingency_matrix = get_metrics_from_mat(contingency_matrix)
    res = dict(adjusted_mutual_info_score=adjusted_mutual_info_score(labels_true, labels_pred),
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


def evaluate(labels_true, labels_pred, X):
    eval_sup = evaluate_supervised(labels_true, labels_pred)
    eval_unsup = evaluate_unsupervised(X, labels_pred)
    return dict(eval_sup=eval_sup, eval_unsup=eval_unsup)

from sklearn.metrics import *


def evaluate_supervised(labels_true, labels_pred):
    res = dict(adjusted_mutual_info_score=adjusted_mutual_info_score(labels_true, labels_pred),
               adjusted_rand_score=adjusted_rand_score(labels_true, labels_pred),
               completeness_score=completeness_score(labels_true, labels_pred),
               contingency_matrix=cluster.contingency_matrix(labels_true, labels_pred),
               fowlkes_mallows_score=fowlkes_mallows_score(labels_true, labels_pred),
               homogeneity_score=homogeneity_score(labels_true, labels_pred),
               v_measure_score=v_measure_score(labels_true, labels_pred))
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

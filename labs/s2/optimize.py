from math import inf


def init_best(goal):
    if goal == 'maximize':
        best_val = -inf
    else:
        best_val = inf
    return best_val


def better(goal, current_best_val, new_val):
    if goal == 'maximize':
        if new_val > current_best_val:
            return True
    else:
        if new_val < current_best_val:
            return True
    return False


def optimize(X, clustering_alg, clustering_alg_params, metric, metric_params, k_values, goal):
    assert goal in ['maximize', 'minimize']
    best_val = init_best(goal)
    best_k = None
    all_vals = {}
    for k in k_values:
        clust_alg = clustering_alg(K=k, **clustering_alg_params)
        clustering = clust_alg.fit_predict(X)
        new_val = metric(clustering, **metric_params)
        all_vals[k] = new_val
        if better(goal, best_val, new_val):
            best_k = k
            best_val = new_val
    return best_k, best_val, all_vals

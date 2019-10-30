from math import inf


def init_best(goal):
    if goal == 'maximize':
        best_val = -inf
    else:
        best_val = inf
    return best_val


def better(goal, current_best_val, new_val):
    if goal == 'maximize':
        return new_val > current_best_val
    else:
        return new_val < current_best_val


def optimize(X, algorithm, algorithm_params, metric, metric_params, k_values, goal):
    """

    :param X:
    :param algorithm:
    :param algorithm_params:
    :param metric:
    :param metric_params:
    :param k_values:
    :param goal:
    :return:
    """
    assert goal in ['maximize', 'minimize']
    best_val = init_best(goal)
    best_k = None
    all_vals = {}
    for k in k_values:
        clust_alg = algorithm(K=k, **algorithm_params)
        clustering = clust_alg.fit_predict(X)
        metric_params['labels'] = clustering
        new_val = metric(**metric_params)
        all_vals[k] = new_val
        if better(goal, best_val, new_val):
            best_k = k
            best_val = new_val
    return dict(best={best_k: best_val}, all_vals=all_vals)

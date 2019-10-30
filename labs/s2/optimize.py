from math import inf
from algorithms.kmeans import KMeans


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
        metric_params['labels'] = clustering
        new_val = metric(**metric_params)
        all_vals[k] = new_val
        if better(goal, best_val, new_val):
            best_k = k
            best_val = new_val
    return dict(best={best_k: best_val}, all_vals=all_vals)


def main():
    import pandas as pd
    from sklearn.metrics import calinski_harabasz_score
    #dataset = pd.read_csv('tests/datasets/iris.csv')
    #y = dataset.iloc[:, 4].values
    #X = dataset.iloc[:,:4].values
    dataset = pd.read_csv('tests/datasets/Mall_Customers.csv')
    X = dataset.iloc[:,[2,3,4]].values
    from sklearn.preprocessing import MinMaxScaler
    sc = MinMaxScaler()
    X = sc.fit_transform(X)
    print(optimize(X=X,clustering_alg=KMeans,clustering_alg_params={}, metric=calinski_harabasz_score, metric_params={'X':X}, k_values=[2,3,4], goal='minimize'))


if __name__ == '__main__':
    main()
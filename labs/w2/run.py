import argparse
import logging
import os
import pickle
from datetime import datetime
from typing import List, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

from algorithms.kprototypes import KPrototypes
from algorithms.pca import PCA as IML_PCA
from algorithms.som import SOM
from preprocessing import adult, connect_4, segment
from utils.evaluate import evaluate_supervised, evaluate_unsupervised


def run_pca(paths: List[Dict[str, str]], min_explained_variance: float, params):
    X_transforms = {}  # Dict to store transformations from our PCA implementation, in order to apply K-Prototypes

    for path in paths:
        X_reconstructed = {}
        explained_variances = []

        X = pd.read_csv(os.path.join('datasets', path['X'])).values
        X_transforms[path['name']] = {}

        f = plt.figure()
        plt.scatter(X[:, 0], X[:, 1])
        plt.title(f'{path["name"].capitalize()} dataset over its two first features')
        plt.xlabel('Feature #0')
        plt.ylabel('Feature #1')
        plt.savefig(os.path.join(params.output_path, f'dataset_{path["name"]}.png'))
        plt.close(f)

        n = 1
        while True:
            # Our PCA
            iml_pca = IML_PCA(n_components=n, name=path['name'])
            X_transform_iml_pca = iml_pca.fit_transform(X)
            X_transforms[path['name']][n] = X_transform_iml_pca.copy()
            X_reconstructed[n] = iml_pca.inverse_transform(X_transform_iml_pca)

            cov_f, cov_matrix = iml_pca.get_cov_matrix(dataset_name=path['name'])
            plt.savefig(os.path.join(params.output_path, f'cov_matrix_{path["name"]}.png'))
            plt.close(cov_f)

            explained_variances.append(np.cumsum(iml_pca.explained_variance_ratio_)[-1])

            # PCA
            pca = PCA(n_components=n)
            X_transform_pca = pca.fit_transform(X)

            # Incremental PCA
            ipca = IncrementalPCA(n_components=n)
            X_transform_ipca = ipca.fit_transform(X)

            if n == 2:
                # Plot comparative with SKLearn methods
                f_2d, ax_2d = plt.subplots(1, 3, figsize=(10, 3), dpi=200)
                f_2d.tight_layout()

                ax_2d[0].set_title('Custom PCA')
                ax_2d[0].scatter(X_transform_iml_pca[:, 0], X_transform_iml_pca[:, 1], c='darkred', s=10, alpha=0.5)

                ax_2d[1].set_title('SKLearn PCA')
                ax_2d[1].scatter(X_transform_pca[:, 0], X_transform_pca[:, 1], c='darkblue', s=10, alpha=0.5)

                ax_2d[2].set_title('SKLearn Incremental PCA')
                ax_2d[2].scatter(X_transform_ipca[:, 0], X_transform_ipca[:, 1], c='teal', s=10, alpha=0.5)

                f_2d.savefig(os.path.join(params.output_path, f'pca_comparative_{path["name"]}.png'))
                plt.close(f_2d)

                # Plot comparative between different solvers
                f_sol, ax_sol = plt.subplots(1, 3, figsize=(10, 3), dpi=200)
                f_sol.tight_layout()

                ax_sol[0].set_title('Eigen')
                ax_sol[0].scatter(X_transform_iml_pca[:, 0], X_transform_iml_pca[:, 1], c='darkred', s=10, alpha=0.5)

                iml_pca = IML_PCA(n_components=n, name=path['name'], solver='hermitan')
                X_transform_iml_pca = iml_pca.fit_transform(X)
                ax_sol[1].set_title('Hermitan Eigen')
                ax_sol[1].scatter(X_transform_iml_pca[:, 0], X_transform_iml_pca[:, 1], c='darkorange', s=10, alpha=0.5)

                iml_pca = IML_PCA(n_components=n, name=path['name'], solver='svd')
                X_transform_iml_pca = iml_pca.fit_transform(X)
                ax_sol[2].set_title('SVD')
                ax_sol[2].scatter(X_transform_iml_pca[:, 0], X_transform_iml_pca[:, 1], c='magenta', s=10, alpha=0.5)

                f_sol.savefig(os.path.join(params.output_path, f'pca_solvers_{path["name"]}.png'))
                plt.close(f_sol)

            print(f'PCA with {n} components\n\tExplained Variance:'
                  f' {explained_variances[-1]:.6f} |'
                  f' {np.cumsum(pca.explained_variance_ratio_)[-1]:.6f} |'
                  f' {np.cumsum(ipca.explained_variance_ratio_)[-1]:.6f}')

            if explained_variances[-1] >= min_explained_variance:
                break
            else:
                n += 1

        explained_variances = np.array(explained_variances)
        n_to_plot = [np.where(explained_variances > 0.90)[0][0] + 1,
                     np.where(explained_variances > 0.95)[0][0] + 1,
                     np.where(explained_variances > 0.99)[0][0] + 1]

        # Plot evolution of expected variances for different N
        f_evo = plt.figure(figsize=(15, 5), dpi=200)
        plt.tight_layout()

        plt.axhline(90, ls='--', c='red')
        plt.axhline(95, ls='--', c='red')
        plt.axhline(99, ls='--', c='red')

        plt.axvline(n_to_plot[0], ls='--', c='red')
        plt.axvline(n_to_plot[1], ls='--', c='red')
        plt.axvline(n_to_plot[2], ls='--', c='red')

        x = list(range(1, n + 1))
        plt.plot(x, explained_variances * 100)
        plt.xticks(x)

        plt.title(f'Evolution of PCA Explained Variance for {path["name"]}', fontsize=18)
        plt.xlabel('# components', fontsize=16)
        plt.ylabel('% explained variance', fontsize=16)

        plt.savefig(os.path.join(params.output_path, f'pca_evolution_{path["name"]}.png'), bbox_inches='tight')
        plt.close(f_evo)

        # Plot reconstructed datasets with 2 components, explained variance of >90% and >99%.
        n_to_plot[0] = 2

        f_rec, ax_rec = plt.subplots(1, 3, figsize=(15, 5), dpi=200)
        plt.tight_layout()

        for idx, n in enumerate(n_to_plot):
            ax_rec[idx].scatter(X_reconstructed[n][:, 0], X_reconstructed[n][:, 1])
            ax_rec[idx].set_title(f'Reconstruction from {n} components', fontsize=18)
            ax_rec[idx].set_xlabel('Feature #0', fontsize=16)
            ax_rec[idx].set_ylabel('Feature #1', fontsize=16)

        plt.savefig(os.path.join(params.output_path, f'pca_reconstructed_{path["name"]}.png'), bbox_inches='tight')
        plt.close(f_rec)

    return X_transforms


def run_som(paths: List[Dict[str, str]], params):
    for path in paths:
        X = pd.read_csv(os.path.join('datasets', path['X'])).values
        Y = pd.read_csv(os.path.join('datasets', path['Y']))
        Y = LabelEncoder().fit_transform(Y)

        som = SOM(
            n_inputs=18,
            features_grid=(10, 10),
            # n_outputs=20,
            learning_radius=5,
            reduce_radius_after=50,
            step=0.5,
            std=1,
            shuffle_data=True,
            verbose=True
        )

        predictions = som.fit_predict(X, epochs=20)
        som_clusters = som.get_predicted_clusters(predictions)
        true_results = evaluate_supervised(Y, som_clusters)

        kp_clusters_path = os.path.join('datasets', 'predictions', f'prediction_{path["name"]}_K{path["k"]}.pkl')
        kp_clusters = pickle.load(open(kp_clusters_path, mode='rb'))
        method_results = evaluate_supervised(kp_clusters, som_clusters)

        # heatmap = som.plot_heatmap(X, Y)
        # plt.imshow(heatmap, cmap='Greys_r', interpolation='nearest')
        # plt.title(f'SOM Heatmap for {path["name"]} dataset')
        # plt.axis('off')
        # plt.colorbar()
        # plt.savefig(os.path.join(params.output_path, f'som_{path["name"]}.png'))


def get_cat_idx(array2d):
    cat_idx = []
    for j in range(array2d.shape[1]):
        if isinstance(array2d[j][0], str):
            cat_idx.append(j)
    return cat_idx


def eval_dict_to_table(res):
    table = '\n| Metric | Score |\n :---: | :---:'
    for metric, score in res.items():
        if metric != 'contingency_matrix':
            table += f'\n {metric} | {score:.6f}'

    if 'contingency_matrix' in res:
        table += '\nContingency Matrix\n'
        rows, cols = res['contingency_matrix'].shape[0], res['contingency_matrix'].shape[1]
        table += '\n|' + ' |' * cols + '\n ' + ':---: |' * cols
        for i in range(rows):
            row = ' | '.join(list(map(str, res['contingency_matrix'][i, :])))
            table += '\n' + row

    return table


def generate_results(X, labels_pred, labels_true, cat_idx, params):
    res_print = ''
    res_sup = evaluate_supervised(labels_pred=labels_pred, labels_true=labels_true)
    if len(cat_idx) > 0 and params.unsupervised_metrics:
        precomputed_distances = KPrototypes(K=1, cat_idx=cat_idx, name='precompute').compute_point_wise_distances(X)
    else:
        precomputed_distances = None
    res_print += '\nSupervised evaluation:\n' + eval_dict_to_table(res_sup) + '\n'
    if params.unsupervised_metrics:
        res_unsup = evaluate_unsupervised(X=X, labels=labels_pred, precomputed_distances=precomputed_distances)
        res_print += '\nUnsupervised evaluation:\n' + eval_dict_to_table(res_unsup) + '\n'
    return res_print


def run_kprototypes(paths: List[Dict[str, str]], params, transformed_data=None):
    res_to_save = '# Evaluation of K-Prototypes (with K = n_classes) clustering with the original data (potentially' \
                  'mixed attribute kinds) and the ones resulting from applying our PCA implementation\n\n'
    for path in paths:
        res_to_save += '## ' + path['name'] + '\n'
        X = pd.read_csv(os.path.join('datasets', path['X'])).values
        Y = pd.read_csv(os.path.join('datasets', path['Y']), header=None)
        n_classes = len(Y[Y.columns[0]].unique())
        cat_idx = get_cat_idx(X)
        predicted = KPrototypes(K=n_classes, name=f"{path['name']} original", fig_save_path=params.output_path,
                                cat_idx=cat_idx).fit_predict(X)
        res_to_save += '### With original data:\n'
        res_to_save += generate_results(X=X, labels_pred=predicted, labels_true=Y.values.flatten(),
                                        cat_idx=cat_idx, params=params) + '\n'
        if transformed_data is not None:
            for n_components in transformed_data[path['name']]:
                predicted = KPrototypes(K=n_classes, name=f"path['name'] {n_components} components",
                                        fig_save_path=params.output_path,
                                        cat_idx=get_cat_idx(transformed_data[path['name']][n_components])).fit_predict(
                    transformed_data[path['name']][n_components])
                res_to_save += f'### With PCA ({n_components} components):\n'
                res_to_save += generate_results(X=transformed_data[path['name']][n_components], labels_pred=predicted,
                                                labels_true=Y.values.flatten(), cat_idx=[], params=params) + '\n'
    with open(os.path.join(params.output_path, 'results.md'), 'a') as f:
        f.write(res_to_save)


def main(params):
    datasets = []
    if params.dataset == 'adult' or params.dataset is None:
        file_adult_num, file_adult_cat, file_adult_mix, file_adult_y = adult.preprocess()
        datasets += [
            {'name': 'adult', 'X': file_adult_num, 'Y': file_adult_y, 'type': 'num', 'k': 2},
            {'name': 'adult', 'X': file_adult_cat, 'Y': file_adult_y, 'type': 'cat', 'k': 2},
            {'name': 'adult', 'X': file_adult_mix, 'Y': file_adult_y, 'type': 'mix', 'k': 2},
        ]

    if params.dataset == 'connect-4' or params.dataset is None:
        file_connect_4_cat, file_connect_4_num, file_connect_4_y = connect_4.preprocess()
        datasets += [
            {'name': 'connect-4', 'X': file_connect_4_num, 'Y': file_connect_4_y, 'type': 'num', 'k': 3},
            {'name': 'connect-4', 'X': file_connect_4_cat, 'Y': file_connect_4_y, 'type': 'cat', 'k': 3},
            {'name': 'connect-4', 'X': file_connect_4_cat, 'Y': file_connect_4_y, 'type': 'mix', 'k': 3}
        ]

    if params.dataset == 'segment' or params.dataset is None:
        file_segment_num, file_segment_cat, file_segment_y = segment.preprocess()
        datasets += [
            {'name': 'segment', 'X': file_segment_num, 'Y': file_segment_y, 'type': 'num', 'k': 7},
            {'name': 'segment', 'X': file_segment_cat, 'Y': file_segment_y, 'type': 'cat', 'k': 7},
            {'name': 'segment', 'X': file_segment_num, 'Y': file_segment_y, 'type': 'mix', 'k': 7},
        ]

    num_paths = list(filter(lambda d: d['type'] == 'num', datasets))
    mix_paths = list(filter(lambda d: d['type'] == 'mix', datasets))

    if params.algorithm == 'PCA' or params.algorithm is None:
        X_transforms = run_pca(paths=num_paths, min_explained_variance=0.99, params=params)
        run_kprototypes(paths=mix_paths, params=params, transformed_data=X_transforms)
    if params.algorithm == 'SOM' or params.algorithm is None:
        run_som(paths=num_paths, params=params)
        if params.algorithm is not None:
            run_kprototypes(paths=mix_paths, params=params)


if __name__ == '__main__':
    print(os.listdir('.'))
    parser = argparse.ArgumentParser(description='Run all PCA + SOM project of MAI-IML')
    parser.add_argument('output_path', type=str, default='output', help='Output path for the logs')
    parser.add_argument('--seed', type=int, help='Seed for random behavior reproducibility')

    parser.add_argument('--algorithm', type=str, help='Select algorithm to run',
                        choices=['PCA', 'SOM'])
    parser.add_argument('--dataset', type=str, help='Select dataset to use',
                        choices=['adult', 'connect-4', 'segment'])

    parser.add_argument('--unsupervised_metrics', help='Whether to compute unsupervised metrics (it takes'
                                                       'a while in the case of the mixed dataset, Adult)',
                        action="store_true", default=False)

    args = parser.parse_args()

    print('+' + '-' * 45 + '+')
    print(f'| > Results will be stored in {args.output_path.upper()}')
    print(f'| > Seed: {args.seed}')
    print(f'| > Algorithms: {args.algorithm if args.algorithm is not None else "ALL"}')
    print(f'| > Datasets: {args.dataset if args.dataset is not None else "ALL"}')
    print('+' + '-' * 45 + '+')

    # Use timestamp as log file name
    current_time = datetime.now()
    log_name = str(current_time.date()) + '_' + (str(current_time.timetz())[:-7]).replace(':', '-')
    log_folder = os.path.join(args.output_path, log_name)

    os.makedirs(log_folder, mode=0o777)
    logging.basicConfig(filename=os.path.join(log_folder, 'log.txt'), level=logging.DEBUG)

    # Disable INFO and DEBUG logging for Matplotlib
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    args.output_path = log_folder
    main(params=args)

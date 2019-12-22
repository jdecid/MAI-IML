import json
import matplotlib.pyplot as plt

import os

import numpy as np


def visualize(dataset: str, var: str):
    assert var in ['k', 'r', 'vp', 'rp'], 'var parameters must be k, r, vp or rp'
    assert dataset in ['hypothyroid', 'pen-based'], 'dataset must be "hypothyroid" or "pen-based"'
    var_name = 'k'
    if var == 'r':
        var_name = 'r'
    elif var == 'vp':
        var_name = 'Voting policy'
    elif var == 'rp':
        var_name = 'Retention policy'
    with open(os.path.join('../output', f'{dataset}_results.json'), mode='r') as f:
        data = json.loads(f.read())
    dataset_name = 'Hypothyroid'
    if dataset == 'pen-based':
        dataset_name = 'Pen-Based'

    _, ax = plt.subplots(2, 1, sharex='col')

    for result in data:
        mean_accuracy = np.mean(list(map(lambda x: x['accuracy'], result['results'])))
        mean_time = np.mean(list(map(lambda x: x['time'], result['results'])))

        ax[0].scatter(result[var], mean_accuracy)
        ax[0].set_title('Performance vs ' + var_name + ' in ' + dataset_name + ' dataset')
        ax[0].set_ylabel('Accuracy')

        ax[1].scatter(result[var], mean_time)
        #ax[1].set_title('Execution Times')
        ax[1].set_xlabel(var_name)
        ax[1].set_ylabel('Execution time in seconds')

    plt.show()


visualize('pen-based', 'k')
visualize('pen-based', 'r')
visualize('pen-based', 'vp')
visualize('pen-based', 'rp')

visualize('hypothyroid', 'k')
visualize('hypothyroid', 'r')
visualize('hypothyroid', 'vp')
visualize('hypothyroid', 'rp')

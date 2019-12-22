import json
import matplotlib.pyplot as plt

import os

import numpy as np

def color_from_rp(rp):
    color = '#ffff00'
    if rp == 'NR':
        color = '#ff0000'
    elif rp == 'AR':
        color = '#00ff00'
    elif rp == 'DF':
        color = '#0000ff'
    elif rp == 'DD':
        color = '#000000'
    return color

def marker_from_vp(vp):
    marker = '*'
    if vp == 'MVS':
        marker = 'o'
    elif vp == 'MP':
        marker = 'x'
    return marker


def visualize(dataset: str):
    assert dataset in ['hypothyroid', 'pen-based'], 'dataset must be "hypothyroid" or "pen-based"'

    with open(os.path.join('../output', f'{dataset}_results.json'), mode='r') as f:
        data = json.loads(f.read())

    dataset_name = 'Hypothyroid'
    if dataset == 'pen-based':
        dataset_name = 'Pen-Based'

    _, ax = plt.subplots(2, 1, sharex='col')

    for result in data:
        mean_accuracy = np.mean(list(map(lambda x: x['accuracy'], result['results'])))
        mean_time = np.mean(list(map(lambda x: x['time'], result['results'])))
        kr = 'k=' + str(result['k']) + '\nr=' + str(result['r'])

        ax[0].scatter(kr, mean_accuracy, s=20, c=color_from_rp(result['rp']), marker=marker_from_vp(result['vp']))
        if dataset == 'pen-based': # Fix visualization of accuracy ylims for pen-based
            ax[0].set_ylim(0.99, .995)
        ax[0].set_title('Performance in ' + dataset_name + ' dataset')
        ax[0].set_ylabel('Accuracy')

        ax[1].scatter(kr, mean_time, s=20, c=color_from_rp(result['rp']), marker=marker_from_vp(result['vp']))
        #ax[1].set_title('Execution Times')
        #ax[1].set_xlabel('k and r')
        ax[1].set_ylabel('Execution time in seconds')

    plt.show()


def visualize_all(dataset: str):
    assert dataset in ['hypothyroid', 'pen-based'], 'dataset must be "hypothyroid" or "pen-based"'

    with open(os.path.join('../output', f'{dataset}_results.json'), mode='r') as f:
        data = json.loads(f.read())

    dataset_name = 'Hypothyroid'
    if dataset == 'pen-based':
        dataset_name = 'Pen-Based'

    _, ax = plt.subplots(2, 1, sharex='col')

    for result in data:
        accuracies = list(map(lambda x: x['accuracy'], result['results']))
        times = list(map(lambda x: x['time'], result['results']))
        kr = ['k=' + str(result['k']) + '   r=' + str(result['r']) + '   vp=' + str(result['vp'])
              + '   rp=' + str(result['rp']) for temp in result['results']]

        ax[0].scatter(kr, accuracies, s=10, c=color_from_rp(result['rp']), marker=marker_from_vp(result['vp']))
        if dataset == 'pen-based': # Fix visualization of accuracy ylims for pen-based
            ax[0].set_ylim(0.985,1)
        ax[0].set_title('Performance in ' + dataset_name + ' dataset')
        ax[0].set_ylabel('Accuracy')

        ax[1].scatter(kr, times, s=10, c=color_from_rp(result['rp']), marker=marker_from_vp(result['vp']))
        ax[1].set_ylim(0,30)
        #ax[1].set_title('Execution Times')
        #ax[1].set_xlabel('k and r')
        ax[1].set_ylabel('Execution time in seconds')
        ax[1].tick_params(axis='x', labelrotation=90, labelsize=7)

    plt.show()


visualize_all('pen-based')
visualize('pen-based')

visualize_all('hypothyroid')
visualize('hypothyroid')

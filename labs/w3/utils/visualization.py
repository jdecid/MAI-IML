import json
import matplotlib.pyplot as plt

import os

import numpy as np


def visualize(name: str):
    with open(os.path.join('../output', f'{name}_results.json'), mode='r') as f:
        data = json.loads(f.read())

    _, ax = plt.subplots(2, 1, sharex='col')

    for result in data:
        mean_accuracy = np.mean(list(map(lambda x: x['accuracy'], result['results'])))
        mean_time = np.mean(list(map(lambda x: x['time'], result['results'])))

        ax[0].scatter(result['k'], mean_accuracy)
        ax[0].set_title('Accuracies')

        ax[1].scatter(result['k'], mean_time)
        ax[1].set_title('Execution Times')

    plt.show()


visualize('pen-based')

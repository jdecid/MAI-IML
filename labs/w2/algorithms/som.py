from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from neupy.algorithms import SOFM

from utils.plotting import get_colors


class SOM(SOFM):
    """Self Organizing Map (SOM) wrapper.

    Parameters
    ----------

    """

    def fit_predict(self, X: np.ndarray, epochs=100):
        self.train(X, epochs=epochs)
        return self.predict(X)

    @staticmethod
    def get_predicted_clusters(C):
        # return np.where(C == 1)[0]
        return np.argmax(C, axis=1)

    def plot_heatmap(self, X: np.ndarray, Y: np.ndarray):
        heatmap = self.__compute_heatmap()
        clusters = self.get_predicted_clusters(self.predict(X))

        f = plt.figure()
        colors = get_colors(max(Y) + 1)

        for actual_class, cluster_index in zip(Y, clusters):
            cluster_x, cluster_y = divmod(cluster_index, self.features_grid[0])
            plt.plot(cluster_x, cluster_y, marker='o', markeredgecolor=colors[actual_class],
                     markersize=14, markeredgewidth=2, markerfacecolor='None')

        plt.imshow(heatmap, cmap='Greys_r', interpolation='nearest')
        plt.title(f'SOM Heatmap for connect-4 dataset')
        plt.axis('off')
        plt.colorbar()
        plt.show()

        return f

    def __compute_heatmap(self):
        weight = self.weight.reshape((self.n_inputs, self.features_grid[0], self.features_grid[1]))
        heatmap = np.zeros(shape=(self.features_grid[0], self.features_grid[1]))

        for (neuron_x, neuron_y), neighbours in self.__iter_neighbours(weight):
            total_distance = 0

            for (neighbour_x, neighbour_y) in neighbours:
                neuron_vec = weight[:, neuron_x, neuron_y]
                neighbour_vec = weight[:, neighbour_x, neighbour_y]

                distance = np.linalg.norm(neuron_vec - neighbour_vec)
                total_distance += distance

            avg_distance = total_distance / len(neighbours)
            heatmap[neuron_x, neuron_y] = avg_distance

        return heatmap

    def score(self, X, y):
        pass

    @staticmethod
    def __iter_neighbours(weights, hexagon=False):
        _, grid_height, grid_width = weights.shape

        hexagon_even_actions = ((-1, 0), (0, -1), (1, 0), (0, 1), (1, 1), (-1, 1))
        hexagon_odd_actions = ((-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (1, -1))
        rectangle_actions = ((-1, 0), (0, -1), (1, 0), (0, 1))

        for neuron_x, neuron_y in product(range(grid_height), range(grid_width)):
            neighbours = []

            if hexagon and neuron_x % 2 == 1:
                actions = hexagon_even_actions
            elif hexagon:
                actions = hexagon_odd_actions
            else:
                actions = rectangle_actions

            for shift_x, shift_y in actions:
                neighbour_x = neuron_x + shift_x
                neighbour_y = neuron_y + shift_y

                if 0 <= neighbour_x < grid_height and 0 <= neighbour_y < grid_width:
                    neighbours.append((neighbour_x, neighbour_y))

            yield (neuron_x, neuron_y), neighbours

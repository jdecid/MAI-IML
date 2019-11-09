from itertools import product

import numpy as np
from neupy.algorithms import SOFM


class SOM:
    def __init__(self):
        self.model: 'SOFM' = None

    def fit(self, X: np.ndarray, epochs=100, shuffle=False, verbose=False):
        self.model = SOFM(
            n_inputs=X.shape[1],
            features_grid=(20, 20),
            learning_radius=5,
            reduce_radius_after=50,
            step=0.5,
            std=1,
            shuffle_data=shuffle,
            verbose=verbose
        )

        self.model.train(X_train=X, epochs=epochs)

    def predict(self, X: np.ndarray):
        return self.model.predict(X)

    def fit_predict(self, X: np.ndarray, epochs=100, shuffle=False, verbose=False):
        self.fit(X, epochs=epochs, shuffle=shuffle, verbose=verbose)
        return self.predict(X)

    def compute_heatmap(self):
        weight = self.model.weight.reshape((self.model.n_inputs, 20, 20))
        heatmap = np.zeros(shape=(20, 20))

        for (neuron_x, neuron_y), neighbours in self.iter_neighbours(weight):
            total_distance = 0

            for (neighbour_x, neighbour_y) in neighbours:
                neuron_vec = weight[:, neuron_x, neuron_y]
                neighbour_vec = weight[:, neighbour_x, neighbour_y]

                distance = np.linalg.norm(neuron_vec - neighbour_vec)
                total_distance += distance

            avg_distance = total_distance / len(neighbours)
            heatmap[neuron_x, neuron_y] = avg_distance

        return heatmap

    @staticmethod
    def iter_neighbours(weights, hexagon=False):
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

    def _init_nodes(self):
        pass


if __name__ == '__main__':
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv('../datasets/segment_clean.csv')

    som = SOM()
    som.fit(df.values, epochs=400, verbose=True)
    hm = som.compute_heatmap()

    plt.imshow(hm, cmap='Greys_r', interpolation='nearest')
    plt.title('SOM Heatmap')
    plt.axis('off')
    plt.colorbar()
    plt.show()

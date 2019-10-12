from collections import Counter
from glob import glob

from scipy.io import arff


def read_dataset(verbose=False):
    dataset = {}

    files = sorted(glob('datasets/*.arff'))
    for idx, path in enumerate(files):
        name = path[9:-5]
        print(f'[{idx:2}/{len(files)}] Reading {name} dataset...')

        with open(path, mode='r') as f:
            data, meta = arff.loadarff(f)

            dataset[name] = {'data': data, 'meta': meta}

        if verbose:
            counter = Counter(meta.types()).items()
            for c, t in counter:
                print(f'\t- {c} {t}')

    return dataset


if __name__ == '__main__':
    read_dataset(verbose=True)

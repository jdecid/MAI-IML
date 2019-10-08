from glob import glob
from io import StringIO

from scipy.io import arff


def read_dataset():
    dataset = {}

    files = sorted(glob('datasets/*.arff'))
    for idx, path in enumerate(files):
        name = path[9:-5]
        print(f'[{idx:2}/{len(files)}] Reading {name} dataset...')

        with open(path, mode='r') as f:
            content = '\n'.join(f.readlines())

            f = StringIO(content)
            data, meta = arff.loadarff(f)

            dataset[name] = {'data': data, 'meta': meta}

    return dataset

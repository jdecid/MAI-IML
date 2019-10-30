from glob import glob

from scipy.io import arff


def read_dataset(name=None, verbose=False):
    """
    Read datasets in format arff inside ./datasets folder.
    :param name: Name of the dataset, if None, reads all datasets.
    :param verbose: Display attributes of the read dataset.
    :return: Dataset[name] or list of datasets.
    """
    datasets = {}

    if name is not None:
        files = [f'datasets/{name}.arff']
    else:
        files = sorted(glob('datasets/*.arff'))

    for idx, path in enumerate(files):
        name = path[9:-5]
        print(f'[{idx:2}/{len(files)}] Reading {name} dataset...')

        with open(path, mode='r') as f:
            data, meta = arff.loadarff(f)

            datasets[name] = {'data': data, 'meta': meta}

        if verbose:
            print(meta)

    if name is not None:
        return datasets[name]
    else:
        return datasets


def filter_datasets_by_attributes_type(dataset, tp):
    assert tp in ['numeric', 'nominal', 'mix']

    if tp is 'numeric' or tp is 'nominal':
        filtered = filter(lambda k: all(t == tp for t in dataset[k]['meta'].types()), dataset)
    else:
        filtered = filter(lambda k: (any(t == 'numeric') for t in dataset[k]['meta'].types()) and
                                    (any(t == 'nominal') for t in dataset[k]['meta'].types()), dataset)

    return list(map(lambda x: dataset[x]['meta'].name, filtered))
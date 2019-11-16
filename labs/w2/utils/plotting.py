import matplotlib.pyplot as plt


def get_colors(n: int):
    """
    Sample RGBA colors from HSV matplotlib colormap.
    :param n: Number of colors to obtain.
    :return: List of n RGBA colors.
    """
    return [plt.cm.hsv(x / n) for x in range(n)]


def mat_print(mat, fmt='g'):
    s = ''
    if len(mat.shape) == 2:
        col_maxes = [max([len(('{:' + fmt + '}').format(x)) for x in col]) for col in mat.T]
        for x in mat:
            for i, y in enumerate(x):
                s += ('{:' + str(col_maxes[i]) + fmt + '}').format(y) + '  '
            s += '\n'
    else:
        col_max = max([len(('{:' + fmt + '}').format(x)) for x in mat.T])
        for i, y in enumerate(mat):
            s += ('{:' + str(col_max) + fmt + '}').format(y) + '  '
        s += '\n'

    return s

import matplotlib.pyplot as plt


def get_colors(n: int):
    """
    Sample RGBA colors from HSV matplotlib colormap.
    :param n: Number of colors to obtain.
    :return: List of n RGBA colors.
    """
    return [plt.cm.hsv(x / n) for x in range(n)]

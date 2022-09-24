import numpy as np


def get_xy_plane(n, width):

    half_width = width / 2

    x = np.linspace(-half_width, half_width, n)
    y = np.arange(-half_width, half_width, n)

    xx, yy = np.meshgrid(x, y)

    return xx, yy


def get_polar_coordinates(xx, yy):

    r = np.sqrt(xx ** 2 + yy ** 2)

    _xx_div = np.copy(xx)
    _xx_div[_xx_div == 0] = 1
    atan_arg = yy / _xx_div
    # atan_arg[atan_arg == np.nan] = 0
    theta = np.arctan(atan_arg)

    return r, theta


if __name__ == '__main__':

    _w = 2
    _n = 51

    _xx, _yy = get_xy_plane(_n, _w)
    _r, _theta = get_polar_coordinates(_xx, _yy)

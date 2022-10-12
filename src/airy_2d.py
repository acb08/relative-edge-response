import numpy as np
import scipy.special

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


def airy_psf(r, d_ap, wavelength, f):

    """
    Calculated 2d incoherent 2d diffraction pattern of a circular aperture of diameter d and focal length f
    at a specific wavelength

    :param r: focal plane radial distance in polar coordinates (meters)
    :param d_ap: aperture diameter (meters)
    :param wavelength: wavelength (meters)
    :param f: focal length (meters)
    :return: 2d airy psf
    """

    psf = scipy.special.jv(1, np.pi * d_ap * r / (wavelength * f))
    psf = (d_ap / (4 * r)) * psf
    psf = psf ** 2

    return psf


if __name__ == '__main__':

    _w = 2
    _n = 51

    _xx, _yy = get_xy_plane(_n, _w)
    _r, _theta = get_polar_coordinates(_xx, _yy)

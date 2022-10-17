"""
Mostly a sandbox playing with diffraction limited PSFs
"""

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from src.fit import nearest_gaussian_psf


def get_xy_plane(n, p=1.0):

    if n % 2 == 0:
        raise ValueError('n must be odd')

    half_width = n // 2

    x = np.arange(-half_width, half_width + 1) * p
    y = np.arange(-half_width, half_width + 1) * p

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


def airy_psf(r, d_ap, wavelength, f, norm=True):

    """
    Calculated 2d incoherent 2d diffraction pattern of a circular aperture of diameter d and focal length f
    at a specific wavelength

    :param r: focal plane radial distance in polar coordinates (meters)
    :param d_ap: aperture diameter (meters)
    :param wavelength: wavelength (meters)
    :param f: focal length (meters)
    :param norm: bool, whether or not to normalize the kernel to a sum of 1
    :return: 2d airy psf
    """

    psf = scipy.special.jv(1, np.pi * d_ap * r / (wavelength * f))
    psf[np.where(r == 0)] = np.pi * d_ap / (wavelength * f)
    r[np.where(r == 0)] = 1
    psf = (d_ap / (4 * r)) * psf
    psf = psf ** 2

    if norm:
        psf = psf / np.sum(psf)

    return psf


def cross_section(kernel):

    size = np.shape(kernel)[0]
    idx_mid = size // 2
    cross_sec = kernel[idx_mid, :]

    return cross_sec


def checkout_psf(psf):

    print(np.sum(psf))

    sigma_hat, g_approx = nearest_gaussian_psf(psf)

    cross_sec = cross_section(psf)
    cross_sec_g_approx = cross_section(g_approx)

    fig, (ax0, ax1) = plt.subplots(1, 2)
    ax0.imshow(psf)
    ax1.plot(cross_sec, label='psf')
    ax1.plot(cross_sec_g_approx, label='gaussian approx')
    ax1.legend(loc='upper right')
    plt.show()


if __name__ == '__main__':

    _n = 15

    _d_ap = 1
    _f = 20
    _wavelength = 0.7 * 1e-6

    _Q = 1.5
    _f_num = _f / _d_ap
    _p = _wavelength * _f_num / _Q

    _xx, _yy = get_xy_plane(_n, p=_p)
    _r, _theta = get_polar_coordinates(_xx, _yy)

    _psf = airy_psf(_r, _d_ap, _wavelength, _f)

    checkout_psf(_psf)

    plt.figure()
    plt.imshow(_psf)
    plt.show()

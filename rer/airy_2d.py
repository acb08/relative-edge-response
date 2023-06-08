"""
Mostly a sandbox playing with diffraction limited PSFs
"""

import numpy as np
import scipy.special
import matplotlib.pyplot as plt
from rer.fit import nearest_gaussian_psf, gaussian2d


class AiryPSF:

    def __init__(self, f_number, wavelength, num, q):

        self.f_number = f_number
        self.wavelength = wavelength
        self.q = q

        self.numerical_aperture = 1 / (2 * f_number)

        self.pixel_pitch = self._get_pixel_pitch()
        self.num = num

        xx, yy = get_xy_plane(n=self.num, p=self.pixel_pitch)

        self.r, self.theta = get_polar_coordinates(xx=xx, yy=yy)

        psf = basic_airy(r=self.r,
                         wavelength=self.wavelength,
                         f_number=self.f_number)

        self._amp_norm_constant = np.sum(psf)
        self.psf = psf / self._amp_norm_constant

        self._normalized_zeros = (1.22, 2.33, 3.238)
        self._normalized_maxima = (0, 1.635, 2.679, 3.699)
        self._normalized_maxima_values = (1, 0.0175, 0.0042, 0.0016)

        self.spatial_scale = self.f_number * self.wavelength

        self.radial_zeros = np.array([self.spatial_scale * val for val in self._normalized_zeros])
        self.radial_zero_vals = np.zeros_like(self.radial_zeros)

        self.radial_maxima = np.array([self.spatial_scale * val for val in self._normalized_maxima])
        self.radial_maxima_vals = np.array([val / self._amp_norm_constant for val in self._normalized_maxima_values])

        # self.sigma_zhang = self._get_zhang_std()
        self.zhang_std, self.zhang_psf = self.get_zhang_approx()
        self.lsq_std, self.lsq_psf = self.get_least_squares_gaussian()

        self._psf_dict = {
            'psf': self.psf,
            'zhang': self.zhang_psf,
            'lsq': self.lsq_psf
        }

    def _get_pixel_pitch(self):
        return self.wavelength * self.f_number / self.q

    def cross_section(self, psf_id='psf'):
        psf = self._psf_dict[psf_id]
        return cross_section(psf)

    def radial_slice(self, psf_id='psf'):
        psf = self._psf_dict[psf_id]
        return radial_slice(psf)

    def cross_sectional_axis(self, normalize=True):
        cross_sec_axis = cross_section(self.r)
        if normalize:
            cross_sec_axis = cross_sec_axis / self.spatial_scale
        return cross_sec_axis

    def radial_axis(self, normalize=True):
        rad_axis = radial_slice(self.r)
        if normalize:
            rad_axis = rad_axis / self.spatial_scale
        return rad_axis

    def get_least_squares_gaussian(self):
        sigma_hat, g_approx = nearest_gaussian_psf(self.psf)
        sigma_hat = sigma_hat * self.pixel_pitch
        return sigma_hat, g_approx

    def _get_zhang_std(self):
        return 0.21 * self.wavelength / self.numerical_aperture

    def get_zhang_approx(self):
        zhang_std = self._get_zhang_std()
        zhang_psf = gaussian2d([zhang_std], self.r)
        zhang_psf = zhang_psf / np.sum(zhang_psf)
        return zhang_std, zhang_psf


def radial_compare(airy_psf: AiryPSF,
                   psf_ids=('psf', 'zhang', 'lsq')):

    x = airy_psf.radial_axis()
    plt.figure()
    for psf_id in psf_ids:
        y = airy_psf.radial_slice(psf_id=psf_id)
        plt.plot(x, y, label=psf_id)
    plt.legend()
    plt.xlabel(r'distance ($\frac{\lambda f}{d}$)')
    plt.ylabel('intensity')
    plt.show()


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


def basic_airy(r, wavelength, f_number):

    x = r / (wavelength * f_number)
    x[np.where(r == 0)] = 1
    psf_coherent = 2 * scipy.special.jv(1, np.pi * x) / (np.pi * x)
    psf_coherent[np.where(r == 0)] = 1
    psf = psf_coherent ** 2

    return psf


def cross_section(kernel):

    size = np.shape(kernel)[0]
    idx_mid = size // 2
    cross_sec = kernel[idx_mid, :]

    return cross_sec


def radial_slice(array, is_psf=False):

    size = np.shape(array)[0]
    idx_mid = size // 2
    cross_sec = array[idx_mid, idx_mid:]

    if is_psf:
        assert cross_sec[0] == np.max(array)

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

    _n = 11

    _d_ap = 1
    _f = 20
    _wavelength = 0.7 * 1e-6

    _Q = 2
    _f_num = _f / _d_ap
    _p = _wavelength * _f_num / _Q

    _airy = AiryPSF(f_number=_f_num,
                    wavelength=_wavelength,
                    num=_n,
                    q=_Q)

    radial_compare(_airy)



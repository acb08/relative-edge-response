from scipy.optimize import leastsq
import numpy as np


class Fitter(object):

    def __init__(self, x, y, fit_function, initial_params):

        self.x = x
        self.y = y
        self.fit_function = fit_function
        self.initial_params = initial_params

    def residuals(self, params):
        return np.ravel(self.y) - np.ravel(self.fit_function(params, self.x))

    def fit(self):
        return leastsq(self.residuals, self.initial_params)[0]


def line(params, x):
    m = params[0]
    b = params[1]
    return m * x + b


def gaussian_fourier_space_1d(params, x):
    sigma = params[0]
    return np.exp(- 2 * np.pi ** 2 * sigma ** 2 * x ** 2)


def gaussian2d(params, r):
    sigma = params[0]
    return (1 / (2 * np.pi * sigma ** 2)) * np.exp(-0.5 * (r / sigma) ** 2)


def _noisy_line(m, b, x, sigma=0.1):
    return m * x + b + np.random.randn(len(x)) * sigma


def _noisy_gaussian1d(params, x, noise_strength=0.1):
    return gaussian_fourier_space_1d(params, x) + noise_strength * np.random.randn(len(x))


def nearest_gaussian_psf(kernel, initial_params=(1, )):

    kernel_size = np.shape(kernel)[0]
    r = get_2d_radial(kernel_size)
    sigma_hat = Fitter(r, kernel, gaussian2d, initial_params).fit()

    return sigma_hat, gaussian2d([sigma_hat], r)


def _noisy_gauss_test_kernel(sigma, size, noise_param):
    r = get_2d_radial(size)
    kernel = gaussian2d([sigma], r)
    kernel = kernel + noise_param * np.random.randn(size, size)
    kernel = kernel
    return kernel


def get_2d_radial(n):
    half_width = n // 2
    if half_width * 2 + 1 != n:
        raise ValueError('n must be odd')
    x = np.arange(-half_width, half_width + 1)
    y = np.arange(-half_width, half_width + 1)
    xx, yy = np.meshgrid(x, y)
    r = np.sqrt(xx ** 2 + yy ** 2)
    return r


if __name__ == '__main__':

    # _x = np.linspace(-2, 2)
    # _sigma = 1
    # _initial_params = [0.5]
    # _data = _noisy_gaussian1d([_sigma], _x, noise_strength=0)
    # _best_fit = Fitter(_x, _data, gaussian_fourier_space_1d, _initial_params).fit()

    _size = 91
    _sigma = 3
    _test_kernel = _noisy_gauss_test_kernel(_sigma, _size, 0.001)
    _sigma_hat, _approx_kernel = nearest_gaussian_psf(_test_kernel)
    print(np.sum(_test_kernel))
    # _test_kernel = _test_kernel / np.sum(_test_kernel)
    print(np.sum(_approx_kernel))




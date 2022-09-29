from scipy.optimize import leastsq
import numpy as np


class Fitter(object):

    def __init__(self, x, y, fit_function, initial_params):

        self.x = x
        self.y = y
        self.fit_function = fit_function
        self.initial_params = initial_params

    def residuals(self, params):
        return np.ravel(self.y) - self.fit_function(params, self.x)

    def fit(self):
        return leastsq(self.residuals, self.initial_params)[0]


def line(params, x):
    m = params[0]
    b = params[1]
    return m * x + b


def gaussian_fourier_space(params, x):
    sigma = params[0]
    return np.exp(- 2 * np.pi ** 2 * sigma ** 2 * x ** 2)


def _noisy_line(m, b, x, sigma=0.1):
    return m * x + b + np.random.randn(len(x)) * sigma


def _noisy_gaussian(params, x, noise_strength=0.1):
    return gaussian_fourier_space(params, x) + noise_strength * np.random.randn(len(x))


if __name__ == '__main__':

    _x = np.linspace(-2, 2)

    _sigma = 1

    _initial_params = [0.5]

    _data = _noisy_gaussian([_sigma], _x, noise_strength=0)

    _best_fit = Fitter(_x, _data, gaussian_fourier_space, _initial_params).fit()

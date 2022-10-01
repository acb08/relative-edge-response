import numpy as np
import matplotlib.pyplot as plt
from src.fit import Fitter
import src.definitions as definitions
from pathlib import Path
import src.functions as functions


def gauss_xfer(params, xi):
    """
    Returns the Fourier transform of a Gaussian of standard deviation sigma. (Incidentally, the Fourier transform of a
    Gaussian is also a Gaussian with the inverse standard deviation and different normalization).

    :param params: array containing a single scalar value (the standard deviation of the pre Fourier transform
    Gaussian
    :param xi: independent variable (spatial frequency axis in this case)
    :return: Fourier transform of normalized Gaussian of standard deviation sigma
    """
    sigma = params[0]
    return np.exp(- 2 * np.pi ** 2 * sigma ** 2 * xi ** 2)


def rect_xfer(xi, p):
    """
    Returns the transfer function of a RECT function of width p, which is a SINC function of width parameter 1 / p
    """
    return np.sinc(p * xi)


def gauss_xfer_fit(xi, y, initial_params=(1, )):
    """
    Fits a non-standard (i.e. not normalized) Gaussian to the input data
    """
    sigma_hat = Fitter(xi, y, gauss_xfer, initial_params).fit()
    return sigma_hat, gauss_xfer([sigma_hat], xi)


def lorentz(params, x):
    b = params[0]
    m = params[1]
    return b / (np.pi * (x - m)**2 + b**2)


def lorentz_fit(x, y, initial_params=(1, 1)):
    fit_params = Fitter(x, y, lorentz, initial_params).fit()
    return fit_params, lorentz(fit_params, x)


def apply_lorentz_correction(sigma, cosmological_constant=1):
    """
    Nothing to do with time dilation or length contraction :(

    Using the correction as fit, the correction allows accurate modeling of RER down to blur standard deviations of
    roughly 0.25 pixels. If we decrease the correction by approximately 25% (i.e. set our cosmological constant to
    0.75), then the fit holds much lower, although
    its meaning is dubious since the combined transfer function is decided non-Gaussian by this point.
    """
    correction = lorentz(definitions.LORENTZ_TERMS, sigma)
    return sigma + cosmological_constant * correction


if __name__ == '__main__':

    output_dir = Path(definitions.ROOT_DIR, definitions.REL_PATHS['transfer_function'])
    if not output_dir.is_dir():
        Path.mkdir(output_dir, parents=True)

    # using a config dict so I can log (if I get around to  it...)
    config = {
        'sigma_vals': [0.1, 0.25, 0.5, .75, 1, 1.25, 1.5, 2, 2.5, 3, 5, 7, 10],
        'pixel_pitch': 1,
        'xi_min': -5,
        'xi_max': 5,
        'plot_pix_xfer': True
    }

    _sigma_vals = config['sigma_vals']
    _sigma_vals = np.asarray(_sigma_vals)
    _p = config['pixel_pitch']
    _xi_min = config['xi_min']
    _xi_max = config['xi_max']
    _plot_pix_transfer = config['plot_pix_xfer']

    _xi = np.linspace(_xi_min, _xi_max, num=800)

    _pixel_xfer = rect_xfer(_xi, _p)
    _sigma_fit_vals = []

    for i, _sigma in enumerate(_sigma_vals):

        plot_save_name = f'transfer_functions_{i}.png'

        _gaussian_transfer_function = gauss_xfer([_sigma], _xi)
        _combined_xfer = _pixel_xfer * _gaussian_transfer_function
        _sigma_fit, _best_fit_curve = gauss_xfer_fit(_xi, _combined_xfer)
        _sigma_fit = _sigma_fit[0]
        _sigma_fit_vals.append(_sigma_fit)

        plt.figure()
        if _plot_pix_transfer:
            plt.plot(_xi, _pixel_xfer, ls='--', label='pixel')
        plt.plot(_xi, _gaussian_transfer_function, ls=':', label=rf'optical ($\sigma=$ {_sigma})')
        plt.plot(_xi, _combined_xfer, label=rf'combined')
        plt.plot(_xi, _best_fit_curve, label=rf'fit, $\sigma_f =$ {round(_sigma_fit,3)}')
        plt.xlabel(r'$\xi$')
        plt.ylabel(r'$H \left ( \xi \right)$')
        plt.legend(loc='upper right')
        plt.savefig(Path(output_dir, plot_save_name))
        plt.show()

    sigma_residuals = np.asarray(_sigma_fit_vals) - _sigma_vals

    _lorentz_params, _lorentz_fit = lorentz_fit(_sigma_vals, sigma_residuals)
    _b, _m = _lorentz_params
    _b_legend = round(_b, 2)
    _m_legend = round(_m, 2)

    residual_save_name = 'residual_fit.png'
    plt.figure()
    plt.scatter(_sigma_vals, sigma_residuals, color='k')
    plt.plot(_sigma_vals, _lorentz_fit, label=rf'Lorentzian fit ($b =$ {_b_legend}, $m =$ {_m_legend})',
             color='k', ls='--')
    plt.xlabel(r'$\sigma_{opt}$')
    plt.ylabel(r'$\sigma_{fit} - \sigma_{opt}$')
    plt.legend()
    plt.savefig(Path(output_dir, residual_save_name))
    plt.show()

    functions.log_config(output_dir=output_dir, config=config)
    lorentz_params_log = {
        'b': float(_b),
        'm': float(_m)
    }
    functions.log_config(output_dir=output_dir, config=lorentz_params_log, config_used_filename='lorentz_params.yml')

"""
Generates blur kernel sets using ParametricSysModel class in sys_psf.py
"""

import numpy as np
from rer.sys_psf import ParametricSysModel
from rer.definitions import ROOT_DIR, REL_PATHS, STANDARD_BLUR_KERNEL_FILENAME_STEM
from pathlib import Path
import rer.functions as functions
import argparse

DEFAULT_RNG = np.random.default_rng()


def create_parametric_system_model(num_samples, f_num, pix_pitch, ff, wl):
    parametric_system_model = ParametricSysModel(num_samples, f_num, pix_pitch, ff, wl)
    parametric_system_model.apply_scaling()
    parametric_system_model.mtf_aperture()

    return ParametricSysModel(num_samples, f_num, pix_pitch, ff, wl)


def create_diffraction_limited_kernels():
    pass


def generate_mtf_psf(parametric_system_model, kernel_half_width, p_smear, sigma_jitter, wl_rms, l_corr=None):
    parametric_system_model.mtf_system(p_smear, sigma_jitter, wl_rms, l_corr)
    parametric_system_model.interp_prf(size=kernel_half_width)
    parametric_system_model.create_uprf()


def choose_val(value_range):
    low, high = value_range
    return DEFAULT_RNG.uniform(low=low, high=high)


def make_blur_kernel_set(config):

    output_dir_parent = Path(ROOT_DIR, REL_PATHS['blur_kernels'])
    if not output_dir_parent.is_dir():
        Path.mkdir(output_dir_parent)

    key, __, __ = functions.key_from_dir(output_dir_parent)
    output_dir = Path(output_dir_parent, key)
    Path.mkdir(output_dir)

    diffraction_only = False
    if 'diffraction_only' in config.keys():
        diffraction_only = config['diffraction_only']

    if diffraction_only:
        kernel_half_width = config['kernel_half_width']
        f_num = config['f_num']
        pix_pitch = config['pix_pitch']
        raise NotImplementedError

    else:
        # parametric system model constants
        num_samples = config['num_samples']
        f_num = config['f_num']
        pix_pitch = config['pix_pitch']
        ff = config['ff']
        wl = config['wl']

        parametric_system_models = []

        if type(wl) in {list, tuple}:
            if 'wl_weighting' in config.keys():
                wl_weighting = config['wl_weighting']
            else:
                wl_weighting = np.ones(len(wl))

            for i, wavelength in enumerate(wl):
                parametric_system_model = create_parametric_system_model(num_samples, f_num, pix_pitch, ff, wavelength)
                parametric_system_models.append(parametric_system_model)
        else:
            parametric_system_model = create_parametric_system_model(num_samples, f_num, pix_pitch, ff, wl)
            parametric_system_models.append(parametric_system_model)
            wl_weighting = [1]

        kernel_half_width = config['kernel_half_width']
        num_kernels = config['num_kernels']

        # parametric system model variables
        p_smear_range = config['p_smear_range']
        sigma_jitter_range = config['sigma_jitter_range']
        wl_rms_range = config['wl_rms_range']
        l_corr_range = config['l_corr_range']

        for i in range(num_kernels):

            p_smear = choose_val(p_smear_range)
            sigma_jitter = choose_val(sigma_jitter_range)
            wl_rms = choose_val(wl_rms_range)
            l_corr = choose_val(l_corr_range)

            psf = None

            for j, parametric_system_model in enumerate(parametric_system_models):

                generate_mtf_psf(parametric_system_model, kernel_half_width, p_smear, sigma_jitter, wl_rms, l_corr=l_corr)
                uprf = parametric_system_model.uprf
                if j == 0:
                    psf = uprf * wl_weighting[j]
                else:
                    psf += uprf * wl_weighting[j]

            psf = np.mean(psf, axis=0)
            psf = psf / np.sum(psf)

            if np.abs(np.sum(psf) - 1) > 0.1:
                raise ValueError('uprf must be normalized')

            filename = f'{STANDARD_BLUR_KERNEL_FILENAME_STEM}_{i}.npz'
            np.savez(Path(output_dir, filename),
                     kernel=psf,
                     p_smear=p_smear,
                     sigma_jitter=sigma_jitter,
                     wl_rms=wl_rms,
                     l_corr=l_corr)

    functions.log_config(output_dir, config)
    print(key)


if __name__ == '__main__':

    config_filename = 'kernel_config_pan.yml'
    config_numbers = [3, 4, 5, 6, 7, 8]

    for num in config_numbers:
        config_filename = f'kernel_config_{num}.yml'

        parser = argparse.ArgumentParser()
        parser.add_argument('--config_name', default=config_filename, help='config filename to be used')
        parser.add_argument('--config_dir',
                            default=Path(ROOT_DIR, 'rer', 'blur_kernel_configs'),
                            help="configuration file directory")
        args_passed = parser.parse_args()
        run_config = functions.get_config(args_passed)

        make_blur_kernel_set(run_config)

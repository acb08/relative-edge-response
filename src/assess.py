import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.measure import load_dataset
import json
import src._fit as fit
from scipy.special import erf
from transfer_func import apply_lorentz_correction


MARKERS = [
    ".",
    "v",
    "^",
    "<",
    '>',
    "1",
    "2",
    "3",
    "4",
    "s",
    "p",
    "P",
    "*",
    "h",
    "+",
    "x",
    "X",
    "D",
    "d",
]


class BlurredEdgeProperties(object):

    def __init__(self, dataset, mtf_lsf, parse_keys=('scaled_blur', 'rer')):  # parse_keys=('std', 'rer')
        self.dataset = dataset
        self.mtf_lsf = mtf_lsf
        self.edge_chip_data = self.dataset['chips']
        self.raw_vector_data = self.vectorize_props()

        self.rer = np.asarray(self.raw_vector_data['rer'])
        self.native_blur = np.asarray(self.raw_vector_data['native_blur'])
        self.scaled_blur = np.asarray(self.raw_vector_data['scaled_blur'])
        self.down_sample_factor = np.asarray(self.raw_vector_data['down_sample_factor'])
        if 'optical' in self.raw_vector_data.keys():
            self.optical_bool_vec = np.asarray(self.raw_vector_data['optical'])
        else:
            self.optical_bool_vec = None
        # self.native_blur, self.scaled_blur, self.down_sample_vector = self._screen_extract_blur_params()
        self.down_sample_values = np.unique(self.down_sample_factor)
        self.parse_keys = parse_keys
        self.scale_sorted_vector_data = self.parse_vector_data()

    def vectorize_props(self):

        vector_data = {}
        for i, (chip_id, chip_data) in enumerate(self.edge_chip_data.items()):

            measured_props = self.mtf_lsf[chip_id]

            if i == 0:
                for key, val in chip_data.items():
                    vector_data[key] = [val]
                for key, val in measured_props.items():
                    vector_data[key] = [val]
            else:
                for key, val in chip_data.items():
                    vector_data[key].append(val)
                for key, val in measured_props.items():
                    vector_data[key].append(val)

        return vector_data

    def parse_vector_data(self):

        scale_sorted_vector_data = {}

        for i, key in enumerate(self.parse_keys):

            for __, down_sample_val in enumerate(self.down_sample_values):

                target_indices = np.where(self.down_sample_factor == down_sample_val)
                target_vector = np.asarray(self.raw_vector_data[key])
                target_vector = target_vector[target_indices]

                if i == 0:
                    scale_sorted_vector_data[down_sample_val] = {key: target_vector}
                else:
                    scale_sorted_vector_data[down_sample_val][key] = target_vector

        return scale_sorted_vector_data

    def compare_optical_gauss_equiv(self):

        if self.optical_bool_vec is None:
            return RuntimeError('Optical/Gaussian approximation status not labeled')

        scaled_blur_values = np.unique(self.scaled_blur)
        optical_rer = []
        gaussian_equiv_rer = []

        for scaled_blur_val in scaled_blur_values:
            optical_indices = tuple([self.optical_bool_vec & (self.scaled_blur == scaled_blur_val)])
            gauss_equiv_indices = tuple([(np.invert(self.optical_bool_vec)) & (self.scaled_blur == scaled_blur_val)])
            optical_rer.append(np.mean(self.rer[optical_indices]))
            gaussian_equiv_rer.append(np.mean(self.rer[gauss_equiv_indices]))

        return optical_rer, gaussian_equiv_rer

    # def rer_filtered(self, optical=True):
    #     if optical:
    #         return self.rer[tuple([self.optical_bool_vec])]
    #     else:
    #         return self.rer[tuple([np.invert(self.optical_bool_vec)])]

    def attribute_filtered(self, attribute, opt_or_gauss):

        acceptable_attributes = {'rer', 'native_blur', 'scaled_blur', 'down_sample_factor'}
        if attribute not in acceptable_attributes:
            raise ValueError(f'only attributes {acceptable_attributes} may be used in attribute_filtered method')

        attribute = getattr(self, attribute)
        if opt_or_gauss == 'optical':
            indices = tuple([self.optical_bool_vec])
        elif opt_or_gauss == 'gaussian':
            indices = tuple([np.invert(self.optical_bool_vec)])
        else:
            raise ValueError('opt_or_gauss must be specified as either optical or gaussian')

        return attribute[indices]


def load_measured_mtf_lsf(directory):
    with open(Path(directory, 'mtf_lsf.json'), 'r') as file:
        data = json.load(file)
    return data


def get_edge_blur_properties(dataset, mtf_lsf_data):
    return BlurredEdgeProperties(dataset, mtf_lsf_data)


def rer_multi_plot(edge_properties, start_idx=0, directory=None):

    plt.figure()
    for blur_val in edge_properties.native_blur_values:
        std, rer = edge_properties.fixed_native_blur_vector_pair(blur_val, key_0='std', key_1='rer')
        plt.plot(std[start_idx:], rer[start_idx:], label=f'native blur: {blur_val} (pixels)')
    plt.xlabel(r'$\sigma_{blur}$ secondary (pixels)')
    plt.ylabel('RER')
    plt.legend()
    if directory:
        save_name = f'rer_multi_plot_st-idx-{start_idx}.png'
        plt.savefig(Path(directory, save_name))
    else:
        plt.show()


# def ideal_rer(sigma_blur):
#     return 1 / (sigma_blur * np.sqrt(2 * np.pi))

def discrete_sampling_rer_model(sigma_blur, apply_blur_correction=True):
    if apply_blur_correction:
        corrected_blur = apply_lorentz_correction(sigma_blur)
        return erf(1 / (2 * np.sqrt(2) * corrected_blur))
    else:
        return erf(1 / (2 * np.sqrt(2) * sigma_blur))


def rer_ideal_slope(sigma_blur, apply_blur_correction=True):
    if apply_blur_correction:
        corrected_blur = apply_lorentz_correction(sigma_blur)
        return 1 / (corrected_blur * np.sqrt(2 * np.pi))
    else:
        return 1 / (sigma_blur * np.sqrt(2 * np.pi))


def rer_fit_multi_plot(edge_properties, start_idx=0, fit_key='rer_0', directory=None):

    if directory:
        sub_dir = Path(directory, f'{fit_key}-st-idx-{start_idx}')
        if not sub_dir.is_dir():
            Path.mkdir(sub_dir)
        log_file = open(Path(sub_dir, 'fit_log.txt'), 'w')
    else:
        log_file = None

    for blur_val in edge_properties.native_blur_values:

        std, rer = edge_properties.fixed_native_blur_vector_pair(blur_val, key_0='std', key_1='rer')
        std = std[start_idx:]
        rer = rer[start_idx:]
        fit_coefficients, rer_fit, correlation = fit_predict_blur_rer(edge_properties, blur_val, fit_key,
                                                                      start_idx=start_idx)
        plt.figure()
        plt.scatter(std, rer, label=f'native blur: {blur_val} (pixels)')
        plt.plot(std, rer_fit, label=f'fit {fit_key}')
        plt.xlabel(r'$\sigma_{blur}$ secondary (pixels)')
        plt.ylabel('RER')
        plt.legend()
        if directory:
            blur_val_str = str(blur_val).replace('.', '-')
            save_name = f'rer_fit_{blur_val_str}.png'
            plt.savefig(Path(sub_dir, save_name))
            plt.close()
        else:
            plt.show()

        print(f'native blur {blur_val} {fit_key} fit coefficients: ', file=log_file)
        print(f'{fit_coefficients}: ', file=log_file)
        print(f'correlation: {correlation} \n', file=log_file)

    if log_file:
        log_file.close()


def fit_predict_blur_rer(edge_properties, native_blur, fit_key, start_idx=0):

    distortion_array, rer_vector = edge_properties.get_distortion_performance_arrays(native_blur,
                                                                                     start_idx=start_idx)
    fit_coefficients = fit.fit(distortion_array, rer_vector, fit_key=fit_key)
    fit_prediction = fit.apply_fit(fit_coefficients, distortion_array, fit_key=fit_key, add_bias=None)
    correlation = fit.evaluate_fit(fit_coefficients, distortion_array, rer_vector, fit_key=fit_key)

    return fit_coefficients, fit_prediction, correlation


def plot_measured_predicted_rer(edge_props,
                                scaled_blur_lower_bound=None, scaled_blur_upper_bound=None,
                                native_blur_lower_bound=None, native_blur_upper_bound=None,
                                output_dir=None,
                                plot_model_unadjusted=False, plot_ideal_adjusted=False, plot_ideal_unadjusted=False,
                                distinguish_blur_type=True):

    if not scaled_blur_lower_bound:
        scaled_blur_lower_bound = -np.inf
    if not scaled_blur_upper_bound:
        scaled_blur_upper_bound = np.inf

    if not native_blur_lower_bound:
        native_blur_lower_bound = -np.inf
    if not native_blur_upper_bound:
        native_blur_upper_bound = np.inf

    scaled_blur = edge_props.scaled_blur
    native_blur = edge_props.native_blur
    rer = edge_props.rer

    bounded_indices = tuple([(scaled_blur >= scaled_blur_lower_bound) & (scaled_blur <= scaled_blur_upper_bound)
                             & (native_blur >= native_blur_lower_bound) & (native_blur <= native_blur_upper_bound)])

    scaled_blur = scaled_blur[bounded_indices]
    rer = rer[bounded_indices]

    blur_plot = np.linspace(np.min(scaled_blur), np.max(scaled_blur), num=128)
    rer_predicted = discrete_sampling_rer_model(blur_plot, apply_blur_correction=True)
    rer_predicted_unadjusted_blur = discrete_sampling_rer_model(blur_plot, apply_blur_correction=False)
    rer_ideal_adjusted_blur = rer_ideal_slope(blur_plot, apply_blur_correction=True)
    rer_ideal_unadjusted_blur = rer_ideal_slope(blur_plot, apply_blur_correction=False)

    # keep_indices = np.where(blur_lower_bound <= scaled_blur <= blur_upper_bound)
    # rer = rer[keep_indices]
    # scaled_blur = scaled_blur[keep_indices]

    name_seed = 'rer_v_blur'
    if scaled_blur_lower_bound:
        name_seed = f'{name_seed}_{str(round(scaled_blur_lower_bound, 2)).replace(".", "p")}'
    if scaled_blur_upper_bound:
        name_seed = f'{name_seed}_{str(round(scaled_blur_upper_bound, 2)).replace(".", "p")}'

    plt.figure()

    if distinguish_blur_type:

        scaled_blur_opt = edge_props.attribute_filtered('scaled_blur', 'optical')
        rer_opt = edge_props.attribute_filtered('rer', 'optical')

        scaled_blur_gauss = edge_props.attribute_filtered('scaled_blur', 'gaussian')
        rer_gauss = edge_props.attribute_filtered('rer', 'gaussian')

        plt.scatter(scaled_blur_opt, rer_opt, label='measured (optical)', marker=MARKERS[0])
        plt.scatter(scaled_blur_gauss, rer_gauss, label='measured (gaussian approx)', marker=MARKERS[1])

        # down_sample_factor = edge_props.down_sample_factor
        # for i, down_sample_val in enumerate(np.unique(down_sample_factor)):
        #     indices = tuple([down_sample_factor == down_sample_val])
        #     num_markers = len(MARKERS)
        #     offset, index = divmod(i, num_markers)
        #     marker = MARKERS[index]
        #     plt.scatter(scaled_blur[indices], rer[indices], label=f'measured, downsample={down_sample_val}',
        #                 marker=marker)

    else:
        plt.scatter(scaled_blur, rer, label='measure', marker='+')

    plt.plot(blur_plot, rer_predicted, label='modeled')

    if plot_model_unadjusted:
        name_seed = f'{name_seed}_mu'
        plt.plot(blur_plot, rer_predicted_unadjusted_blur, label='modeled, uncorrected')
    if plot_ideal_adjusted:
        name_seed = f'{name_seed}_ia'
        plt.plot(blur_plot, rer_ideal_adjusted_blur, label='ideal slope model, corrected')
    if plot_ideal_unadjusted:
        name_seed = f'{name_seed}_iu'
        plt.plot(blur_plot, rer_ideal_unadjusted_blur, label='ideal slope model, uncorrected')

    plt.legend()
    plt.xlabel(r'$\sigma_{blur}$ (pixels)')
    plt.ylabel('RER')
    if output_dir:
        plt.savefig(Path(output_dir, f'{name_seed}.png'))
    plt.show()


def optical_gauss_compare(edge_props):

    try:
        optical, gaussian = edge_props.compare_optical_gauss_equiv()
        x_min = np.min(optical)
        x_max = np.max(optical)
        x = np.linspace(x_min, x_max)
        plt.figure()
        plt.scatter(optical, gaussian)
        plt.plot(x, x)
        plt.show()

    except RuntimeError:
        print('edge_props object does not distinguish optical and Gaussian blur kernels')


if __name__ == '__main__':

    _directory_key = '0019'
    _directory, _dataset = load_dataset(_directory_key)
    _mtf_lsf_data = load_measured_mtf_lsf(_directory)
    _output_dir = Path(_directory, 'rer_assessment')
    if not _output_dir.is_dir():
        Path.mkdir(_output_dir)

    _edge_props = get_edge_blur_properties(_dataset, _mtf_lsf_data)

    plot_measured_predicted_rer(_edge_props, output_dir=_output_dir,
                                plot_model_unadjusted=False, plot_ideal_adjusted=False, plot_ideal_unadjusted=False)
    plot_measured_predicted_rer(_edge_props, output_dir=_output_dir,
                                plot_model_unadjusted=False, plot_ideal_adjusted=False, plot_ideal_unadjusted=False,
                                native_blur_lower_bound=0.51)

    optical_gauss_compare(_edge_props)


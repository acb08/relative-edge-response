import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from src.measure import load_dataset
import json
import src._fit as fit
from scipy.special import erf
from transfer_func import apply_lorentz_correction


class BlurredEdgeProperties(object):

    def __init__(self, dataset, mtf_lsf, parse_keys=('scaled_blur', 'rer')):  # parse_keys=('std', 'rer')
        self.dataset = dataset
        self.mtf_lsf = mtf_lsf
        self.edge_chip_data = self.dataset['chips']
        self.raw_vector_data = self.vectorize_props()

        self.rer = np.asarray(self.raw_vector_data['rer'])
        self.native_blur = np.asarray(self.raw_vector_data['native_blur'])
        self.scaled_blur = np.asarray(self.raw_vector_data['scaled_blur'])
        self.down_sample_vector = np.asarray(self.raw_vector_data['down_sample_factor'])

        # self.native_blur, self.scaled_blur, self.down_sample_vector = self._screen_extract_blur_params()
        self.down_sample_values = np.unique(self.down_sample_vector)
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

            for __, down_sample_factor in enumerate(self.down_sample_values):

                target_indices = np.where(self.down_sample_vector == down_sample_factor)
                target_vector = np.asarray(self.raw_vector_data[key])
                target_vector = target_vector[target_indices]

                if i == 0:
                    scale_sorted_vector_data[down_sample_factor] = {key: target_vector}
                else:
                    scale_sorted_vector_data[down_sample_factor][key] = target_vector

        return scale_sorted_vector_data

    # def fixed_native_blur_vector_pair(self, native_blur_value, key_0='std', key_1='rer'):
    #     return self.scale_sorted_vector_data[native_blur_value][key_0], self.scale_sorted_vector_data[native_blur_value][key_1]
    #
    # def get_distortion_performance_arrays(self, native_blur, start_idx=0):
    #
    #     blur, rer = self.fixed_native_blur_vector_pair(native_blur, key_0='std', key_1='rer')
    #     native_blur_vector = native_blur * np.ones_like(blur)
    #
    #     distortion_array = np.zeros((len(native_blur_vector), 2))
    #     distortion_array[:, 0] = native_blur_vector
    #     distortion_array[:, 1] = blur
    #
    #     distortion_array = distortion_array[start_idx:, :]
    #     rer = rer[start_idx:]
    #
    #     return distortion_array, rer


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

def ideal_rer(sigma_blur):
    # (erf(1 / (2 * np.sqrt(2) * sigma_blur)) - erf(- 1 / (2 * np.sqrt(2) * sigma_blur))
    return erf(1 / (2 * np.sqrt(2) * sigma_blur))


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


def rer_predict_measured_plot(rer_measured, rer_predicted, scaled_blur, blur_lower_bound=None, blur_upper_bound=None):

    if not blur_lower_bound:
        blur_lower_bound = -np.inf
    if not blur_upper_bound:
        blur_upper_bound = np.inf

    keep_indices = np.where(blur_lower_bound <= scaled_blur <= blur_upper_bound)
    rer_measured_plot = rer_measured[keep_indices]
    rer_predicted_plot = rer_predicted[keep_indices]
    scaled_blur_plot = scaled_blur[keep_indices]

    plt.figure()
    plt.scatter(scaled_blur_plot, rer_measured_plot)
    plt.plot(scaled_blur_plot, rer_predicted_plot)
    plt.show()


if __name__ == '__main__':

    _directory_key = '0003'
    _directory, _dataset = load_dataset(_directory_key)
    _mtf_lsf_data = load_measured_mtf_lsf(_directory)

    _edge_props = get_edge_blur_properties(_dataset, _mtf_lsf_data)

    _scaled_blur = _edge_props.scaled_blur
    _corrected_blur = apply_lorentz_correction(_scaled_blur)
    _rer = _edge_props.rer

    _rer_ideal = ideal_rer(np.unique(_corrected_blur))

    plt.figure()
    plt.scatter(_scaled_blur, _rer)
    plt.plot(np.unique(_scaled_blur), _rer_ideal)
    plt.show()

    _scaled_blur_plot = np.unique(_scaled_blur)

    plt.figure()
    plt.scatter(_scaled_blur[np.where(_scaled_blur < 1)], _rer[np.where(_scaled_blur < 1)])
    plt.plot(_scaled_blur_plot[np.where(_scaled_blur_plot < 1)], _rer_ideal[np.where(_scaled_blur_plot < 1)])
    plt.show()
    # rer_multi_plot(_edge_props, directory=_directory)

    # _start_idx = 4
    #
    # # rer_fit_multi_plot(_edge_props, start_idx=_start_idx, fit_key='rer_0', directory=_directory)
    # # rer_fit_multi_plot(_edge_props, start_idx=_start_idx, fit_key='rer_1', directory=_directory)
    # # rer_fit_multi_plot(_edge_props, start_idx=_start_idx, fit_key='rer_2', directory=_directory)
    # # rer_fit_multi_plot(_edge_props, start_idx=_start_idx, fit_key='rer_3', directory=_directory)
    # rer_fit_multi_plot(_edge_props, start_idx=_start_idx, fit_key='rer_4', directory=_directory)

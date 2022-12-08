"""
Uses the slanted edge method to measure RER (as well as MTF) of an image chip containing a single near-vertical edge
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import scipy.stats as stats
from pathlib import Path
from rer.definitions import ROOT_DIR, REL_PATHS, STANDARD_DATASET_FILENAME
import json


def plot_rows(chip, interval=1):

    n_rows = np.shape(chip)[0]
    plt.figure()
    for i in range(n_rows):
        row = chip[i, :]
        if i % interval == 0:
            plt.plot(row)
    plt.show()


def pad_signal(x, num):

    x_new = np.zeros(len(x) + 2 * num)
    x_new[:num] = x[0]
    x_new[num:-num] = x
    x_new[-num:] = x[-1]

    return x_new


def get_lsf(esf, kernel=np.asarray([0.5, 0, -0.5]), pad=False):

    if pad:
        num_pad = int(np.floor(len(kernel) / 2))
        esf = pad_signal(esf, num_pad)

    lsf = np.convolve(esf, kernel, mode='valid')
    lsf = lsf * np.hamming(len(lsf))
    lsf = lsf / np.sum(lsf)

    return lsf


def estimate_edge(row, kernel=np.asarray([-0.5, 0.5])):

    lsf = get_lsf(row, kernel=kernel)
    indices = np.arange(len(lsf))
    centroid = (np.sum(indices * lsf))/np.sum(lsf)

    return centroid


def fit_edge(rows, plot=True):

    n, m = np.shape(rows)
    centroids = np.zeros(n)
    row_indices = np.arange(n)
    for idx in row_indices:
        row = rows[idx, :]
        centroid = estimate_edge(row)

        centroids[idx] = centroid

    fit = stats.linregress(row_indices, centroids)
    slope, offset = fit[0], fit[1]

    if plot:

        best_fit_line = slope * row_indices + offset
        plt.figure()
        plt.scatter(row_indices, centroids, marker='+', s=30, color='r')
        plt.plot(row_indices, best_fit_line, color='b')
        plt.xlabel('row index')
        plt.ylabel('edge location')
        plt.show()

    return centroids, fit


def oversampled_esf(rows, fit, oversample_factor=4, plot=False):

    n, m = np.shape(rows)
    x_coarse = np.arange(m)
    row_indices = np.arange(n)
    x_shifted = np.zeros((n, m))

    slope, offset = fit[0], fit[1]

    for idx in row_indices:
        row_shift = -slope * idx + offset
        x_scaled_shifted = oversample_factor * (x_coarse + row_shift)
        x_scaled_quantized = np.rint(x_scaled_shifted)
        x_shifted[idx] = x_scaled_quantized / oversample_factor

    x_oversampled = np.unique(x_shifted)

    esf_estimate = np.zeros_like(x_oversampled)
    for idx, x_val in enumerate(x_oversampled):
        indices = np.where(x_shifted == x_val)
        esf_mean = np.mean(rows[indices])
        esf_estimate[idx] = esf_mean

    if plot:

        plt.figure()
        for i in range(n):
            plt.scatter(x_shifted[i], rows[i])
        plt.plot(x_oversampled, esf_estimate)
        plt.title('shifted')
        plt.show()

    target_length = oversample_factor * m
    x_oversampled, esf_estimate = fix_sample_length(x_oversampled, esf_estimate,
                                                    target_length)

    return x_oversampled, esf_estimate, oversample_factor


def get_rer(esf, oversample_factor=4):

    esf = np.asarray(esf)
    esf = esf - np.min(esf)
    esf = esf / np.max(esf)

    mid_val = (np.max(esf) + np.min(esf)) / 2
    mid_point_idx = np.argmin(np.abs(esf - mid_val))
    left_val = esf[mid_point_idx - int(oversample_factor / 2)]
    right_val = esf[mid_point_idx + int(oversample_factor / 2)]
    rer = np.abs(right_val - left_val)

    return float(rer)


def fix_sample_length(x_oversampled, esf_estimate, target_length):

    n_start = len(x_oversampled)
    if n_start > target_length:
        x_oversampled = x_oversampled[-target_length:]
        esf_estimate = esf_estimate[-target_length:]

    if n_start < target_length:

        print(f'warning, {n_start - target_length} fewer edge samples than expected')
        x_ext = np.zeros(target_length)
        x_ext[:len(x_oversampled)] = x_oversampled
        delta_x = x_oversampled[1] - x_oversampled[0]
        last_x = x_oversampled[-1]
        num_extend = target_length - len(x_oversampled)
        extension = [(last_x + delta_x * (i + 1)) for i in range(num_extend)]
        x_ext[-num_extend:] = extension

        esf_ext = np.zeros_like(x_ext)
        esf_ext[-num_extend:] = esf_estimate[-1]

        return x_ext, esf_ext

    return x_oversampled, esf_estimate


def get_mtf(lsf):
    otf = np.fft.fft(lsf)
    mtf = np.abs(otf)
    mtf = mtf / np.max(mtf)
    return mtf


def estimate_mtf(chip, plot=False):

    centroids, fit = fit_edge(chip, plot=plot)
    x_oversampled, esf, oversample_factor = oversampled_esf(chip, fit, plot=plot)
    lsf = get_lsf(esf, pad=True)
    mtf = get_mtf(lsf)

    return mtf, esf, oversample_factor


def measure_props(dataset, directory, chip_data_key='chips', plot=False, chip_sub_directory=REL_PATHS['edge_chips']):

    blurred_chip_data = dataset[chip_data_key]

    output_dir = Path(directory, 'mtf_plots')
    if plot and not output_dir.is_dir():
        Path.mkdir(output_dir)

    if chip_sub_directory is not None:
        chip_directory = Path(directory, chip_sub_directory)
    else:
        chip_directory = directory
    properties = {}

    for distorted_chip_name in blurred_chip_data.keys():

        distorted_chip = get_image_array(chip_directory, distorted_chip_name)
        mtf, esf, oversample_factor = estimate_mtf(distorted_chip)
        rer = get_rer(esf, oversample_factor=oversample_factor)
        properties[str(distorted_chip_name)] = {
            'mtf': [float(val) for val in mtf],
            'esf': [float(val) for val in esf],
            'rer': rer,
            'oversample_factor': oversample_factor
        }
        if plot:
            plot_mtf(mtf, output_dir=output_dir, chip_name=distorted_chip_name)

    with open(Path(directory, 'mtf_lsf.json'), 'w') as file:
        json.dump(properties, file)

    return properties


def get_freq_axis(mtf, oversample_factor=4):
    f_max = oversample_factor / 2
    f = np.linspace(0, f_max, num=len(mtf))
    return f


def plot_mtf(mtf, output_dir=None, chip_name=None, f_max=None):

    freq_axis = get_freq_axis(mtf)
    n_plot = int((len(freq_axis) / 2))
    n_plot_nyquist = int(n_plot / 2)

    plt.figure()
    plt.plot(freq_axis[:n_plot], mtf[:n_plot], label='standard')
    plt.xlabel('spatial frequency [cycles / pixel]')
    plt.ylabel('MTF')
    plt.legend()
    if output_dir and chip_name:
        plt.savefig(Path(output_dir, chip_name))
    plt.show()

    plt.figure()
    plt.plot(freq_axis[:n_plot_nyquist], mtf[:n_plot_nyquist], label='standard')
    plt.xlabel('spatial frequency [cycles / pixel]')
    plt.ylabel('MTF')
    plt.legend()
    if output_dir and chip_name:
        plt.savefig(Path(output_dir, f'nyquist_{chip_name}'))
    plt.show()


def load_dataset(directory_key):

    directory = Path(ROOT_DIR, REL_PATHS['edge_datasets'], directory_key)
    with open(Path(directory, STANDARD_DATASET_FILENAME), 'r') as file:
        dataset = json.load(file)

    return directory, dataset


def get_image_array(directory, name):
    return np.asarray(Image.open(Path(directory, name)))


if __name__ == '__main__':

    _directory_keys = ['0033']

    for _directory_key in _directory_keys:
        _directory, _dataset = load_dataset(_directory_key)
        _properties = measure_props(_dataset, _directory, plot=False)

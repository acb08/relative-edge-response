from PIL import Image
import numpy as np
import src.rer_defs as rer_defs
import torch
import torchvision.transforms as transforms
from src.definitions import ROOT_DIR, REL_PATHS, STANDARD_DATASET_FILENAME
from pathlib import Path
import json
import argparse
import src.functions as functions
from scipy.signal import convolve2d
import src.fit as fit


def check_edge_buffer(edge_indices, kernel_size, down_sample_factor, chip_size):

    edge_buffer_left = np.floor(min(edge_indices - (kernel_size + 1)))
    edge_buffer_left = int(np.floor(edge_buffer_left / down_sample_factor)) - 1

    edge_buffer_right = np.ceil(max(edge_indices + (kernel_size + 1)))
    edge_buffer_right = chip_size - (int(np.ceil(edge_buffer_right) / down_sample_factor) + 1)
    edge_buffer = min(edge_buffer_left, edge_buffer_right)

    if edge_buffer < 10:
        print(f'Warning: narrow edge buffer: {edge_buffer}')

    return edge_buffer


def get_transition_frac(left_edge, right_edge, offset=0):

    """
    Calculates the fraction of two adjacent pixels that are on the left side of a near-vertical line passing through at
    least one of the pixels. The equation of the line is defined in by numpy array indices, so the +x-direction is
    down and the +y-direction is right. Accordingly, a line with a small positive slope is a near vertical edge that
    moves gradually left to right going from the top to the bottom of the image.

    :param left_edge: the boundary (in units of fraction of pixel width) of the line crossing the top edge of the two
    adjacent pixels
    :param right_edge: the boundary (in units of fraction of pixel width) of the line crossing the bottom edge of the
    two adjacent pixels
    :param offset: if left edge and right edge are specified in raw pixel values, subtracting the offset allows them to
    fall back on [0, 1]
    :return: fraction of each pixel to the left of the near vertical line
    """

    left_edge = left_edge - offset
    right_edge = right_edge - offset

    if left_edge < 0 or left_edge > 1:
        raise ValueError('left edge location must fall between 0 and 1')
    if right_edge < 0 or right_edge > 2:
        raise ValueError('right edge location must fall between 0 and 2')

    if right_edge <= 1:
        triangle_area_1 = 0.5 * (right_edge - left_edge)
        rectangle_area = left_edge
        total_area_1 = triangle_area_1 + rectangle_area
        total_area_2 = 0

    else:
        rectangle_area = 1
        triangle_area_1 = 0.5 * ((1 - left_edge) / (right_edge - left_edge)) * (1 - left_edge)
        total_area_1 = rectangle_area - triangle_area_1

        triangle_area_2 = 0.5 * ((right_edge - 1) / (right_edge - left_edge)) * (right_edge - 1)
        total_area_2 = triangle_area_2

    return total_area_1, total_area_2


def make_perfect_edge(size=rer_defs.pre_sample_size, theta=rer_defs.angle,
                      dark_val=rer_defs.lam_black_reflectance, light_val=rer_defs.lam_white_reflectance):

    """
    Generates an ideal, near-vertical edge image by defining a line in pixel (array) coordinates that starts at the
    midpoint of the array and moves from left to right going down in the array, with the edge going from dark on the
    left side to light on the right side.

    :param size: size of the image (size, size)
    :param theta: angle of the edge
    :param dark_val: the value of the dark pixels in the array
    :param light_val: the value of the light pixels in the array
    :return: edge image plus some diagnostic metadata
    """

    edge_start = size / 2
    row_indices = np.arange(size)

    edge_location_indices = np.arange(size + 1)
    slope = np.tan(np.pi * theta / 180)

    if slope < 0 or slope > 0.1:
        raise ValueError('slope, tan(theta), must fall between 0 and 0.1')

    top_edge_crossings = edge_start + slope * edge_location_indices

    edge_image = light_val * np.ones((size, size), dtype=np.float32)

    # diagnostic stuff
    edge_vals = np.zeros(size)
    edge_indices = np.zeros(size, dtype=np.int32)
    left_transition_pixel_fracs = np.zeros_like(edge_vals)
    right_transition_pixel_fracs = np.zeros_like(edge_vals)

    for row_idx in row_indices:

        edge_index = int(top_edge_crossings[row_idx])
        edge_indices[row_idx] = edge_index  # diagnostic

        left_edge_location = top_edge_crossings[row_idx]
        right_edge_location = top_edge_crossings[row_idx + 1]

        frac_1, frac_2 = get_transition_frac(left_edge_location, right_edge_location, offset=edge_index)

        left_transition_pixel_fracs[row_idx] = frac_1
        right_transition_pixel_fracs[row_idx] = frac_2

        edge_val_left = frac_1 * dark_val + (1-frac_1) * light_val
        edge_val_right = frac_2 * dark_val + (1 - frac_2) * light_val

        edge_vals[row_idx] = edge_val_left

        edge_image[row_idx, :edge_index] = dark_val
        edge_image[row_idx, edge_index] = edge_val_left
        edge_image[row_idx, edge_index+1] = edge_val_right

    return edge_image, top_edge_crossings, left_transition_pixel_fracs, right_transition_pixel_fracs, edge_vals


def get_kernel_size(std):
    return 8 * int(np.round(std, 0)) + 1


def apply_gaussian_blur(edge_image, kernel_size, sigma):

    edge_image = torch.tensor(edge_image, requires_grad=False)
    edge_image = torch.unsqueeze(edge_image, dim=0)
    edge_image = transforms.GaussianBlur(kernel_size, sigma=sigma)(edge_image)
    edge_image = torch.squeeze(edge_image, dim=0)
    edge_image = edge_image.numpy()

    return edge_image


def apply_optical_blur(edge_image, kernel):

    pad_width = np.shape(kernel)[0] // 2
    if np.abs(np.sum(kernel) - 1) > 0.9:
        raise ValueError('kernel must be normalized')
    edge_image = convolve2d(edge_image, kernel, boundary='symm')
    edge_image = edge_image[pad_width:-pad_width, pad_width:-pad_width]

    return edge_image


def integer_downsample(image, downsample_step_size):
    """
    Performs an interpolation-free down sampling by combining n-by-n pixel blocks to form new a new downscaled image,
    where n must be an integer.
    """

    if np.max(image) > 1 or np.min(image) < 0:
        raise ValueError('input image must fall on [0, 1]')

    start_size, compare = np.shape(image)[0], np.shape(image)[1]

    if start_size != compare:
        raise Exception('input image must be square')
    new_size, remainder = divmod(start_size, downsample_step_size)

    # if rem != 0 or np.log2(div) != int(np.log2(div)):
    #     raise Exception('p2_downsample requires down sampling by a power of 2')
    # new_size = int(start_size / downsample_step_size)

    chip = 2 * np.ones((new_size, new_size))  # initialize with all values set to 2 to verify now stowaway initial vals

    for i in range(new_size):
        v_idx = i * downsample_step_size
        for j in range(new_size):
            h_idx = j * downsample_step_size
            sample = image[v_idx:v_idx + downsample_step_size, h_idx:h_idx + downsample_step_size]
            chip[i, j] = np.mean(sample)

    return chip, new_size


def convert_to_pil(image):
    if np.max(image) > 1 or np.min(image) < 0:
        raise ValueError('input image values must fall between 0 and 1')
    image = 255 * image
    image = np.asarray(image, dtype=np.uint8)
    image = Image.fromarray(image)
    return image


def generate_blurred_edges_and_chips(blur_iterable, blur_kernel_dir, using_external_blur_kernels, perfect_edge,
                                     edge_indices, edge_output_dir, chip_output_dir, down_sample_ratios,
                                     edge_filename_stem='opt_edge',
                                     chip_filename_stem='opt_chip'):
    chips = {}
    edges = []
    nearest_gaussian_stds = []

    for i, entry in enumerate(blur_iterable):

        if using_external_blur_kernels:
            blur_kernel_filename = entry
            kernel_data = functions.load_npz_data(blur_kernel_dir, blur_kernel_filename)

            kernel = kernel_data['kernel']
            p_smear = float(kernel_data['p_smear'])
            sigma_jitter = float(kernel_data['sigma_jitter'])
            wl_rms = float(kernel_data['wl_rms'])
            l_corr = float(kernel_data['l_corr'])

            blurred_edge = apply_optical_blur(perfect_edge, kernel)
            kernel_size = int(np.shape(kernel)[0])

            std, __ = fit.nearest_gaussian_psf(kernel)
            std = float(std)
            nearest_gaussian_stds.append(std)

        else:
            std = entry
            kernel_size = get_kernel_size(std)
            blurred_edge = apply_gaussian_blur(perfect_edge, kernel_size, std)

            p_smear = None
            sigma_jitter = None
            wl_rms = None
            l_corr = None

        blurred_edge_save = convert_to_pil(blurred_edge)
        blurred_edge_name = f'{edge_filename_stem}_{i}.png'
        edges.append(blurred_edge_name)
        blurred_edge_save.save(Path(edge_output_dir, blurred_edge_name))

        for j, down_sample_factor in enumerate(down_sample_ratios):

            scaled_std = std / down_sample_factor
            chip, chip_size = integer_downsample(blurred_edge, down_sample_factor)
            edge_buffer = check_edge_buffer(edge_indices, kernel_size, down_sample_factor, chip_size)

            chip = convert_to_pil(chip)
            chip_name = f'{chip_filename_stem}_{i}_{j}.png'
            chips[chip_name] = {
                'native_blur': std,
                'native_blur_kernel_size': kernel_size,
                'scaled_blur': scaled_std,
                'down_sample_factor': down_sample_factor,
                'edge_buffer': edge_buffer,
                'parents': [blurred_edge_name],

                'optical': using_external_blur_kernels,
                'p_smear': p_smear,
                'sigma_jitter': sigma_jitter,
                'wl_rms': wl_rms,
                'l_corr': l_corr,
            }

            chip.save(Path(chip_output_dir, chip_name))

    return chips, edges, nearest_gaussian_stds


def make_edge_chips(config):

    native_stds = config['native_stds']
    down_sample_ratios = config['down_sample_ratios']

    if 'blur_kernel_dir_key' in config.keys():
        if native_stds is not None:
            raise Exception('cannot use both native_stds and externally generated blur kernels')

        blur_kernel_dir_key = config['blur_kernel_dir_key']
        blur_kernel_dir = Path(ROOT_DIR, REL_PATHS['blur_kernels'], blur_kernel_dir_key)
        blur_kernel_filenames = list(blur_kernel_dir.iterdir())
        blur_kernel_filenames = [filename for filename in blur_kernel_filenames if str(filename)[-3:] == 'npz']
        external_kernels = True

        generate_gaussian_equiv_chips = config['generate_gaussian_equiv_chips']

    else:
        external_kernels = False
        generate_gaussian_equiv_chips = False
        blur_kernel_filenames = None
        blur_kernel_dir = None

    edge_image_size = rer_defs.pre_sample_size
    theta = rer_defs.angle  # degrees
    dark_reflectance = rer_defs.lam_black_reflectance
    light_reflectance = rer_defs.lam_white_reflectance

    perfect_edge, edge_indices, __, __, __ = make_perfect_edge(size=edge_image_size, theta=theta,
                                                               dark_val=dark_reflectance,
                                                               light_val=light_reflectance)

    save_dir_parent = Path(ROOT_DIR, REL_PATHS['edge_datasets'])
    if not save_dir_parent.is_dir():
        Path.mkdir(save_dir_parent, parents=True)

    key, __, __ = functions.key_from_dir(save_dir_parent)
    save_dir = Path(save_dir_parent, key)
    chip_dir = Path(save_dir, REL_PATHS['edge_chips'])
    edge_dir = Path(save_dir, REL_PATHS['edges'])

    if not chip_dir.is_dir():
        Path.mkdir(chip_dir, parents=True, exist_ok=True)
    if not edge_dir.is_dir():
        Path.mkdir(edge_dir, parents=True, exist_ok=True)

    perfect_edge_save = convert_to_pil(perfect_edge)
    perfect_edge_name = 'perfect_edge.png'
    perfect_edge_save.save(Path(edge_dir, perfect_edge_name))

    # chips = {}
    edges = [perfect_edge_name]

    if external_kernels:
        chips, opt_edges, nearest_gaussian_stds = generate_blurred_edges_and_chips(blur_kernel_filenames,
                                                                                   blur_kernel_dir,
                                                                                   external_kernels,
                                                                                   perfect_edge,
                                                                                   edge_indices,
                                                                                   edge_dir,
                                                                                   chip_dir,
                                                                                   down_sample_ratios,
                                                                                   edge_filename_stem='opt_edge',
                                                                                   chip_filename_stem='opt_chip')
        edges.extend(opt_edges)

        if generate_gaussian_equiv_chips:
            gauss_equiv_chips, gauss_edges, __ = generate_blurred_edges_and_chips(nearest_gaussian_stds,
                                                                                  blur_kernel_dir=None,
                                                                                  using_external_blur_kernels=False,
                                                                                  perfect_edge=perfect_edge,
                                                                                  edge_indices=edge_indices,
                                                                                  edge_output_dir=edge_dir,
                                                                                  chip_output_dir=chip_dir,
                                                                                  down_sample_ratios=down_sample_ratios,
                                                                                  edge_filename_stem='gauss_edge',
                                                                                  chip_filename_stem='gauss_chip')

            chips.update(gauss_equiv_chips)
            edges.extend(gauss_edges)

    else:
        chips, gauss_edges, __ = generate_blurred_edges_and_chips(native_stds,
                                                                  blur_kernel_dir=None,
                                                                  using_external_blur_kernels=False,
                                                                  perfect_edge=perfect_edge,
                                                                  edge_indices=edge_indices,
                                                                  edge_output_dir=edge_dir,
                                                                  chip_output_dir=chip_dir,
                                                                  down_sample_ratios=down_sample_ratios,
                                                                  edge_filename_stem='gauss_edge',
                                                                  chip_filename_stem='gauss_chip')

        edges.extend(gauss_edges)

    # for i, std in enumerate(native_stds):
    #
    #     kernel_size = get_kernel_size(std)
    #     blurred_edge = apply_gaussian_blur(perfect_edge, kernel_size, std)
    #
    #     blurred_edge_save = convert_to_pil(blurred_edge)
    #     blurred_edge_name = f'gaussian_blurred_edge_{i}.png'
    #     blurred_edge_save.save(Path(edge_dir, blurred_edge_name))
    #
    #     for j, down_sample_factor in enumerate(down_sample_ratios):
    #
    #         # if i == 0:
    #         #
    #         #     perfect_edge_chip, chip_size = integer_downsample(perfect_edge, down_sample_factor)
    #         #     edge_buffer = check_edge_buffer(edge_indices, 1, down_sample_factor, chip_size)
    #         #     perfect_edge_chip = convert_to_pil(perfect_edge_chip)
    #         #     perfect_edge_chip_name = 'perfect_edge_chip.png'
    #         #     chips[perfect_edge_chip_name] = {
    #         #         'native_blur': 0,
    #         #         'native_blur_kernel_size': None,
    #         #         'scaled_blur': 0,
    #         #         'down_sample_factor': down_sample_factor,
    #         #         'edge_buffer': edge_buffer,
    #         #         'parents': [perfect_edge_name],
    #         #
    #         #         'p_smear': None,
    #         #         'sigma_jitter': None,
    #         #         'wl_rms': None,
    #         #         'l_corr': None
    #         #     }
    #         #     perfect_edge_chip.save(Path(chip_dir, perfect_edge_chip_name))
    #
    #         scaled_std = std / down_sample_factor
    #
    #         chip, chip_size = integer_downsample(blurred_edge, down_sample_factor)
    #         edge_buffer = check_edge_buffer(edge_indices, kernel_size, down_sample_factor, chip_size)
    #
    #         chip = convert_to_pil(chip)
    #         chip_name = f'gauss_chip_{i}_{j}.png'
    #         chips[chip_name] = {
    #             'native_blur': std,
    #             'native_blur_kernel_size': kernel_size,
    #             'scaled_blur': scaled_std,
    #             'down_sample_factor': down_sample_factor,
    #             'edge_buffer': edge_buffer,
    #             'parents': [perfect_edge_name, blurred_edge_name],
    #
    #             'p_smear': None,
    #             'sigma_jitter': None,
    #             'wl_rms': None,
    #             'l_corr': None,
    #         }
    #         chip.save(Path(chip_dir, chip_name))

    metadata = {
        'image_size': edge_image_size,
        'down_sample_ratios': down_sample_ratios,
        'native_stds': native_stds,
    }

    dataset = {
        'chips': chips,
        'edges': edges,
        'metadata': metadata
    }

    with open(Path(save_dir, STANDARD_DATASET_FILENAME), 'w') as file:
        json.dump(dataset, file)

    functions.log_config(save_dir, config)

    print(key)


if __name__ == '__main__':

    chip_config_filename = 'chip_config_exk_pan.yml'

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default=chip_config_filename, help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(ROOT_DIR, 'rer', 'chip_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = functions.get_config(args_passed)

    make_edge_chips(run_config)

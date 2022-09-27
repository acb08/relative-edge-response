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


def make_perfect_edge(size=rer_defs.pre_sample_size, theta=rer_defs.angle,
                      dark_val=rer_defs.lam_black_reflectance, light_val=rer_defs.lam_white_reflectance,
                      half_step=True):

    edge_start = size / 2
    row_indices = np.arange(size)

    edge_location_indices = np.arange(size + 1)
    edge_locations = edge_start + np.tan(np.pi * theta / 180) * edge_location_indices

    edge_image = light_val * np.ones((size, size), dtype=np.float32)
    edge_vals = np.zeros(size)
    edge_indices = np.zeros(size, dtype=np.int)
    transition_pixel_fracs = np.zeros_like(edge_vals)

    for row_idx in row_indices:
        edge_index = int(edge_locations[row_idx])
        edge_indices[row_idx] = edge_index
        transition_pixel_frac = 1 - (0.5 * (edge_locations[row_idx] + edge_locations[row_idx + 1]) - edge_index)
        transition_pixel_fracs[row_idx] = transition_pixel_frac
        edge_val = transition_pixel_frac * (light_val - dark_val)
        edge_vals[row_idx] = edge_val
        edge_image[row_idx, :edge_index] = dark_val
        edge_image[row_idx, edge_index] = edge_val

    return edge_image, edge_indices, edge_locations, transition_pixel_fracs, edge_vals


def get_blur_parameters(target_q, scale_factor, conversion=rer_defs.airy_gauss_conversion):

    airy_radius = scale_factor * target_q
    std = airy_radius / conversion
    kernel_size = 8 * int(std) + 1

    return std, kernel_size


def apply_optical_blur(edge_image, kernel_size, sigma):

    edge_image = torch.tensor(edge_image, requires_grad=False)
    edge_image = torch.unsqueeze(edge_image, dim=0)
    edge_image = transforms.GaussianBlur(kernel_size, sigma=sigma)(edge_image)
    edge_image = torch.squeeze(edge_image, dim=0)
    edge_image = edge_image.numpy()

    return edge_image


def p2_downsample(image, downsample_step_size):
    """
    Performs an interpolation-free down sampling by combining n-by-n pixel regions to form new a new downscaled image,
    where n is a power of 2
    """
    start_size, compare = np.shape(image)[0], np.shape(image)[1]

    if start_size != compare:
        raise Exception('input image must be square')
    div, rem = divmod(start_size, downsample_step_size)
    if rem != 0 or np.log2(div) != int(np.log2(div)):
        raise Exception('p2_downsample requires down sampling by a power of 2')
    new_size = int(start_size / downsample_step_size)
    chip = np.zeros((new_size, new_size))

    for i in range(new_size):
        v_idx = i * downsample_step_size
        for j in range(new_size):
            h_idx = j * downsample_step_size
            sample = image[v_idx:v_idx + downsample_step_size, h_idx:h_idx + downsample_step_size]
            chip[i, j] = np.mean(sample)

    return chip


def convert_to_pil(image):
    if np.max(image) > 1 or np.min(image) < 0:
        raise ValueError('input image values must fall between 0 and 1')
    image = 256 * image
    image = np.asarray(image, dtype=np.uint8)
    image = Image.fromarray(image)
    return image


def make_edge_chips(config):

    q_values = config['q_values']
    chip_size = config['chip_size']

    edge_image_size = rer_defs.pre_sample_size
    theta = rer_defs.angle
    half_step = rer_defs.half_step
    dark_reflectance = rer_defs.lam_black_reflectance
    light_reflectance = rer_defs.lam_white_reflectance

    scale_factor = int(edge_image_size / chip_size)

    perfect_edge, edge_indices = make_perfect_edge(size=edge_image_size, theta=theta, dark_val=dark_reflectance,
                                                   light_val=light_reflectance)

    save_dir_parent = Path(ROOT_DIR, REL_PATHS['analysis'], REL_PATHS['rer_study'])
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

    stds = []
    kernel_sizes = []
    for q in q_values:
        std, k = get_blur_parameters(q, scale_factor)
        stds.append(std)
        kernel_sizes.append(k)

    kernel_size = max(kernel_sizes)  # just go with the max kernel size
    edge_buffer_left = np.floor(min(edge_indices - (kernel_size + 1)))
    edge_buffer_left = int(np.floor(edge_buffer_left / scale_factor)) - 1
    edge_buffer_right = np.ceil(max(edge_indices + (kernel_size + 1)))
    edge_buffer_right = int(np.ceil(edge_buffer_right) / scale_factor) + 1
    edge_buffer = min(edge_buffer_left, edge_buffer_right)
    if edge_buffer < 10:
        print(f'Warning: narrow edge buffer: {edge_buffer}')

    chips = {}
    edges = [perfect_edge_name]

    for i, std in enumerate(stds):

        blurred_edge = apply_optical_blur(perfect_edge, kernel_size, std)

        blurred_edge_save = convert_to_pil(blurred_edge)
        blurred_edge_name = f'blurred_edge_{i}.png'
        blurred_edge_save.save(Path(edge_dir, blurred_edge_name))
        edges.append(blurred_edge_name)

        chip = p2_downsample(blurred_edge, scale_factor)
        chip = convert_to_pil(chip)
        chip_name = f'chip_{i}.png'
        chips[chip_name] = {
            'native_blur': q_values[i],
            'edge_buffer': edge_buffer,
            'parents': [perfect_edge_name, blurred_edge_name]
        }
        chip.save(Path(chip_dir, chip_name))

    perfect_edge_chip = p2_downsample(perfect_edge, scale_factor)
    perfect_edge_chip = convert_to_pil(perfect_edge_chip)
    perfect_edge_chip_name = 'perfect_edge_chip.png'
    chips[perfect_edge_chip_name] = {
        'native_blur': 0,
        'edge_buffer': edge_buffer,
        'parents': [perfect_edge_name]
    }
    perfect_edge_chip.save(Path(chip_dir, perfect_edge_chip_name))

    metadata = {
        'image_size': edge_image_size,
        'scale_factor': scale_factor,
        'q_values': q_values,
        'half_step': half_step
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='chip_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'chip_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = functions.get_config(args_passed)

    make_edge_chips(run_config)

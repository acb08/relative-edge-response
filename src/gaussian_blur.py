import json
from torchvision import transforms
from PIL import Image
from pathlib import Path
import src.functions as functions
import src.definitions as definitions
import numpy as np
import copy
import argparse


def apply_blur(img, std, kernel_size):
    return transforms.GaussianBlur(kernel_size=kernel_size, sigma=std)(img), 'std', std


def blur_chips(chip_dir, blurred_chip_dir, kernel_size, std_values, datatype_key='np.uint8'):

    directory = Path(chip_dir).parents[0]

    if not blurred_chip_dir.is_dir():
        Path.mkdir(blurred_chip_dir)

    dataset = functions.read_json_artifact(directory, definitions.STANDARD_DATASET_FILENAME)
    chips = dataset['chips']
    blurred_chips = {}

    for chip_name, chip_data in chips.items():
        chip = Image.open(Path(chip_dir, chip_name))
        for i, std in enumerate(std_values):

            chip_name_stem = chip_name.split('.')[0]
            blurred_chip_name = f'{chip_name_stem}_{i}.png'

            blurred_chip, val, distortion_key = apply_blur(chip, std, kernel_size)
            blurred_chip = np.array(blurred_chip, dtype=definitions.DATATYPE_MAP[datatype_key])
            blurred_chip = Image.fromarray(blurred_chip)
            blurred_chip.save(Path(blurred_chip_dir, blurred_chip_name))

            blurred_chips[blurred_chip_name] = copy.deepcopy(chips[chip_name])
            blurred_chips[blurred_chip_name]['parents'].append(chip_name)
            blurred_chips[blurred_chip_name]['std'] = std
            blurred_chips[blurred_chip_name]['kernel_size'] = kernel_size

    dataset['blurred_chips'] = blurred_chips
    with open(Path(blurred_chip_dir, definitions.STANDARD_DATASET_FILENAME), 'w') as f:
        json.dump(dataset, f)


def main(config):

    blur_config_project_id = config['blur_config_project_id']
    num_blur_vals = config['num_blur_values']
    dir_numbers = config['dir_numbers']

    kernel_size, std_min, std_max = definitions.DISTORTION_RANGE[blur_config_project_id]['blur']
    std_values = np.linspace(std_min, std_max, num=num_blur_vals, endpoint=True)

    for dir_number in dir_numbers:
        directory = Path(definitions.ROOT_DIR, definitions.REL_PATHS['analysis'], definitions.REL_PATHS['rer_study'],
                         dir_number)
        chip_dir = Path(directory, 'edge_chips')
        blurred_chip_dir = Path(directory, f'distorted_chips_{kernel_size}')

        blur_chips(chip_dir, blurred_chip_dir, kernel_size, std_values)
        functions.log_config(blurred_chip_dir, config)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--config_name', default='blur_config.yml', help='config filename to be used')
    parser.add_argument('--config_dir',
                        default=Path(Path(__file__).parents[0], 'blur_configs'),
                        help="configuration file directory")
    args_passed = parser.parse_args()
    run_config = functions.get_config(args_passed)

    main(run_config)

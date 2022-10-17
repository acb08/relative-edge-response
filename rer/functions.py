""""
Miscellaneous functions for data loading/logging
"""

from rer.definitions import ROOT_DIR, REL_PATHS, KEY_LENGTH, STANDARD_CONFIG_USED_FILENAME
import json
from pathlib import Path
from yaml import safe_load, dump
import numpy as np


def get_sub_dirs(parent_dir):

    """
    :param parent_dir: directory for evaluation, absolute path
    :return: list of non-hidden subdirectories in parent_dir
    """

    sub_dirs = []

    for item in Path(parent_dir).iterdir():
        if item.is_dir():
            sub_dirs.append(item.name)

    return sub_dirs


def dict_val_lists_to_arrays(data_dict, dtype=np.float32):

    for key, val in data_dict.items():
        data_dict[key] = np.asarray(val, dtype=dtype)

    return data_dict


def key_num_from_list(items):

    """
    gets the largest key value found in items
    :param items: list of strings to be checked for KEY_LENGTH keys
    :return: the integer value of the largest key
    """

    key_length = KEY_LENGTH
    key_num = 0
    keyed_items_count = 0
    keyed_items_found = False
    problematic_key_flag = False
    problematic_items = []

    if len(items) > 0:

        for item in items:

            check_string = item[:key_length]
            check_string_ext = item[:key_length + 1]

            if len(check_string_ext) != key_length and check_string_ext.isdecimal():
                problematic_items.append(item)
                problematic_key_flag = True

            # screen out non-keyed sub_directories
            if check_string.isdecimal():
                keyed_items_found = True
                keyed_items_count += 1
                check_key_num = int(check_string)
                key_num = max(key_num, check_key_num)

    if keyed_items_found:
        key_num += 1

    return keyed_items_count, key_num, problematic_key_flag, problematic_items


def key_from_dir(parent_dir):

    """
    generates a 4-digit key string, where the number represented by the string is the larger of (a) the number of
    sub_directories in parent_dir or (b) the largest number encoded by an existing sub-directory + 1 if sub-directories
    exist.
    :param parent_dir: directory for which the key will be generated
    :return: 4-digit key string which should be unique within the parent_dir subdirectories (but recommend
    double checking before a potential overwrite :p )
    """

    key_length = KEY_LENGTH
    sub_dirs = get_sub_dirs(parent_dir)

    keyed_dir_count, key_num, problem_key_flag, problematic_items = key_num_from_list(sub_dirs)

    if problem_key_flag:
        print('potential directory key problem(s):')
        print(problematic_items)

    key_num = max(keyed_dir_count, key_num)
    key_str = str(key_num)

    return key_str.zfill(key_length), problem_key_flag, problematic_items


def get_config(args):

    with open(Path(args.config_dir, args.config_name), 'r') as file:
        config = safe_load(file)

    print(json.dumps(config, indent=3))

    return config


def log_config(output_dir, config, return_path=False, config_used_filename=STANDARD_CONFIG_USED_FILENAME):

    log_path = Path(output_dir, config_used_filename)

    with open(log_path, 'w') as file:
        dump(config, file)

    if return_path:
        return log_path


def read_json_artifact(directory, filename):

    with open(Path(directory, filename), 'r') as f:
        data = json.load(f)

    return data


def load_npz_data(directory, filename):
    data = np.load(Path(directory, filename))
    return data

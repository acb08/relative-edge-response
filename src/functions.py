import torchvision.models as models
import torch
from src.definitions import ROOT_DIR, REL_PATHS, KEY_LENGTH, STANDARD_CONFIG_USED_FILENAME
# ORIGINAL_DATASETS, KEY_LENGTH, ARTIFACT_TYPE_TAGS NUM_CLASSES
import json
from pathlib import Path
from yaml import safe_load, dump
import numpy as np
# from src.d00_utils.classes import Sat6ResNet, Sat6ResNet50, Sat6DenseNet161
import copy
import time


# def load_original_dataset(dataset_id):
#
#     """
#     :param dataset_id: dictionary key for ORIGINAL_DATASETS defined in definitions.py
#     :return: dictionary with the relative path to dataset umbrella, relative path to from dataset to image directory,
#     and list of images names and labels  in format [(image_name.jpg, image_label), ...]
#     """
#
#     path_info = ORIGINAL_DATASETS[dataset_id]
#     dataset_dir = path_info['rel_path']
#     # img_dir = REL_PATHS['images']
#     label_filename = path_info['names_labels_filename']
#     names_labels = []
#
#     with open(Path(ROOT_DIR, dataset_dir, label_filename)) as f:
#         for i, line in enumerate(f):
#             img_name, img_label = line.split()
#             if img_name[0] == '/':
#                 img_name = img_name[1:]
#             names_labels.append((img_name, img_label))
#
#     image_metadata = {
#         'dataset_dir': dataset_dir,
#         # 'img_dir': img_dir,
#         'names_labels': names_labels
#     }
#
#     return image_metadata


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


def get_path_and_key(artifact_type):

    """
    Maps use_tags to appropriate directories in project structure to save script output data and provides a sequential
    key for both (1) naming new directory to house output data and (2) referencing run information in a json database

    :param artifact_type: purpose of the data to be generated

    :return:
        rel_path: path from root_dir to appropriate target directory based on use_tag
        key: unique, sequential key for creating (1) new directory to hold output data and (2) referencing metadata
        in a .json file
    """

    rel_path = REL_PATHS[artifact_type]
    target_dir = Path(ROOT_DIR, rel_path)
    dir_key_query_result = key_from_dir(target_dir)

    return rel_path, dir_key_query_result


# def id_from_tags(artifact_type, tags, return_dir=False):
#
#     target_dir, dir_key_query_result = get_path_and_key(artifact_type)
#     dir_key = dir_key_query_result[0]
#
#     name = dir_key
#     type_tag = ARTIFACT_TYPE_TAGS[artifact_type]
#
#     local_tags = copy.deepcopy(tags)  # avoids having the type tag inserted in the mutable list passed to the function
#     if type_tag != 'mdl':  # for models, tags[0] = arch (e.g. resnet18), so model type tag unnecessary
#         local_tags.insert(0, type_tag)
#
#     tag_string = string_from_tags(local_tags)
#     name = name + tag_string
#
#     if not return_dir:
#         return name
#     else:
#         artifact_rel_dir = Path(REL_PATHS[artifact_type], name)
#         return name, artifact_rel_dir


def string_from_tags(tags):

    tag_string = ""
    for tag in tags:
        tag_string = tag_string + f"-{tag}"

    return tag_string


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


def potential_overwrite(path):
    return path.exists()


# def load_wandb_data_artifact(run, artifact_id, artifact_filename, retries=10):
#
#     attempts = 0
#     success = False
#     while attempts < retries and not success:
#         try:
#             artifact = run.use_artifact(artifact_id)
#         except Exception as e:
#             print(f'wandb artifact access failed attempt {attempts}, {artifact_id}: \n {e}')
#             attempts += 1
#             time.sleep(60)
#         else:
#             success = True
#
#     artifact_dir = artifact.download()
#     data = read_json_artifact(artifact_dir, artifact_filename)
#
#     return artifact, data

#
# def log_model_helper(model_dir, data):
#
#     """
#     Log a helper file to enable easy loading of models with potentially different filenames
#     """
#
#     filename = 'helper.json'
#     file_path = Path(model_dir, filename)
#     with open(file_path, 'w') as f:
#         json.dump(data, f)
#
#     return file_path


def read_json_artifact(directory, filename):

    with open(Path(directory, filename), 'r') as f:
        data = json.load(f)

    return data


# def load_npz_data(directory, filename):
#
#     data = np.load(Path(directory, filename))
#
#     return data
#
#
# def load_data_vectors(shard_id, directory):
#     """
#     Extracts image and data vectors from the .npz file corresponding to shard_id in directory. Intended to provide
#     image/label vectors to create an instance of the NumpyDatasetBatchDistortion class.
#     """
#
#     data = load_npz_data(directory, shard_id)
#
#     image_vector = data['images']
#     label_vector = data['labels']
#
#     return image_vector, label_vector


def get_model_path(model_rel_dir, model_filename):
    """
    Arguments named so function can be called with **model_file_config
    """
    model_path = Path(ROOT_DIR, model_rel_dir, model_filename)
    return model_path


def check_model_metadata(model_metadata):

    """
    Verifies that model_metadata contains the minimum necessary information for model
    saving/loading
    """

    min_key_set = {'model_file_config', 'arch', 'artifact_type'}
    metadata_key_set = set(model_metadata.keys())
    return min_key_set.issubset(metadata_key_set)

#
# def save_model(model, model_metadata):
#
#     """
#     Save Pytorch model and its associated metadata and return paths for each. Paths are returned to enable
#     subsequent wandb logging.
#     """
#
#     if not check_model_metadata(model_metadata):
#         raise Exception("Model metadata lacks a key (one or more of {'model_file_config', 'arch', 'artifact_type'}")
#
#     model_file_config = model_metadata['model_file_config']  # i.e. {'model_rel_dir': '/foo', 'model_filename': 'bar.pt}
#     model_path = get_model_path(**model_file_config)
#     model_dir = model_path.parents[0]
#     if not model_dir.is_dir():
#         Path.mkdir(model_dir, parents=True)
#
#     torch.save(model.state_dict(), model_path)
#     helper_path = log_model_helper(model_dir, model_metadata)  # save model metadata in json file
#
#     return model_path, helper_path  # paths returned for logging


# def load_model(model_path, arch):
#
#     # if arch == 'resnet18':
#     #     model = models.__dict__[arch](num_classes=365)
#     #     # model.load_state_dict(torch.load(model_path))
#
#     if arch == 'resnet18_sat6':
#         model = Sat6ResNet()
#     elif arch == 'resnet50_sat6':
#         model = Sat6ResNet50()
#     elif arch == 'densenet161_sat6':
#         model = Sat6DenseNet161()
#
#     else:
#         model = models.__dict__[arch](num_classes=NUM_CLASSES)
#
#     model.load_state_dict(torch.load(model_path))
#
#     return model


# def load_wandb_model_artifact(run, artifact_id, return_configs=False):
#
#     artifact = run.use_artifact(artifact_id)
#     artifact_dir = artifact.download()
#     artifact_abs_dir = Path(Path.cwd(), artifact_dir)
#
#     helper_data = read_json_artifact(artifact_abs_dir, 'helper.json')
#     arch = helper_data['arch']
#     # arch = 'resnet18_sat6'
#     artifact_type = helper_data['artifact_type']
#     model_filename = helper_data['model_file_config']['model_filename']
#     model_path = Path(artifact_dir, model_filename)
#     model = load_model(model_path, arch)
#
#     if not return_configs:
#         return model
#
#     else:
#         return model, arch, artifact_type


def increment_suffix(suffix):
    suffix_num = int(suffix[1:])
    suffix_num += 1
    new_suffix = f'v{suffix_num}'
    return new_suffix


def construct_artifact_id(artifact_name, artifact_alias=None):

    if ':' in artifact_name:
        artifact_stem = artifact_name.split(':')[0]
        return artifact_name, artifact_stem

    artifact_stem = artifact_name

    if not artifact_alias:
        return f'{artifact_name}:latest', artifact_stem

    return f'{artifact_name}:{artifact_alias}', artifact_stem



import os

"""
Contains global constants. 

Additional note: throughout project code, dataset generally refers to dataset metadata rather than the image files 
themselves. The dataset effectively records relevant metadata and relative paths to image files. This approach 
simplifies integration with W&B and allows quick artifact downloads, where the artifacts themselves are pointers
to local image files. 

"""

ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '../..'))

STANDARD_DATASET_FILENAME = 'dataset.json'
STANDARD_BLUR_KERNEL_FILENAME_STEM = 'blur_kernel'
STANDARD_CONFIG_USED_FILENAME = 'config_used.yml'
KEY_LENGTH = 4

LORENTZ_TERMS = (0.2630388847587775, -0.4590111280646474)  # correction for Gaussian to account for pixel xfer function

# defines standard paths in project structure for different artifact types
REL_PATHS = {
    'edge_datasets': 'edge_datasets',
    'transfer_function': 'transfer_function',
    'blur_kernels': 'blur_kernels',
    'edges': 'edges',
    'rer_study': 'rer_study',
    'edge_chips': 'edge_chips',
}

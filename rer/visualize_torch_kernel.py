import matplotlib.pyplot as plt
from rer.build_edge_dataset import get_kernel_size
import numpy as np
import torch
from torch import Tensor
from typing import List


def _get_gaussian_kernel1d(kernel_size: int, sigma: float) -> Tensor:
    """
    _get_gaussian_kernel1d() returns the underlying Gaussian for the kernel in torchvision.transforms.GaussianBlur(),
    with several functions called along the way. The version here was copied from
    https://github.com/pytorch/vision/blob/main/torchvision/transforms/functional_tensor.py on 14 Dec 2022.
    """
    ksize_half = (kernel_size - 1) * 0.5

    x = torch.linspace(-ksize_half, ksize_half, steps=kernel_size)
    pdf = torch.exp(-0.5 * (x / sigma).pow(2))
    kernel1d = pdf / pdf.sum()

    return kernel1d


def _get_gaussian_kernel2d(
    kernel_size: List[int], sigma: List[float], dtype: torch.dtype, device: torch.device
) -> Tensor:
    kernel1d_x = _get_gaussian_kernel1d(kernel_size[0], sigma[0]).to(device, dtype=dtype)
    kernel1d_y = _get_gaussian_kernel1d(kernel_size[1], sigma[1]).to(device, dtype=dtype)
    kernel2d = torch.mm(kernel1d_y[:, None], kernel1d_x[None, :])
    return kernel2d


def plot_kernel_1d(std, ax=None):

    kernel_size = get_kernel_size(std)
    kernel = _get_gaussian_kernel1d(kernel_size, std)
    kernel = kernel.numpy()
    endpoint = kernel_size // 2
    x_plot = np.arange(-endpoint, endpoint + 1)
    if ax is None:
        fig, ax = plt.subplots()

    annotation = r'$\sigma=$'
    annotation = f'{annotation} {std}'
    ax.annotate(annotation, (2, 0.85))
    ax.plot(x_plot, kernel, color='k')
    ax.set_ylabel('amplitude')
    ax.set_xlabel('pixel')
    ax.label_outer()


if __name__ == '__main__':

    # _std_vals = [[0.1, 0.2, 0.3], [0.5, 0.7, 1], [1.5, 2, 2.5]]
    _std_vals = [[0.1, 0.3, 0.5], [0.7, 1.25, 2]]

    _n_rows = len(_std_vals)
    _n_cols = len(_std_vals[0])

    _fig_width = 8
    _fig_height = 2 * _n_rows

    fig, axes = plt.subplots(_n_rows, _n_cols, sharey=True, sharex=True, figsize=(_fig_width, _fig_height))
    for i, row in enumerate(_std_vals):
        for j, _std in enumerate(row):
            _ax = axes[i, j]
            plot_kernel_1d(_std, _ax)

    fig.tight_layout()
    fig.show()





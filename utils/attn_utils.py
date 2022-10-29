import torch
import numpy as np
from torch import Tensor, nn
from torch.nn import functional as F


def pad_to_multiple_2d(x, block_shape):
    '''
    PyTorch version
    '''
    old_shape = x.shape
    last = old_shape[-1]
    if len(old_shape) == 4:
        height_pad = -x.shape[1] % block_shape[0]
        width_pad = -x.shape[2] % block_shape[1]
        paddings = [0, 0, 0, width_pad, 0, height_pad]
    elif len(old_shape) == 5:
        height_pad = -x.shape[2] % block_shape[0]
        width_pad = -x.shape[3] % block_shape[1]
        paddings = [0, 0, 0, width_pad, 0, height_pad]
    padded_x = F.pad(x, paddings)
    return padded_x


def gather_indices_2d(x: Tensor, block_shape, block_stride):
    '''
    PyTorch version
    '''
    import functools, operator
    kernel = torch.eye(block_shape[0] * block_shape[1])
    kernel = kernel.reshape(kernel.shape[1], 1, block_shape[0], block_shape[1])
    x_shape = x.shape
    indices = torch.arange(x_shape[2] * x_shape[3])
    indices = indices.reshape(1, 1, x_shape[2], x_shape[3])
    indices = F.conv2d(indices.float(), kernel.float(), stride=[block_stride[0], block_stride[1]], padding='valid')
    indices = indices.permute(0, 2, 3, 1)  # move output channel to last
    dims = indices.shape[:3]
    num_blocks = functools.reduce(operator.mul, dims, 1)
    indices = indices.reshape(num_blocks, -1)
    return indices.long().to(x.device)  # PyTorch wants indices as long


def gather_blocks_2d(x, indices):
    x_shape = x.shape
    x = x.reshape(*x_shape[:2], x_shape[2] * x_shape[3], *x_shape[4:])
    x_t = x.permute([2, 0, 1, 3])
    x_new = x_t[indices]
    return x_new.permute([2, 3, 0, 1, 4])


def get_shifted_center_blocks(x, indices):
    '''
    PyTorch version
    '''
    center_x = gather_blocks_2d(x, indices)

    def shift_right_2d_blocks(x):
        return F.pad(x, [0, 0, 1, 0])[:, :, :, :-1, :]

    x_shifted = shift_right_2d_blocks(center_x)
    return x_shifted


def scatter_blocks_2d(x, indices, shape):
    x_shape = x.shape
    x_t = x.reshape(x_shape[0], x_shape[1], -1, x_shape[-1]).permute([2, 0, 1, 3])
    x_t_shape = x_t.shape
    indices = indices.reshape(-1)
    scattered_x = torch.index_add(torch.zeros_like(x_t).to(x.device), 0, indices, x_t)
    scattered_x = scattered_x.permute([1, 2, 0, 3])
    return scattered_x.reshape(shape)


def ones_matrix_band_part(rows, cols, num_lower, num_upper, out_shape=None) -> Tensor:
    '''
    Output a matrix with ones in diagonal and sub-diagonal to lower and upper triangles
    '''
    if num_lower < 0:
        num_lower = rows - 1
    if num_upper < 0:
        num_upper = cols - 1
    lower_mask = np.tri(cols, rows, num_lower).T
    upper_mask = np.tri(rows, cols, num_upper)
    band = np.ones((rows, cols)) * lower_mask * upper_mask
    if out_shape:
        band = band.reshape(out_shape)
    return torch.from_numpy(band)


def attn_bias_local(length, max_backward, max_forward):
    band = ones_matrix_band_part(length, length, max_backward, max_forward, out_shape=[1, 1, length, length])
    return -1e9 * (1.0 - band)


def attn_bias_lower_triangle(length):
    return attn_bias_local(length, -1, 0)

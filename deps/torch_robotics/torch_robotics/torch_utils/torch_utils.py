import collections
import random
from typing import List

import numpy as np
import torch


def get_torch_device(device='cuda'):
    if 'cuda' in device and torch.cuda.is_available():
        device = 'cuda'
    elif 'mps' in device:
        device = 'mps'
    else:
        device = 'cpu'
    return torch.device(device)


DEFAULT_TENSOR_ARGS = {'device': get_torch_device('cuda'), 'dtype': torch.float32}

def dict_to_device(ob, device):
    if isinstance(ob, collections.Mapping):
        return {k: dict_to_device(v, device) for k, v in ob.items()}
    else:
        return ob.to(device)


def to_numpy(x, dtype=np.float32, clone=False):
    if torch.is_tensor(x):
        x = x.detach().cpu().numpy().astype(dtype)
        return x
    if isinstance(x, np.ndarray):
        return x.astype(dtype, copy=clone)
    return np.array(x).astype(dtype)


def to_torch(x, device='cpu', dtype=torch.float, requires_grad=False, clone=False):
    if torch.is_tensor(x):
        if clone:
            x = x.clone()
        return x.to(device=device, dtype=dtype)
    return torch.tensor(x, dtype=dtype, device=device, requires_grad=requires_grad)


def to_torch_2d_min(variable):
    if isinstance(variable, np.ndarray) or isinstance(variable, List):
        tensor_var = to_torch(variable)
    else:
        tensor_var = variable
    if len(tensor_var.shape) == 1:
        return tensor_var.unsqueeze(0)
    else:
        return tensor_var


def freeze_torch_model_params(model):
    for param in model.parameters():
        param.requires_grad = False
    # If the model is frozen we do not save it again, since the parameters did not change
    model.is_frozen = True


def batch_cov(points):
    B, N, D = points.size()
    mean = points.mean(dim=1).unsqueeze(1)
    diffs = (points - mean).reshape(B * N, D)
    prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
    bcov = prods.sum(dim=1)
    if N > 1:
        bcov = bcov / (N - 1)  # Unbiased estimate
    else:
        bcov = bcov / N
    return bcov  # (B, D, D)


def batch_trace(covs):
    return covs.diagonal(offset=0, dim1=-1, dim2=-2).sum(-1)


@torch.jit.script
def tensor_linspace_v1(start: torch.Tensor, end: torch.Tensor, steps: int = 10):
    # https://github.com/zhaobozb/layout2im/blob/master/models/bilinear.py#L246
    """
    Vectorized version of torch.linspace.
    Inputs:
    - start: Tensor of any shape
    - end: Tensor of the same shape as start
    - steps: Integer
    Returns:
    - out: Tensor of shape start.size() + (steps,), such that
      out.select(-1, 0) == start, out.select(-1, -1) == end,
      and the other elements of out linearly interpolate between
      start and end.
    """
    assert start.size() == end.size()
    view_size = start.size() + (1,)
    w_size = (1,) * start.dim() + (steps,)
    out_size = start.size() + (steps,)

    start_w = torch.linspace(1, 0, steps=steps).to(start)
    start_w = start_w.view(w_size).expand(out_size)
    end_w = torch.linspace(0, 1, steps=steps).to(start)
    end_w = end_w.view(w_size).expand(out_size)

    start = start.contiguous().view(view_size).expand(out_size)
    end = end.contiguous().view(view_size).expand(out_size)

    out = start_w * start + end_w * end
    return out


@torch.jit.script
def torch_linspace_v2(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional bahaviour of numpy.linspace in PyTorch.
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=start.dtype, device=start.device) / (num - 1)

    # reshape the 'steps' tensor to [-1, *([1]*start.ndim)] to allow for broadcastings
    # - using 'steps.reshape([-1, *([1]*start.ndim)])' would be nice here but torchscript
    #   "cannot statically infer the expected size of a list in this contex", hence the code below
    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]

    return out


# @torch.jit.script
def batched_weighted_dot_prod(x: torch.Tensor, M: torch.Tensor, y: torch.Tensor, with_einsum: bool = False):
    """
    Computes batched version of weighted dot product (distance) x.T @ M @ x
    """
    assert x.ndim >= 2
    if with_einsum:
        My = M.unsqueeze(0) @ y
        r = torch.einsum('...ij,...ij->...j', x, My)
    else:
        r = x.transpose(-2, -1) @ M.unsqueeze(0) @ x
        r = r.diagonal(dim1=-2, dim2=-1)
    return r


def is_positive_semi_definite(mat):
    # checks if mat is a positive semi-definite matrix
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real >= 0).all())

def is_positive_definite(mat):
    # checks if mat is a positive definite matrix
    return bool((mat == mat.T).all() and (torch.linalg.eigvals(mat).real > 0).all())


def torch_intersect_1d(a, b):
    a_cat_b, counts = torch.cat([a, b]).unique(return_counts=True)
    intersection = a_cat_b[torch.where(counts.gt(1))]
    return intersection



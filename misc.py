import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import torchdiffeq
import numpy as np
from einops import rearrange, repeat
import time
import torch.optim as optim
import glob
import imageio
from math import pi
from random import random
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
from torchvision import datasets, transforms
import sys
from matplotlib import pyplot as plt
import pickle

# Format [time, batch, diff, vector]

tol = 1e-3


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class ArgumentParser:
    def add_argument(self, str, type, default):
        setattr(self, str[2:], default)

    def parse_args(self):
        return self


def str_rec(names, data, unit=None, sep=', ', presets='{}'):
    if unit is None:
        unit = [''] * len(names)
    out_str = "{{}}: {} {{{{}}}}" + sep
    out_str *= len(names)
    out_str = out_str.format(*data)
    out_str = out_str.format(*names)
    out_str = out_str.format(*unit)
    out_str = presets.format(out_str)
    return out_str


def to_float(arr, truncate=False):
    if isinstance(arr, list):
        return [to_float(i, truncate=truncate) for i in arr]
    if arr is None:
        return None
    if isinstance(arr, torch.Tensor):
        arr = arr.detach().cpu().numpy()
    if isinstance(arr, np.ndarray):
        arr = arr.flatten()[0]
    if truncate:
        arr = int(arr * 10 ** truncate) / 10 ** truncate
    return arr

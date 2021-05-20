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
import csv
# Format [time, batch, diff, vector]

tol = 1e-3


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def shrink_parameters(model, ratio):
    model_dict = model.state_dict()
    for i in model_dict:
        model_dict[i] *= ratio
    model.load_state_dict(model_dict)
    return model


def gradnorm(model, p=2):
    param_normp = [param.grad.data.norm(p) ** p for param in model.parameters() if param.grad is not None]
    total_normp = sum(param_normp)
    total_norm = total_normp ** (1 / p)
    return total_norm


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


class EmptyClass:
    pass


class Recorder:
    def __init__(self):
        self.store = []
        self.current = dict()

    def __setitem__(self, key, value):
        for method in ['detach', 'cpu', 'numpy']:
            if hasattr(value, method):
                value = getattr(value, method)()
        if key in self.current:
            self.current[key].append(value)
        else:
            self.current[key] = [value]

    def capture(self, verbose=False):
        for i in self.current:
            self.current[i] = np.mean(self.current[i])
        self.store.append(self.current.copy())
        self.current = dict()
        if verbose:
            for i in self.store[-1]:
                if i[0] != '_':
                    print('{}: {}'.format(i, self.store[-1][i]))
        return self.store[-1]

    def tolist(self):
        labels = set()
        labels = sorted(labels.union(*self.store))
        outlist = []
        for obs in self.store:
            outlist.append([obs.get(i, np.nan) for i in labels])
        return labels, outlist

    def writecsv(self, writer):
        labels, outlist = self.tolist()
        if isinstance(writer, str):
            outfile = open(writer, 'w')
            csvwriter = csv.writer(outfile)
            csvwriter.writerow(labels)
            csvwriter.writerows(outlist)
            outfile.close()
        else:
            csvwriter = writer
            csvwriter.writerow(labels)
            csvwriter.writerows(outlist)


class NLayerNN(nn.Module):
    def __init__(self, *args, actv=nn.ReLU()):
        super().__init__()
        self.linears = nn.ModuleList()
        for i in range(len(args)):
            self.linears.append(nn.Linear(args[i], args[i+1]))
        self.actv = actv

    def forward(self, x):
        for i in range(self.layer_cnt):
            x = self.linears[i](x)
            if i < self.layer_cnt - 1:
                x = self.actv(x)
        return x

    @property
    def layer_cnt(self):
        return len(self.linears)
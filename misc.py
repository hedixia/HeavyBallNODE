import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from einops import rearrange

tol = 1e-3
gpu = 0
niters = 10
lr = 1e-2
# Formet [time, batch, diff, vector]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

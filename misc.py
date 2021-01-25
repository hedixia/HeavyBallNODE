import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
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
    out_str = "{}: {{}} {{{{}}}}" + sep
    out_str *= len(names)
    out_str = out_str.format(*names)
    out_str = out_str.format(*data)
    out_str = out_str.format(*unit)
    out_str = presets.format(out_str)
    return out_str


rec_names = ["iter", "loss", "nfe", "time/iter", "time"]
rec_unit = ["", "", "", "s", "min"]


def train(model, optimizer, trdat, tsdat, args, evalfreq=2, stdout=sys.stdout, **extraprint):
    defaultout = sys.stdout
    sys.stdout = stdout
    print("==> Train model {}, params {}".format(type(model), count_parameters(model)))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("==> Use accelerator: ", device)
    epoch = 0
    itrcnt = 0
    loss_func = nn.CrossEntropyLoss()
    itr_arr = np.zeros(args.niters)
    loss_arr = np.zeros(args.niters)
    nfe_arr = np.zeros(args.niters)
    time_arr = np.zeros(args.niters)
    acc_arr = np.zeros(args.niters)

    # training
    start_time = time.time()
    while epoch < args.niters:
        epoch += 1
        iter_start_time = time.time()
        for x, y in trdat:
            itrcnt += 1
            model[1].df.nfe = 0
            optimizer.zero_grad()
            # forward in time and solve ode
            pred_y = model(x.to(device=args.gpu))
            # compute loss
            loss = loss_func(pred_y, y.to(device=args.gpu))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            # make arrays
            itr_arr[epoch - 1] = epoch
            loss_arr[epoch - 1] += loss
            nfe_arr[epoch - 1] += model[1].df.nfe
        iter_end_time = time.time()
        time_arr[epoch - 1] = iter_end_time - iter_start_time
        loss_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        nfe_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        printouts = [epoch, loss_arr[epoch - 1], nfe_arr[epoch - 1], time_arr[epoch - 1],
                     (time.time() - start_time) / 60]
        print(str_rec(rec_names, printouts, rec_unit, presets="Train|| {}"))
        print('Extra: ', extraprint)
        try:
            print(torch.sigmoid(model[1].df.gamma))
        except Exception:
            pass
        if epoch % evalfreq == 0:
            model[1].df.nfe = 0
            end_time = time.time()
            loss = 0
            acc = 0
            dsize = 0
            bcnt = 0
            for x, y in tsdat:
                # forward in time and solve ode
                dsize += y.shape[0]
                y = y.to(device=args.gpu)
                pred_y = model(x.to(device=args.gpu))
                pred_l = torch.argmax(pred_y, dim=1)
                acc += torch.sum((pred_l == y).float())
                bcnt += 1
                # compute loss
                loss += loss_func(pred_y, y) * y.shape[0]

            loss /= dsize
            acc /= dsize
            printouts = [epoch, loss.detach().cpu(), acc.detach().cpu(), str(model[1].df.nfe / bcnt),
                         str(count_parameters(model))]
            names = ["iter", "loss", "acc", "nfe", "param cnt"]
            print(str_rec(names, printouts, presets="Test|| {}"))
            acc_arr[epoch-1] = acc.detach().cpu().numpy()
    sys.stdout = defaultout
    return itr_arr, loss_arr, nfe_arr, time_arr, acc_arr

import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
import numpy as np
from einops import rearrange, repeat
import time
import torch.optim as optim
import glob
import imageio
import numpy as np
import torch
from math import pi
from random import random
from torch.utils.data import Dataset, DataLoader
from torch.distributions import Normal
from torchvision import datasets, transforms
import argparse

import utils
import models

class DF(nn.Module):

    def __init__(self, in_channels, nhidden, args=None):
        super(DF, self).__init__()
        self.args = args
        if self.args.model == 'hbnode':
            in_dim = in_channels
        if self.args.model == 'anode':
            in_dim = in_channels
        if self.args.model == 'sonode':
            in_dim =  2*in_channels
        if self.args.model == 'node':
            in_dim = in_channels
        self.activation = nn.ReLU() #nn.LeakyReLU(0.3)
        self.fc1 = nn.Conv2d(in_dim + 1, nhidden, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(nhidden + 1, nhidden, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhidden + 1, in_channels, kernel_size=1, padding=0)


    def forward(self, t, x0):
        if self.args.model == 'hbnode':
            out = rearrange(x0, 'b 1 c x y -> b c x y')
        if self.args.model == 'anode':
            out = rearrange(x0, 'b d c x y -> b (d c) x y')
        if self.args.model == 'node':
            out = rearrange(x0, 'b d c x y -> b (d c) x y')
        if self.args.model == 'sonode':
            out = rearrange(x0, 'b d c x y -> b (d c) x y')
        t_img = torch.ones_like(out[:, :1, :, :]).to(device=self.args.gpu) * t
        out = torch.cat([out, t_img], dim=1)

        out = self.fc1(out)
        out = self.activation(out)
        out = torch.cat([out, t_img], dim=1)

        out = self.fc2(out)
        out = self.activation(out)
        out = torch.cat([out, t_img], dim=1)

        out = self.fc3(out)
        out = rearrange(out, 'b c x y -> b 1 c x y')
        if self.args.model == 'hbnode':
            return out + self.args.xres * x0
        elif self.args.model == 'anode':
            return out
        elif self.args.model == 'sonode':
            return out

class NODEintegrate(nn.Module):

    def __init__(self, df=None, x0=None):
        """
        Create an OdeRnnBase model
            x' = df(x)
            x(t0) = x0
        :param df: a function that computes derivative. input & output shape [batch, channel, feature]
        :param x0: initial condition.
            - if x0 is set to be nn.parameter then it can be trained.
            - if x0 is set to be nn.Module then it can be computed through some network.
        """
        super().__init__()
        self.df = df
        self.x0 = x0

    def forward(self, initial_condition, evaluation_times, x0stats=None):
        """
        Evaluate odefunc at given evaluation time
        :param initial_condition: shape [batch, channel, feature]. Set to None while training.
        :param evaluation_times: time stamps where method evaluates, shape [time]
        :param x0stats: statistics to compute x0 when self.x0 is a nn.Module, shape required by self.x0
        :return: prediction by ode at evaluation_times, shape [time, batch, channel, feature]
        """
        if initial_condition is None:
            initial_condition = self.x0
        if x0stats is not None:
            initial_condition = self.x0(x0stats)
        out = odeint(self.df, initial_condition, evaluation_times, rtol=args.tol, atol=args.tol)
        return out

    @property
    def nfe(self):
        return self.df.nfe

class NODElayer(nn.Module):
    def __init__(self, df, args, evaluation_times=(0.0, 1.0)):
        super(NODElayer, self).__init__()
        self.df = df
        self.evaluation_times = torch.as_tensor(evaluation_times)
        self.args = args

    def forward(self, x0):
        out = odeint(self.df, x0, self.evaluation_times, rtol=self.args.tol, atol=self.args.tol)
        return out[1]

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.evaluation_times.to(device)

class NODE(nn.Module):
    def __init__(self, df=None, **kwargs):
        super(NODE, self).__init__()
        self.__dict__.update(kwargs)
        self.df = df
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        return self.df(t, x)


class SONODE(NODE):
    def forward(self, t, x):
        """
        Compute [y y']' = [y' y''] = [y' df(t, y, y')]
        :param t: time, shape [1]
        :param x: [y y'], shape [batch, 2, vec]
        :return: [y y']', shape [batch, 2, vec]
        """
        self.nfe += 1
        v = x[:, 1:, :]
        out = self.df(t, x)
        return torch.cat((v, out), dim=1)


class HeavyBallNODE(NODE):
    def __init__(self, df, gamma=None, thetaact=None, gammaact='sigmoid', timescale=1):
        super().__init__(df)
        self.gamma = nn.Parameter(torch.Tensor([-3.0])) if gamma is None else gamma
        self.gammaact = nn.Sigmoid() if gammaact == 'sigmoid' else gammaact
        self.timescale = timescale
        self.thetaact = nn.Identity() if thetaact is None else thetaact

    def forward(self, t, x):
        """
        Compute [theta' m' v'] with heavy ball parametrization in
        $$ theta' = -m / sqrt(v + eps) $$
        $$ m' = h f'(theta) - rm $$
        $$ v' = p (f'(theta))^2 - qv $$
        https://www.jmlr.org/papers/volume21/18-808/18-808.pdf
        because v is constant, we change c -> 1/sqrt(v)
        c has to be positive
        :param t: time, shape [1]
        :param x: [theta m v], shape [batch, 3, dim]
        :return: [theta' m' v'], shape [batch, 3, dim]
        """
        self.nfe += 1
        theta, m = torch.split(x, 1, dim=1)
        dtheta = self.thetaact(-m)
        dm = self.df(t, theta) - self.timescale * torch.sigmoid(self.gamma) * m
        return torch.cat((dtheta, dm), dim=1)

class initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels, nhidden):
        super(initial_velocity, self).__init__()
        assert (3 * out_channels >= in_channels)
        self.actv = nn.LeakyReLU(0.3)
        self.fc1 = nn.Conv2d(in_channels, nhidden, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(nhidden, nhidden, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhidden, 2 * out_channels - in_channels, kernel_size=1, padding=0)
        self.out_channels = out_channels
        self.in_channels = in_channels

    def forward(self, x0):
        x0 = x0.float()
        out = self.fc1(x0)
        out = self.actv(out)
        out = self.fc2(out)
        out = self.actv(out)
        out = self.fc3(out)
        out = torch.cat([x0, out], dim=1)
        out = rearrange(out, 'b (d c) ... -> b d c ...', d=2)
        return out

class anode_initial_velocity(nn.Module):

    def __init__(self, in_channels, aug, args):
        super(anode_initial_velocity, self).__init__()
        self.args = args
        self.aug = aug
        self.in_channels = in_channels

    def forward(self, x0):
        x0 = rearrange(x0.float(), 'b c x y -> b 1 c x y')
        outshape = list(x0.shape)
        outshape[2] = self.aug
        out = torch.zeros(outshape).to(self.args.gpu)
        out[:, :, :3] += x0
        return out

class predictionlayer(nn.Module):
    def __init__(self, in_channels):
        super(predictionlayer, self).__init__()
        self.dense = nn.Linear(in_channels * 32 * 32, 10)
        #self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = rearrange(x[:,0], 'b c x y -> b (c x y)')
        #x = self.dropout(x)
        x = self.dense(x)
        return x
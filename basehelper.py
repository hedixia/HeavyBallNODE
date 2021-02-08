from misc import *


class Zeronet(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class TVnorm(nn.Module):
    def forward(self, x, v):
        return torch.norm(v, 1)


class NormAct(nn.Module):
    def __init__(self, bound):
        super().__init__()
        self.bound = bound
        self.relu = nn.ReLU()
        self.elu = nn.ELU()

    def forward(self, x):
        x = x - self.bound + 1
        x = self.relu(x) * self.elu(-x) + 1
        return x

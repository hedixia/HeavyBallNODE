from misc import *


class Zeronet(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)

zeronet = Zeronet()

class TVnorm(nn.Module):
    def __init__(self):
        super(TVnorm, self).__init__()
        self.osize = 1

    def forward(self, t, x, v):
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


class Parameter(nn.Module):
    def __init__(self, val, frozen=False):
        super().__init__()
        val = torch.Tensor(val)
        self.val = val
        self.param = nn.Parameter(val)
        self.frozen = frozen

    def forward(self):
        if self.frozen:
            self.val = self.val.to(self.param.device)
            return self.val
        else:
            return self.param

    def freeze(self):
        self.val = self.param.detach().clone()
        self.frozen = True

    def unfreeze(self):
        self.frozen = False

    def __repr__(self):
        return "val: {}, param: {}".format(self.val.cpu(), self.param.detach().cpu())
from base import *
from anode_data_loader import mnist
from einops.layers.torch import Rearrange

parser = ArgumentParser()
parser.add_argument('--network', type=str, default='odenet')
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--downsampling-method', type=str, default='conv')
parser.add_argument('--nepochs', type=int, default=120)
parser.add_argument('--data_aug', type=eval, default=True)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--test_batch_size', type=int, default=1000)
parser.add_argument('--niters', type=int, default=120)
parser.add_argument('--save', type=str, default='./experiment_sonode_conv_v1')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def norm(dim):
    return nn.GroupNorm(min(32, dim), dim)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.norm1 = norm(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.norm2 = norm(planes)
        self.conv2 = conv3x3(planes, planes)

    def forward(self, x):
        shortcut = x

        out = self.relu(self.norm1(x))

        if self.downsample is not None:
            shortcut = self.downsample(out)

        out = self.conv1(out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(out)

        return out + shortcut


class ConcatConv2d(nn.Module):

    def __init__(self, dim_in, dim_out, ksize=3, stride=1, padding=0, dilation=1, groups=1, bias=True, transpose=False):
        super(ConcatConv2d, self).__init__()
        module = nn.ConvTranspose2d if transpose else nn.Conv2d
        self._layer = module(
            dim_in + 1, dim_out, kernel_size=ksize, stride=stride, padding=padding, dilation=dilation, groups=groups,
            bias=bias
        )

    def forward(self, t, x):
        tt = torch.ones_like(x[:, :1, :, :]) * t
        ttx = torch.cat([tt, x], 1)
        return self._layer(ttx)


class initial_velocity(nn.Module):

    def __init__(self, dim):
        super(initial_velocity, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)

    def forward(self, x0):
        out = self.norm1(x0) * 0
        out = self.relu(out)
        out = self.conv1(0, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(0, out)
        out = self.norm3(out)
        return torch.stack((x0, out), dim=1)


class DF(nn.Module):

    def __init__(self, dim):
        super(DF, self).__init__()
        self.norm1 = norm(dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm2 = norm(dim)
        self.conv2 = ConcatConv2d(dim, dim, 3, 1, 1)
        self.norm3 = norm(dim)
        self.nfe = 0

    def forward(self, t, x):
        xshape = list(x.shape)
        out = self.norm1(x.view(-1, *xshape[2:]))
        out = self.relu(out)
        out = self.conv1(t, out)
        out = self.norm2(out)
        out = self.relu(out)
        out = self.conv2(t, out)
        out = self.norm3(out)
        return out.view(xshape)


class predictionlayer(nn.Module):
    def __init__(self, in_channels):
        super(predictionlayer, self).__init__()
        self.dense = nn.Linear(in_channels * 28 * 28, 10)

    def forward(self, x):
        x = rearrange(x[:, 0], 'b c x y -> b (c x y)')
        x = self.dense(x)
        return x


class Flatten(nn.Module):

    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        shape = torch.prod(torch.tensor(x.shape[1:])).item()
        return x.view(-1, shape)


class getx(nn.Module):
    def forward(self, x):
        return x[:, 0]


downsampling_layers = [
    nn.Conv2d(1, 64, 3, 1),
    norm(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, 4, 2, 1),
    norm(64),
    nn.ReLU(inplace=True),
    nn.Conv2d(64, 64, 4, 2, 1), ]

feature_layers = [initial_velocity(64), NODElayer(HeavyBallNODE(DF(64), None))]
fc_layers = [getx(), norm(64), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1)), Flatten(), nn.Linear(64, 10)]

model = nn.Sequential(*downsampling_layers, *feature_layers, *fc_layers).to(0)
model_dict = model.state_dict()
model_dict['8.df.gamma'] += 3
#for i in model_dict:
    #model_dict[i] *= 0.01
    #model_dict[i] -= 0.2
model.load_state_dict(model_dict)
trdat, tsdat = mnist(batch_size=128)
optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.000)
model[1].df = model[8].df
lrscheduler = torch.optim.lr_scheduler.StepLR(optimizer, 40, 0.1)
train(model, optimizer, trdat, tsdat, args, 1, lrscheduler=lrscheduler)

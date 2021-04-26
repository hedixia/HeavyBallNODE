from os import path
import sys

sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
import argparse
import csv

from anode_data_loader import mnist
from base import *
from mnist.mnist_train import train

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=40)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


# shape: [time, batch, derivatives, channel, x, y]


class anode_initial_velocity(nn.Module):

    def __init__(self, in_channels, aug, dch=1):
        super(anode_initial_velocity, self).__init__()
        self.aug = aug
        self.in_channels = in_channels
        self.dch = dch

    def forward(self, x0):
        outshape = list(x0.shape)
        outshape[1] = self.aug * self.dch
        out = torch.zeros(outshape).to(args.gpu)
        out[:, :1] += x0
        out = rearrange(out, 'b (d c) ... -> b d c ...', d=self.dch)
        return out


class hbnode_initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels, nhid):
        super(hbnode_initial_velocity, self).__init__()
        assert (3 * out_channels >= in_channels)
        self.actv = nn.LeakyReLU(0.3)
        self.fc1 = nn.Conv2d(in_channels, nhid, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(nhid, nhid, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhid, 2 * out_channels - in_channels, kernel_size=1, padding=0)
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


class DF(nn.Module):

    def __init__(self, in_channels, nhid, out_channels=None):
        super(DF, self).__init__()
        if out_channels is None:
            out_channels = in_channels
        self.activation = nn.ReLU(inplace=True)
        self.fc1 = nn.Conv2d(in_channels + 1, nhid, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(nhid + 1, nhid, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhid + 1, out_channels, kernel_size=1, padding=0)

    def forward(self, t, x0):
        x0 = rearrange(x0, 'b d c x y -> b (d c) x y')
        t_img = torch.ones_like(x0[:, :1, :, :]).to(device=args.gpu) * t
        out = torch.cat([x0, t_img], dim=1)
        out = self.fc1(out)
        out = self.activation(out)
        out = torch.cat([out, t_img], dim=1)
        out = self.fc2(out)
        out = self.activation(out)
        out = torch.cat([out, t_img], dim=1)
        out = self.fc3(out)
        out = rearrange(out, 'b c x y -> b 1 c x y')
        return out


class predictionlayer(nn.Module):
    def __init__(self, in_channels, truncate=False, dropout=0.0):
        super(predictionlayer, self).__init__()
        self.dense = nn.Linear(in_channels * 28 * 28, 10)
        self.truncate = truncate
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        if self.truncate:
            x = rearrange(x[:, 0], 'b ... -> b (...)')
        else:
            x = rearrange(x, 'b ... -> b (...)')
        x = self.dropout(x)
        x = self.dense(x)
        return x


class tv4node(nn.Module):
    osize = 1

    def forward(self, t, x, v):
        return torch.norm(v.reshape(v.shape[0], -1), p=2, dim=1)


class tvSequential(nn.Sequential):
    def __init__(self, ic, layer, predict):
        super(tvSequential, self).__init__(ic, layer, predict)
        self.ic = ic
        self.layer = layer
        self.predict = predict

    def forward(self, x):
        x = self.ic(x)
        x, rec = self.layer(x)
        out = self.predict(x)
        return out, rec


trdat, tsdat = mnist()


def model_gen(name):
    if name == 'node':
        dim = 1
        nhid = 92
        layer = NODElayer(NODE(DF(dim, nhid)))
        model = nn.Sequential(anode_initial_velocity(1, dim),
                              layer, predictionlayer(dim))
    elif name == 'anode':
        dim = 6
        nhid = 64
        layer = NODElayer(NODE(DF(dim, nhid)))
        model = nn.Sequential(anode_initial_velocity(1, dim),
                              layer, predictionlayer(dim))
    elif name == 'sonode-':
        dim = 1
        nhid = 65
        hblayer = NODElayer(SONODE(DF(2 * dim, nhid, dim)))
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              hblayer, predictionlayer(dim, truncate=True)).to(device=args.gpu)
    elif name == 'sonode':
        dim = 5
        nhid = 50
        hblayer = NODElayer(SONODE(DF(2 * dim, nhid, dim)))
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              hblayer, predictionlayer(dim, truncate=True)).to(device=args.gpu)
    elif name == 'hbnode':
        dim = 5
        nhid = 50
        layer = NODElayer(HeavyBallNODE(DF(dim, nhid), None))
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              layer, predictionlayer(dim, truncate=True)).to(device=args.gpu)
    elif name == 'ghbnode':
        dim = 6
        nhid = 45
        layer = NODElayer(HeavyBallNODE(DF(dim, nhid), actv_h=nn.Tanh(), corr=2.0, corrf=False))
        model = nn.Sequential(hbnode_initial_velocity(1, dim, nhid),
                              layer, predictionlayer(dim, truncate=True)).to(device=args.gpu)
    elif name == 'avnode':
        dim = 6
        nhid = 64
        layer = NODElayer(NODE(DF(dim, nhid)), shape=(1, 6, 28, 28), recf=tv4node())
        model = tvSequential(anode_initial_velocity(1, dim),
                             layer, predictionlayer(dim))
    elif name == 'areg':
        dim = 6
        nhid = 64
        layer = NODElayer(NODE(DF(dim, nhid)))
        model = nn.Sequential(anode_initial_velocity(1, dim),
                              layer, predictionlayer(dim))
    else:
        print('model {} not supported.'.format(name))
        model = None
    return model.to(args.gpu)


#names = ['node', 'anode', 'sonode', 'hbnode', 'ghbnode']
names = ['ghbnode']
rec_names = ["model", "test#", "train/test", "iter", "loss", "acc", "forwardnfe", "backwardnfe", "time/iter",
             "time_elapsed"]
csvfile = open('../imgdat/outdat0.csv', 'w')
writer = csv.writer(csvfile)
writer.writerow(rec_names)
csvfile.close()

dat = []
for name in names:
    runnum = name[:3]
    log = open('../output/mnist/log_{}.txt'.format(runnum), 'w')
    datfile = open('../output/mnist/mnist_dat_{}.txt'.format(runnum), 'wb')
    for i in range(3):
        model = model_gen(name)
        print(name, count_parameters(model), *[count_parameters(i) for i in model])
        optimizer = optim.Adam(model.parameters(), lr=args.lr / 2, weight_decay=0.000)
        lrscheduler = torch.optim.lr_scheduler.StepLR(optimizer, 200, 0.9)
        # train_out = train(model, optimizer, trdat, tsdat, args, evalfreq=1)
        train_out = train(model, optimizer, trdat, tsdat, args, name, i, evalfreq=1, csvname='../imgdat/outdat0.csv')
        dat.append([name, i, train_out])
        log.writelines(['\n'] * 5)
    pickle.dump(dat, datfile)
    log.close()
    datfile.close()

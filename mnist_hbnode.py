import argparse
import time

import torch.optim as optim

from anode_data_loader import mnist
from base import *

parser = ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=40)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# shape: [time, batch, derivatives, channel, x, y]
dim = 5
hidden = 49
trdat, tsdat = mnist(batch_size=256)


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
        # out = repeat(torch.zeros_like(x0[:,:1]), 'b 1 ... -> b d c ...', d=2, c=self.out_channels)
        # out[:, 0, :self.in_channels] = x0

        # """
        out = self.fc1(x0)
        out = self.actv(out)
        out = self.fc2(out)
        out = self.actv(out)
        out = self.fc3(out)
        out = torch.cat([x0, out], dim=1)
        out = rearrange(out, 'b (d c) ... -> b d c ...', d=2)
        # """
        return out


class DF(nn.Module):

    def __init__(self, in_channels, nhidden):
        super(DF, self).__init__()
        self.activation = nn.LeakyReLU(0.01)
        self.fc1 = nn.Conv2d(in_channels + 1, nhidden, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(nhidden + 1, nhidden, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhidden + 1, in_channels, kernel_size=1, padding=0)

    def forward(self, t, x0):
        out = rearrange(x0, 'b 1 c x y -> b c x y')
        t_img = torch.ones_like(out[:, :1, :, :]).to(device=args.gpu) * t
        out = torch.cat([out, t_img], dim=1)
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
    def __init__(self, in_channels):
        super(predictionlayer, self).__init__()
        self.dense = nn.Linear(in_channels * 28 * 28, 10)

    def forward(self, x):
        x = rearrange(x[:, 0], 'b c x y -> b (c x y)')
        x = self.dense(x)
        return x


def train(model, optimizer, trdat, tsdat):
    epoch = 0
    itrcnt = 0
    loss_func = nn.CrossEntropyLoss()
    itr_arr = np.zeros(args.niters)
    loss_arr = np.zeros(args.niters)
    nfe_arr = np.zeros(args.niters)
    time_arr = np.zeros(args.niters)

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
        try:
            print(torch.sigmoid(model[1].df.gamma))
        except Exception:
            pass
        if epoch % 2 == 0:
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("==> Use accelerator: ", device)
hblayer = NODElayer(HeavyBallODE(DF(dim, hidden), None))
model = nn.Sequential(initial_velocity(1, dim, hidden), hblayer, predictionlayer(dim)).to(device=args.gpu)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.000)
print(count_parameters(model))
train(model, optimizer, trdat, tsdat)

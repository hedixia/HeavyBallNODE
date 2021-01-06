import argparse
import time

import torch.optim as optim

from anode_data_loader import mnist
from base import *

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-2)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=20)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# shape: [time, batch, derivatives, channel, x, y]
dim = 1
hidden = 30
trdat, tsdat = mnist(batch_size=256)


class initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels, nhidden):
        super(initial_velocity, self).__init__()
        assert (out_channels >= in_channels)
        self.actv = nn.LeakyReLU(0.3)
        self.fc1 = nn.Conv2d(in_channels, nhidden, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(nhidden, nhidden, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhidden, 3 * out_channels - in_channels, kernel_size=1, padding=0)

    def forward(self, x0):
        x0 = x0.float()
        out = self.fc1(x0)
        out = self.actv(out)
        out = self.fc2(out)
        out = self.actv(out)
        out = self.fc3(out)
        out = torch.cat([x0, out], dim=1)
        out = rearrange(out, 'b (d c) x y -> b d c x y', d=3)
        return out


class DF(nn.Module):

    def __init__(self, in_channels, nhidden):
        super(DF, self).__init__()
        self.activation = nn.LeakyReLU(0.3)
        self.fc1 = nn.Conv2d(in_channels + 1, nhidden, kernel_size=1, padding=0)
        self.fc2 = nn.Conv2d(nhidden + 1, nhidden, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhidden + 1, in_channels, kernel_size=1, padding=0)

    def forward(self, t, x0):
        out = rearrange(x0, 'b 1 c x y -> b c x y')
        t_img = torch.ones_like(x[:, :1, :, :]) * t
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
        self.dense = nn.Linear(3 * in_channels * 28 * 28, 10)

    def forward(self, x):
        x = rearrange(x, 'b d c x y -> b (d c x y)')
        x = self.dense(x)
        return x


gamma = 0.0  # nn.Parameter(torch.tensor([0.0]))
hblayer = NODElayer(HeavyBallODE(DF(dim, dim), gamma))
model = nn.Sequential(initial_velocity(dim, dim, hidden), hblayer, predictionlayer(dim))#.to(device=args.gpu)

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00)
loss_func = nn.CrossEntropyLoss()

itr_arr = np.empty(args.niters)
loss_arr = np.empty(args.niters)
nfe_arr = np.empty(args.niters)
time_arr = np.empty(args.niters)

itrcnt = 0
# training
start_time = time.time()
min_loss = 1.0  # set arbitrary loss
labeldefault = repeat(torch.arange(10), 'c -> b c', b=256)
epoch = 0
while epoch < args.niters:
    epoch += 1
    iter_start_time = time.time()
    for x, y in trdat:
        itrcnt += 1
        model[1].df.nfe = 0
        optimizer.zero_grad()
        # forward in time and solve ode
        pred_y = model(x)

        # compute loss
        loss = loss_func(pred_y, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        # make arrays
        itr_arr[epoch - 1] = epoch
        loss_arr[epoch - 1] += loss
        nfe_arr[epoch - 1] += model[1].df.nfe
        if itrcnt % 100 == 0:
            print(itrcnt, (time.time() - start_time) / 60, loss, model[1].df.nfe)
    iter_end_time = time.time()
    time_arr[epoch - 1] = iter_end_time - iter_start_time
    loss_arr[epoch - 1] /= (itrcnt // epoch)
    nfe_arr[epoch - 1] /= (itrcnt // epoch)
    print('Iter: {}, running loss: {:.4f}, nfe: {}'.format(epoch, loss_arr[epoch - 1], nfe_arr[epoch - 1]))
    if epoch % 5 == 0:
        model[1].df.nfe = 0
        end_time = time.time()
        print('\n')
        print('Training complete after {} iters.'.format(epoch))
        print('Time = ' + str((end_time - start_time) / 3600), 'h')
        loss = 0
        acc = 0
        dsize = 0
        bcnt = 0
        for x, y in tsdat:
            # forward in time and solve ode
            dsize += y.shape[0]
            pred_y = model(x)
            pred_l = torch.argmax(pred_y, dim=1)
            acc += torch.sum((pred_l == y).float())
            bcnt += 1
            # compute loss
            loss += loss_func(pred_y, y) * y.shape[0]

        loss /= dsize
        acc /= dsize
        print('Test loss = ' + str(loss))
        print('Test acc = ' + str(acc))
        print('NFE = ' + str(model[1].df.nfe / bcnt))
        print('Parameters = ' + str(count_parameters(model)))

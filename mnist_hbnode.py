import argparse
import time

import torch.optim as optim

from anode_data_loader import mnist
from base import *

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=1)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# shape: [time, batch, derivatives, channel, x, y]
dim = 1
hidden = 64
trdat, tsdat = mnist()



class initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels, nhidden):
        super(initial_velocity, self).__init__()
        assert (out_channels >= in_channels)
        self.tanh = nn.Hardtanh(min_val=-5, max_val=5, inplace=False)
        self.fc1 = nn.Conv2d(in_channels, nhidden, kernel_size=3, padding=1)
        self.fc2 = nn.Conv2d(nhidden, nhidden, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhidden, 3 * out_channels - in_channels, kernel_size=3, padding=1)

    def forward(self, x0):
        x0 = x0.float()
        out = self.fc1(x0)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = torch.cat([x0, out], dim=1)
        out = rearrange(out, 'b (d c) x y -> b d c x y', d=3)
        return out


class DF(nn.Module):

    def __init__(self, in_channels, nhidden):
        super(DF, self).__init__()
        self.tanh = nn.Hardtanh(min_val=-5, max_val=5, inplace=False)
        self.fc1 = nn.Conv2d(in_channels, nhidden, kernel_size=3, padding=1)
        self.fc2 = nn.Conv2d(nhidden, nhidden, kernel_size=3, padding=1)
        self.fc3 = nn.Conv2d(nhidden, in_channels, kernel_size=3, padding=1)

    def forward(self, t, x0):
        x0 = rearrange(x0, 'b 1 c x y -> b c x y')
        out = self.fc1(x0)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = rearrange(out, 'b c x y -> b 1 c x y')
        return out


class predictionlayer(nn.Module):
    def __init__(self, in_channels):
        super(predictionlayer, self).__init__()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv1 = nn.Conv2d(3 * in_channels, 3 * in_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(3 * in_channels, 3 * in_channels, kernel_size=3)
        self.dense = nn.Linear(27 * in_channels, 10)

    def forward(self, x):
        x = rearrange(x, 'b d c x y -> b (d c) x y')
        x = self.pool(x)
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = rearrange(x, 'b (d c) x y -> b (d c x y)', d=3)
        x = self.dense(x)
        return x



gamma = nn.Parameter(torch.tensor([0.0]))
hblayer = NODElayer(HeavyBallODE(DF(dim, dim), gamma))
model = nn.Sequential(initial_velocity(dim, dim, hidden), hblayer, predictionlayer(dim), nn.Softmax(dim=1))


optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.01)
loss_func = nn.MSELoss()

itr_arr = np.empty(args.niters)
loss_arr = np.empty(args.niters)
nfe_arr = np.empty(args.niters)
time_arr = np.empty(args.niters)

itrcnt = 0
# training
start_time = time.time()
min_loss = 1.0  # set arbitrary loss
labeldefault = repeat(torch.arange(10), 'c -> b c', b=64)
epoch = 0
while epoch <  args.niters:
    epoch += 1
    for x, y in trdat:
        if itrcnt %100 == 0:
            print(itrcnt, (time.time()-start_time)/60)
        itrcnt += 1
        model[1].df.nfe = 0
        iter_start_time = time.time()
        optimizer.zero_grad()

        # forward in time and solve ode
        pred_y = model(x)
        y = (repeat(y, 'b -> b c', c=10) == labeldefault[:y.shape[0]]).float()
        # compute loss
        loss = loss_func(pred_y, y)
        loss.backward()
        optimizer.step()
        iter_end_time = time.time()
        # make arrays
        itr_arr[epoch - 1] = epoch
        loss_arr[epoch - 1] = loss
        nfe_arr[epoch - 1] = model[1].df.nfe
        time_arr[epoch - 1] = iter_end_time - iter_start_time

    print('Iter: {}, running MSE: {:.4f}, nfe: {}'.format(epoch, loss, model[1].df.nfe))

end_time = time.time()
print('\n')
print('Training complete after {} iters.'.format(epoch))
print('Time = ' + str(end_time - start_time))
loss = 0
acc = 0
n = 0
for x, y in tsdat:
    # forward in time and solve ode
    pred_y = model(x)
    pred_l = torch.argmax(pred_y, dim=1)
    acc += torch.mean((pred_l == y).float())
    y = (repeat(y, 'b -> b c', c=10) == labeldefault[:y.shape[0]]).float()
    # compute loss
    loss += loss_func(pred_y, y) * 10
    n += 1
loss /= n
acc /= n
print('Test MSE = ' + str(loss))
print('Test acc = ' + str(acc))
print('NFE = ' + str(model[1].df.nfe))
print('Parameters = ' + str(count_parameters(model)))

from base import *
import argparse
import time

import matplotlib.pyplot as plt
import torch.optim as optim

from base import *
from anode_data_loader import ConcentricSphere

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

dim = 30
aug = 0
dataset = ConcentricSphere(dim, (0, 0.5), (1, 1.5), 500, 1500)
x = torch.stack(dataset.data).reshape(-1, 1, dim)
y = torch.stack(dataset.targets)
print(x.shape, y.shape)


class initial_velocity(nn.Module):

    def __init__(self, dim, nhidden, augmentation=0):
        super(initial_velocity, self).__init__()
        self.tanh = nn.Hardtanh(min_val=-5, max_val=5, inplace=False)
        self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, 2 * dim+3*augmentation)

    def forward(self, x0):
        out = self.fc1(x0)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = torch.cat([x0, out], dim=-1)
        out = rearrange(out, 'b 1 (c v) -> b c v', c=3)
        return out


class DF(nn.Module):

    def __init__(self, dim, nhidden):
        super(DF, self).__init__()
        self.elu = nn.LeakyReLU(0.3, inplace=False)
        self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, dim)
        self.nfe = 0

    def forward(self, t, x):
        out = self.fc1(x)
        out = self.elu(out)
        out = self.fc2(out)
        out = self.elu(out)
        out = self.fc3(out)
        return out


class predictionlayer(nn.Module):
    def __init__(self):
        super(predictionlayer, self).__init__()
        self.dense = nn.Linear(3 * (dim+aug), 1)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.dense(x)
        return x

gamma = nn.Parameter(torch.tensor([0.0]))
hblayer = NODElayer(HeavyBallNODE(DF(dim + aug, 32), gamma))
model = nn.Sequential(initial_velocity(dim, 32, aug), hblayer, predictionlayer(), nn.Tanh())
print(model(x).shape)


optimizer = optim.Adam(model.parameters(), lr=args.lr)
loss_func = nn.MSELoss()

itr_arr = np.empty(args.niters)
loss_arr = np.empty(args.niters)
nfe_arr = np.empty(args.niters)
time_arr = np.empty(args.niters)

# training
start_time = time.time()
min_loss = 1.0  # set arbitrary loss
for itr in range(1, args.niters + 1):
    model[1].df.nfe = 0
    iter_start_time = time.time()
    optimizer.zero_grad()

    # forward in time and solve ode
    pred_y = model(x)

    # compute loss
    loss = loss_func(pred_y, y)
    loss.backward()
    optimizer.step()
    iter_end_time = time.time()
    # make arrays
    itr_arr[itr - 1] = itr
    loss_arr[itr - 1] = loss
    nfe_arr[itr - 1] = model[1].df.nfe
    time_arr[itr - 1] = iter_end_time - iter_start_time

    print('Iter: {}, running MSE: {:.4f}, nfe: {}'.format(itr, loss, model[1].df.nfe))

end_time = time.time()
print('\n')
print('Training complete after {} iters.'.format(itr))
print('Time = ' + str(end_time - start_time))
loss = loss_func(pred_y, y).detach().numpy()
print('Train MSE = ' + str(loss))
print('NFE = ' + str(model[1].df.nfe))
print('Parameters = ' + str(count_parameters(model)))

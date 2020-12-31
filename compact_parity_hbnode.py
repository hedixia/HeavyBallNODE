import argparse
import time

import matplotlib.pyplot as plt
import torch.optim as optim

from base import *

parser = argparse.ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-2)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()


class initial_velocity(nn.Module):

    def __init__(self, dim, nhidden):
        super(initial_velocity, self).__init__()
        self.tanh = nn.Hardtanh(min_val=-5, max_val=5, inplace=False)
        self.fc1 = nn.Linear(dim, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, 2 * dim)

    def forward(self, x0):
        out = self.fc1(x0)
        out = self.tanh(out)
        out = self.fc2(out)
        out = self.tanh(out)
        out = self.fc3(out)
        out = self.tanh(out)
        out = rearrange(out, 'b 1 (c v) -> b c v', c=2)
        return torch.cat((x0, out), dim=1)


class DF(nn.Module):

    def __init__(self, dim, nhidden):
        super(DF, self).__init__()
        self.elu = nn.ELU(inplace=False)
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


if __name__ == '__main__':
    device = torch.device('cuda:' + str(args.gpu) if torch.cuda.is_available() else 'cpu')

    # make data
    n = 5
    z0 = torch.tensor([1.0, -1.0]).reshape(-1, 1, 1).float().to(device)
    zN = torch.tensor([-1.0, 1.0]).reshape(-1, 1, 1).float().to(device)
    z = torch.tensor([1.0, -1.0] * n).reshape(2, -1, 1, 1).transpose(0, 1).float().to(device)
    # model
    t0, tN = 0.0, 1.0
    tarr = torch.arange(0, 1, 1 / n)
    dim = 1
    nhid = 50
    model = NODEintegrate(HeavyBallODE(DF(dim, nhid), nn.Parameter(torch.tensor([0.0]))), initial_velocity(dim, nhid))
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
        model.df.nfe = 0
        iter_start_time = time.time()
        optimizer.zero_grad()

        # forward in time and solve ode
        pred_z = model(None, tarr, z0)

        # compute loss
        loss = loss_func(pred_z[:, :, :1, :], z)
        loss.backward()
        optimizer.step()
        iter_end_time = time.time()
        # make arrays
        itr_arr[itr - 1] = itr
        loss_arr[itr - 1] = loss
        nfe_arr[itr - 1] = model.nfe
        time_arr[itr - 1] = iter_end_time - iter_start_time

        print('Iter: {}, running MSE: {:.4f}'.format(itr, loss))

    end_time = time.time()
    print('\n')
    print('Training complete after {} iters.'.format(itr))
    print('Time = ' + str(end_time - start_time))
    loss = loss_func(pred_z[:, :, :1, :], z).detach().numpy()
    print('Train MSE = ' + str(loss))
    print('NFE = ' + str(model.nfe))
    print('Parameters = ' + str(count_parameters(model)))

    if args.visualize:
        ntimestamps = 30
        ts = torch.tensor(np.linspace(t0, tN, ntimestamps)).float()
        pred_z = model(None, ts, z0[:, :, :1])
        pred_z = pred_z.detach().numpy()
        z = z.detach().numpy()
        print(pred_z.shape)
        ts = ts.detach().numpy()
        plt.figure()
        for i in range(2):
            plt.plot(ts, pred_z[:, i, 0, 0])
            plt.plot(tarr.numpy(), z[:, i, 0, 0])
        plt.show()

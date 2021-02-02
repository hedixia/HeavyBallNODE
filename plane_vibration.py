from base import *

parser = ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualise', type=eval, default=True)
parser.add_argument('--niters', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--npoints', type=int, default=1000)
parser.add_argument('--experiment_no', type=int, default=1)
args = parser.parse_args()

sbdat = np.loadtxt('./data/pv.csv', skiprows=1, delimiter=',', usecols=(2, 3))
data = np.transpose(sbdat)
v1_data = data[0]
v2_data = data[1]
v1_data = v1_data - np.full_like(v1_data, np.mean(v1_data))
v2_data = v2_data - np.full_like(v2_data, np.mean(v2_data))
rescaling = 1
v1_data = torch.Tensor(rescaling * v1_data).to(0)
v2_data = torch.Tensor(rescaling * v2_data).to(0)
time_rescale = 100.0 / 2 / np.pi

plt.plot(v2_data[:200].detach().cpu())
plt.show()
def v1_func(time):
    t1 = torch.clamp(torch.floor(time), 0, len(v1_data) - 1).type(torch.long)
    delta = time - t1
    return v1_data[t1] + delta * (v1_data[t1 + 1] - v1_data[t1])


class initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels, ddim):
        super(initial_velocity, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels * ddim - in_channels, bias=False)
        self.ddim = ddim

    def forward(self, x0):
        out = self.fc1(torch.ones_like(x0))
        out = torch.cat([x0, out], dim=1)
        out = rearrange(out, 'b (d c) ... -> b d c ...', d=self.ddim)
        return out


class DF(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super(DF, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.fc1 = nn.Linear(in_channels + 1, out_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, t, x):
        v1 = v1_func(t).reshape(-1, 1, 1)
        x = rearrange(x, 'b d c -> b 1 (d c)')
        z_ = torch.cat((x, v1), dim=2)
        out = self.fc1(z_) + x
        return out


# from torchdiffeq import odeint
dim = 2
hbnodeparams = {
    'timescale': 1,
    'thetaact': nn.Hardtanh(-3,3),
    'gamma': 0.1,
    'gammaact': nn.ELU()
}
model = NODEintegrate(HeavyBallNODE(DF(dim), **hbnodeparams), initial_velocity(1, dim, 2)).to(0)
model_dict = model.state_dict()
# for i in model_dict:
# model_dict[i] *= 0.01
# model_dict[i] -= 0.2
model.load_state_dict(model_dict)
criteria = nn.MSELoss()
print(count_parameters(model))

optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
for trsz in 2 ** np.arange(10):
    for epoch in range(10):
        model.df.nfe = 0
        predict = model(None, torch.arange(trsz * 1.0)/time_rescale, v2_data[:1].view(1, 1)).view(trsz, -1)[:, 0]
        loss = criteria(predict, v2_data[:trsz])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()

trsz = 1000
tssz = 3000

# train start
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.001)
timelist = [time.time()]
for epoch in range(300):
    model.df.nfe = 0
    predict = model(None, torch.arange(trsz * 1.0)/time_rescale, v2_data[:1].view(1, 1)).view(trsz, -1)[:, 0]
    loss = criteria(predict, v2_data[:trsz])
    loss.backward()
    loss = loss.detach().cpu().numpy()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    timelist.append(time.time())
    if (epoch + 10) % 10 == 0:
        forecast = model(None, (trsz + torch.arange(tssz))/time_rescale, v2_data[:1].view(1, 1)).view(tssz, -1)[:, 0]
        floss = criteria(forecast, v2_data[trsz:trsz + tssz])
        print(str_rec(['epoch', 'loss', 'nfe', 'floss', 'time', 'gamma'],
                      [epoch, loss, model.df.nfe, floss, timelist[-1] - timelist[-2],
                       model.df.gamma.detach().cpu().numpy()]))
        #print(model.df.df.fc.weight)
        plt.plot(v2_data[:200].detach().cpu())
        plt.plot(predict[:200].detach().cpu())
        plt.show()
    else:
        print(str_rec(['epoch', 'loss', 'nfe', 'time', 'gamma'],
                      [epoch, loss, model.df.nfe, timelist[-1] - timelist[-2], model.df.gamma.detach().cpu().numpy()]))

from matplotlib import pyplot as plt

trange = torch.arange(tssz + trsz * 1.0)
plt.plot(trange, v2_data[:len(trange)].cpu())
plt.plot(trange, model(None, trange, v2_data[:1].view(1, 1)).view(len(trange), -1)[:, 0].detach().cpu())
plt.show()

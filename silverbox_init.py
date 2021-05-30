"""
Fig. 3
"""

from base import *
from sonode_data_loader import load_data

parser = ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualise', type=eval, default=True)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--npoints', type=int, default=1000)
parser.add_argument('--experiment_no', type=int, default=1)
args = parser.parse_args()

v1_data, v2_data = load_data('./data/sb.csv', skiprows=1, usecols=(0, 1), rescaling=100)
time_rescale = 1.0
input_t = 1
forecast_t = 999
trsize = 1000
tssize = 4000
args.MODE = 0  # 0 for train and 1 for test


def preprocess(data):
    trdat = data[:trsize]
    tsdat = data[trsize:trsize + tssize]
    return trdat, tsdat


v1_data = preprocess(v1_data)
v2_data = preprocess(v2_data)
trdat = (v2_data[0][:input_t], v2_data[0])
tsdat = (v2_data[1][:input_t], v2_data[1])


class Vdiff(nn.Module):
    def __init__(self):
        super(Vdiff, self).__init__()
        self.osize = 1

    def forward(self, t, x, v):
        truev = v2_vfunc(t)
        return torch.norm(v[:, 0] - truev, 1)


def v1_func(time):
    t1 = torch.clamp(torch.floor(time), 0, len(v1_data) - 1).type(torch.long)
    delta = time - t1
    data = v1_data[args.MODE]
    return data[t1] + delta * (data[t1 + 1] - data[t1])


def v2_vfunc(time):
    t1 = torch.clamp(torch.floor(time), 0, len(v2_data) - 1).type(torch.long)
    data = v2_data[args.MODE]
    return data[t1 + 1] - data[t1]


class initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels, ddim, zpad=0):
        super(initial_velocity, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels * ddim - in_channels - zpad, bias=False)
        self.ddim = ddim
        self.zpad = zpad

    def forward(self, x0):
        if self.zpad > 0:
            xpad = torch.cat([x0, torch.zeros(self.zpad)], dim=0)
        else:
            xpad = x0
        out = self.fc1(torch.ones_like(x0))
        out = torch.cat([xpad, out], dim=0).reshape(1, self.ddim, -1)

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
        # z_ = torch.cat((x, 0.01 * x ** 3, v1), dim=2)
        z_ = torch.cat((x, v1), dim=2)
        out = self.fc1(z_)
        # out = 0.4905 * x + 0.00618 * x ** 3 + 0.0613 * v1
        return out


modelnames = ['NODE', 'ANODE', 'SONODE', 'HBNODE', 'GHBNODE']
modelclass = [NODE, NODE, SONODE, HeavyBallNODE, HeavyBallNODE]
icparams = [(1, 1, 0), (2, 1, 1), (1, 2, 0), (1, 2, 0), (1, 2, 0)]  # out_channels, ddim, zpad
dfparams = [(1,), (2,), (2, 1), (1,), (1,)]
cellparams = [dict(), dict(), dict(), dict(), {'corr': 0, 'actv_h': nn.Hardtanh(-5, 5)}]
model_list = []
dim = 1

plt.figure(figsize=(20, 10))
axes = plt.gca()
axes.tick_params(axis='x', labelsize=50)
axes.tick_params(axis='y', labelsize=50)
colors = ['b', 'y', 'g', 'r', 'm']
# '''
sizedata = []

for i in range(5):
    print(i)
    odesizelist = []
    for r in range(10):
        cell = modelclass[i](DF(*dfparams[i]), **cellparams[i])
        ic = initial_velocity(input_t, *icparams[i])
        nint = NODEintegrate(cell, evaluation_times=torch.arange(64.), tol=1e-7)
        model = nn.Sequential(ic, nint)
        ode_states = model(trdat[0])
        ode_size = torch.norm(ode_states.reshape(ode_states.shape[0], -1), dim=1)
        odesizelist.append(ode_size.detach().numpy())
    dat = np.log10(np.mean(odesizelist, axis=0))
    plt.plot(dat, label=modelnames[i], linewidth=5, color=colors[i])
    sizedata.append(dat)
# '''
# sizedata = np.loadtxt('output/sb/sbinit.csv', delimiter=',')
for i in range(5):
    plt.plot(sizedata[i], label=modelnames[i], linewidth=5, color=colors[i])

plt.plot(np.log10(np.abs(trdat[1][:64])), label='Exact', linewidth=5, color='k')
tickrange = np.linspace(0, 18, 7)
plt.yticks(tickrange, ['$10^{{{}}}$'.format(int(i)) for i in tickrange])
# ax.yaxis.set_major_formatter('10^{x}')
plt.xlabel("$t$", fontsize=50)
plt.ylabel("||${\\mathbf{h}}(t)||_2$", fontsize=50)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend(loc='upper left', fontsize=50)
plt.tight_layout()
plt.savefig('output/sb/blow_up.pdf')
plt.show()
np.savetxt('output/sb/sbinit.csv', np.array(sizedata), delimiter=',')

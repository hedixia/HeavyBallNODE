from base import *
from sonode_data_loader import load_data

parser = ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualise', type=eval, default=True)
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--npoints', type=int, default=1000)
parser.add_argument('--experiment_no', type=int, default=1)
args = parser.parse_args()

v1_data, v2_data = load_data('./data/sb.csv', skiprows=1, usecols=(0, 1), rescaling=100)
time_rescale = 1.0


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
    return v1_data[t1] + delta * (v1_data[t1 + 1] - v1_data[t1])


def v2_vfunc(time):
    t1 = torch.clamp(torch.floor(time), 0, len(v1_data) - 1).type(torch.long)
    return v1_data[t1 + 1] - v1_data[t1]


class initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels, ddim, zeropad=False):
        super(initial_velocity, self).__init__()
        self.fc1 = nn.Linear(in_channels, out_channels * ddim - 0 * in_channels, bias=False)
        self.ddim = ddim
        self.zeropad = zeropad

    def forward(self, x0):
        out = self.fc1(torch.ones_like(x0))
        # out = torch.cat([x0, out], dim=1)
        out[:, 0] = x0
        if self.zeropad:
            out[:, 1] = 0
        out = rearrange(out, 'b (d c) ... -> b d c ...', d=self.ddim)
        return out


class DF(nn.Module):

    def __init__(self, in_channels, out_channels=None):
        super(DF, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.fc1 = nn.Linear(2 * in_channels + 1, out_channels)
        self.act = nn.ReLU(inplace=False)

    def forward(self, t, x):
        v1 = v1_func(t).reshape(-1, 1, 1)
        x = rearrange(x, 'b d c -> b 1 (d c)')
        z_ = torch.cat((x, 0.01 * x ** 3, v1), dim=2)
        out = self.fc1(z_)
        return out


class MODEL(nn.Module):
    def __init__(self, ode, ic):
        super(MODEL, self).__init__()
        self.ode = ode
        self.ic = ic

    def forward(self, x, t):
        t = torch.Tensor([-1e-7, *t])
        ic = self.ic(x)
        out = odeint(self.ode, ic, t, rtol=1e-7, atol=1e-7)[1:]
        return out


torch.manual_seed(8)
dim = 1
modelname = 'SONODE'
fname = 'output/sb/direct/' + modelname
if modelname == 'HBNODE':
    ode = HeavyBallNODE(DF(dim), corr=0.1, corrf=False)
    ic = initial_velocity(1, dim, 2)
elif modelname == 'SONODE':
    ode = SONODE(DF(2 * dim, dim))
    ic = initial_velocity(1, dim, 2)
else:
    ode = NODE(DF(2))
    ic = initial_velocity(1, dim + 1, 1, zeropad=True)
model = shrink_parameters(MODEL(ode, ic), 0.001)
print('Number of Parameters: {}'.format(count_parameters(model)))

# train start
trsz = 1000
tssz = 3000
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00)
criteria = nn.MSELoss()
rec = Recorder()

for epoch in range(300):
    rec['epoch'] = epoch
    ode.nfe = 0
    epoch_start_time = time.time()

    # Train Forward
    predict = model(v2_data[:1].view(1, 1), torch.arange(trsz)).view(trsz, -1)[:, 0]
    loss = criteria(predict, v2_data[:trsz])
    rec['forward_nfe'] = ode.nfe

    # Train Backward
    loss.backward()
    rec['epoch_nfe'] = ode.nfe
    rec['train_loss'] = loss.detach().cpu().numpy()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    rec['train_time'] = time.time() - epoch_start_time

    if (epoch + 10) % 10 == 0:
        ode.nfe = 0
        forecast = model(v2_data[:1].view(1, 1), trsz + torch.arange(tssz)).view(tssz, -1)[:, 0]
        floss = criteria(forecast, v2_data[trsz:trsz + tssz])
        rec['forecast_loss'] = floss.detach().cpu().numpy()
        rec['test_nfe'] = ode.nfe

    rec.capture(verbose=True)

rec.writecsv(fname)

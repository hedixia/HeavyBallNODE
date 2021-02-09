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
        self.fc1 = nn.Linear(2 * in_channels + 1, out_channels, bias=False)
        self.act = nn.ReLU(inplace=False)

    def forward(self, t, x):
        v1 = v1_func(t).reshape(-1, 1, 1)
        x = rearrange(x, 'b d c -> b 1 (d c)')
        z_ = torch.cat((x, 0.01 * x ** 3, v1), dim=2)
        out = self.fc1(z_)
        return out


# from torchdiffeq import odeint
dim = 1
hbnodeparams = {
    'thetaact': nn.Hardtanh(-10, 10),
}
torch.manual_seed(8)
model = NODEintegrate(HeavyBallNODE(DF(dim), **hbnodeparams), initial_velocity(1, dim, 2), tol=args.tol,
                      adjoint=args.adjoint).to(0)
criteria = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00)
print(count_parameters(model))


def train(trsz):
    model.df.nfe = 0
    trange = torch.arange(trsz * 1.0) / time_rescale
    predict = model(None, trange.to(args.gpu), v2_data[:1].view(1, 1)).view(trsz, -1)[:, 0]
    loss = criteria(predict, v2_data[:trsz])
    loss.backward()
    loss = loss.detach().cpu().numpy()
    nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    optimizer.step()
    timelist.append(time.time())
    return predict, loss


recattrname = ['epoch', 'loss', 'nfe', 'floss', 'time', 'gamma']


def validation(trsz, tssz):
    trange = (trsz + torch.arange(tssz)) / time_rescale
    forecast = model(None, trange.to(args.gpu), v2_data[:1].view(1, 1)).view(tssz, -1)[:, 0]
    floss = criteria(forecast, v2_data[trsz:trsz + tssz])
    plt.plot(v2_data[:200].detach().cpu())
    plt.plot(predict[:200].detach().cpu())
    plt.show()
    timelist.append(time.time())
    return floss


timelist = [time.time()]

# train start
for epoch in range(args.niters):
    predict, loss = train(trsz=1000)
    floss = None
    if (epoch + 1) % 10 == 0 or epoch == 0:
        floss = validation(trsz=1000, tssz=3000)
    dtime = timelist[-1] - timelist[-2]
    gamma = model.df.gamma.detach().cpu().numpy()
    print(str_rec(recattrname, [epoch, loss, model.df.nfe, floss, dtime, gamma]))

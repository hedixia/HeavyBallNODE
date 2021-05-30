from matplotlib import pyplot as plt
import torchdiffeq
from base import *

bcnt = 30
torch.random.manual_seed(0)
trange = 10 * torch.rand(50)
trainic = torch.randn(bcnt, 2)
testic = torch.randn(bcnt, 2)
trange = torch.sort(trange)[0].to(0)

def truedf(t, x):
    dx = torch.zeros_like(x)
    dx[:,0] = x[:,1]
    dx[:,1] = -1.01 * x[:,0] - 0.2 * x[:,1]
    return dx

soltr = torchdiffeq.odeint(truedf, trainic, trange).to(0)
solts = torchdiffeq.odeint(truedf, testic, trange).to(0)
trdat = ((soltr[0, :, :], soltr[:, :, 0]), )
tsdat = ((solts[0, :, :], solts[:, :, 0]), )

plt.plot(trange.cpu(), soltr[:, 0, 0].cpu())
plt.show()

parser = ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-2)
parser.add_argument('--adjoint', type=eval, default=True)
parser.add_argument('--niters', type=int, default=200)
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

dim = 2
hidden = 20

# shape: [time, batch, derivatives]
aug = 1

class initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(initial_velocity, self).__init__()
        assert (3 * out_channels >= in_channels)
        self.diff_size = 3 * out_channels - in_channels

    def forward(self, x0):
        x0 = x0.float()
        out = torch.zeros_like(x0[:,0]) - 1
        out = repeat(out, 'b -> b c', c=self.diff_size)
        out = torch.cat([x0, out], dim=1)
        out = rearrange(out, 'b (d c) -> b d c', d=3)
        return out

class DF(nn.Module):

    def __init__(self, in_channels, nhidden):
        super(DF, self).__init__()
        self.activation = nn.LeakyReLU(0.01)
        self.fc1 = nn.Linear(in_channels, nhidden)
        self.fc2 = nn.Linear(nhidden, nhidden)
        self.fc3 = nn.Linear(nhidden, in_channels)

    def forward(self, t, x0):
        out = self.fc1(x0)
        out = self.activation(out)
        out = self.fc2(out)
        out = self.activation(out)
        out = self.fc3(out)
        return out



hbnode = NODEintegrate(HeavyBallNODE(DF(aug, hidden), None), initial_velocity(dim, aug)).to(0)
optimizer = optim.Adam(hbnode.parameters(), lr=args.lr, weight_decay=0.00)
hbtr = train(hbnode, optimizer, trdat, tsdat, trange)

node = NODEintegrate(NODE(DF(dim, hidden)), lambda x:x).to(0)
optimizer = optim.Adam(node.parameters(), lr=args.lr, weight_decay=0.00)
ntr = train(node, optimizer, trdat, tsdat, trange)

anode = NODEintegrate(NODE(DF(aug, hidden)), initial_velocity(dim, aug)).to(0)
optimizer = optim.Adam(anode.parameters(), lr=args.lr, weight_decay=0.00)
antr = train(anode, optimizer, trdat, tsdat, trange)

sonode = NODEintegrate(SONODE(DF(1, hidden)), lambda x:x).to(0)
optimizer = optim.Adam(sonode.parameters(), lr=args.lr, weight_decay=0.00)
sotr = train(sonode, optimizer, trdat, tsdat, trange)
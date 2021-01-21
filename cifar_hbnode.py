from base import *
from anode_data_loader import cifar10 as cifar

parser = ArgumentParser()
parser.add_argument('--tol', type=float, default=1e-3)
parser.add_argument('--adjoint', type=eval, default=False)
parser.add_argument('--visualize', type=eval, default=True)
parser.add_argument('--niters', type=int, default=60)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

# shape: [time, batch, derivatives, channel, x, y]
dim = 15
hidden = 30
trdat, tsdat = cifar(batch_size=256)


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
        #out = repeat(torch.zeros_like(x0[:, :1]), 'b 1 ... -> b d c ...', d=2, c=self.out_channels)
        #out[:, 0, :self.in_channels] = x0

        #"""
        out = self.fc1(x0)
        out = self.actv(out)
        out = self.fc2(out)
        out = self.actv(out)
        out = self.fc3(out)
        out = torch.cat([x0, out], dim=1)
        out = rearrange(out, 'b (d c) ... -> b d c ...', d=2)
        #"""
        return out


class DF(nn.Module):

    def __init__(self, in_channels, nhidden):
        super(DF, self).__init__()
        self.activation = nn.ReLU()  # nn.LeakyReLU(0.3)
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
        self.dense = nn.Linear(in_channels * 32 * 32, 10)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        x = rearrange(x[:, 0], 'b c x y -> b (c x y)')
        # x = self.dropout(x)
        x = self.dense(x)
        return x

#file = open('./data/cifar0.txt', 'a')
hblayer = NODElayer(HeavyBallODE(DF(dim, hidden), None))
model = nn.Sequential(initial_velocity(3, dim, hidden), hblayer, predictionlayer(dim)).to(device=args.gpu)
optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.00)
print(count_parameters(model))
if count_parameters(model) > 173000 or count_parameters(model) < 171000:
    exit(1)
train(model, optimizer, trdat, tsdat, args, 1,  gamma=model[1].df.gamma)

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


def v1_func(time):
    t1 = torch.clamp(torch.floor(time), 0, len(v1_data) - 1).type(torch.long)
    delta = time - t1
    return v1_data[:, t1] + delta * (v1_data[:, t1 + 1] - v1_data[:, t1])


class initial_velocity(nn.Module):

    def __init__(self, in_channels, out_channels, ddim, nhid=5):
        super(initial_velocity, self).__init__()
        self.fc1 = nn.Linear(in_channels, nhid)
        self.fc3 = nn.Linear(nhid, out_channels * ddim - in_channels)
        self.act = nn.ELU(inplace=False)
        self.ddim = ddim

    def forward(self, x0):
        out = self.fc1(x0)
        out = self.act(out)
        out = self.fc3(out)
        out = torch.cat([x0, out], dim=1)
        out = rearrange(out, 'b (d c) ... -> b d c ...', d=self.ddim)
        return out


class DF(nn.Module):

    def __init__(self, in_channels, out_channels=None, nhid=5):
        super(DF, self).__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.fc1 = nn.Linear(in_channels + 1, nhid)
        self.fc3 = nn.Linear(nhid, out_channels)
        self.act = nn.ELU(inplace=False)

    def forward(self, t, x):
        v1 = v1_func(t).reshape(-1, 1, 1)
        x = rearrange(x, 'b d c -> b 1 (d c)')
        z_ = torch.cat((x, v1), dim=2)
        out = self.fc1(z_)
        out = self.act(out)
        out = self.fc3(out)
        return out


sbdat = np.loadtxt('./data/pv.csv', skiprows=1, delimiter=',', usecols=(2, 3))
data = np.transpose(sbdat)
v1_data = data[0]
v2_data = data[1]
v1_data = v1_data - np.full_like(v1_data, np.mean(v1_data))
v2_data = v2_data - np.full_like(v2_data, np.mean(v2_data))
rescaling = 1
v1_data = torch.Tensor(rescaling * v1_data).to(0)
v2_data = torch.Tensor(rescaling * v2_data).to(0)
seqlen = 1000
v1_data = v1_data.unfold(0, seqlen, 500)
v2_data = v2_data.unfold(0, seqlen, 500)
print(v1_data.shape, v2_data.shape)
trsz = 100
tssz = 40


def test_init(modelname, dim, nhid, pretrain_epoch):
    try:
        if modelname == 'hbnode':
            model = NODEintegrate(HeavyBallNODE(DF(dim, nhid=nhid), None), initial_velocity(1, dim, 2, nhid=3)).to(0)
        elif modelname == 'sonode':
            model = NODEintegrate(SONODE(DF(2 * dim, nhid=nhid, out_channels=dim)),
                                  initial_velocity(1, dim, 2, nhid=nhid)).to(0)
        else:
            print('No model matched')
            return
        print('model size:', count_parameters(model))
        optimizer = optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=0.00)
        criteria = nn.MSELoss()
        pretrain_start_time = time.time()
        for subseqlen in 2 ** np.arange(np.int(np.log2(seqlen))):
            for epoch in range(pretrain_epoch):
                model.df.nfe = 0
                predict = model(None, torch.arange(subseqlen * 1.0), v2_data[:, :1])[:, :, 0, 0]
                loss = criteria(predict[:, :trsz], v2_data.transpose(0, 1)[:subseqlen, :trsz])
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 10.0)
                optimizer.step()
        print('Pretrain takes {} seconds'.format(time.time() - pretrain_start_time))
        predict = model(None, torch.arange(seqlen * 1.0), v2_data[:, :1])[:, :, 0, 0]
        loss = criteria(predict[:, :trsz], v2_data.transpose(0, 1)[:seqlen, :trsz])
        loss.backward()
        return True
    except (RuntimeError, AssertionError):
        return False


ntest = 100
npre = 0
srate = np.zeros(ntest)
hrate = np.zeros(ntest)
for i in range(ntest):
    print('test', i)
    srate[i] = test_init('sonode', 2, 3, npre)
    hrate[i] = test_init('hbnode', 2, 4, npre)

print('Sonode initialization success rate: {}'.format(np.mean(srate)))
print('Hbnode initialization success rate: {}'.format(np.mean(hrate)))

ntest = 100
npre = 1
srate = np.zeros(ntest)
hrate = np.zeros(ntest)
for i in range(ntest):
    print('test', i)
    srate[i] = test_init('sonode', 2, 3, npre)
    hrate[i] = test_init('hbnode', 2, 4, npre)

print('Sonode initialization success rate: {}'.format(np.mean(srate)))
print('Hbnode initialization success rate: {}'.format(np.mean(hrate)))

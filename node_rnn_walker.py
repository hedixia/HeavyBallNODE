from base import *

from odelstm_data import Walker2dImitationData

seqlen = 64
data = Walker2dImitationData(seq_len=seqlen)


class tempf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels, out_channels)

    def forward(self, h, x):
        out = self.dense1(x)
        return out


class temprnn(nn.Module):
    def __init__(self, in_channels, out_channels, nhidden):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels + nhidden, nhidden * 4)
        self.dense2 = nn.Linear(nhidden * 4, out_channels)

    def forward(self, h, x):
        out = torch.cat([h, x], dim=1)
        out = self.dense1(out)
        out = self.actv(out)
        out = self.dense2(out)
        out = out.reshape(h.shape)
        return out


class tempout(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        out = self.dense1(x)
        return out


class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        nhid = 32
        self.cell = NODE(tempf(nhid, nhid))
        self.rnn = temprnn(17, nhid, nhid)
        self.ode_rnn = ODE_RNN(self.cell, self.rnn, nhid, tol=1e-7)
        self.outlayer = tempout(nhid, 17)

    def forward(self, t, x):
        out = self.ode_rnn(t, x)
        out = self.outlayer(out)[:-1]
        return out


torch.manual_seed(0)
model = MODEL()
fname = 'output/walker/log_node.txt'
outfile = open(fname, 'w')
outfile.write(model.__str__())
outfile.write('\n' * 3)
outfile.close()
criteria = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
print('Number of Parameters: {}'.format(count_parameters(model)))
timelist = [time.time()]
for epoch in range(1000):
    model.cell.nfe = 0
    batchsize = 512
    losslist = []
    for b_n in range(0, data.train_x.shape[1], batchsize):
        predict = model(data.train_times[:, b_n:b_n + batchsize] / 64.0, data.train_x[:, b_n:b_n + batchsize])
        loss = criteria(predict, data.train_y[:, b_n:b_n + batchsize])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        losslist.append(loss.detach().cpu().numpy())
        optimizer.step()
    tloss = np.mean(losslist)
    timelist.append(time.time())
    nfe = model.cell.nfe / len(range(0, data.train_x.shape[1], batchsize))
    if epoch == 0 or (epoch + 1) % 5 == 0:
        predict = model(data.valid_times / 64.0, data.valid_x)
        vloss = criteria(predict, data.valid_y)
        vloss = vloss.detach().cpu().numpy()
        outfile = open(fname, 'a')
        outstr = str_rec(['epoch', 'tloss', 'nfe', 'vloss', 'time'],
                         [epoch, tloss, nfe, vloss, timelist[-1] - timelist[-2]])
        print(outstr)
        outfile.write(outstr + '\n')
        outfile.close()
    if epoch == 0 or (epoch + 1) % 20 == 0:
        model.cell.nfe = 0
        predict = model(data.test_times / 64.0, data.test_x)
        sloss = criteria(predict, data.test_y)
        sloss = sloss.detach().cpu().numpy()
        outfile = open(fname, 'a')
        outstr = str_rec(['epoch', 'nfe', 'sloss', 'time'],
                         [epoch, model.cell.nfe, sloss, timelist[-1] - timelist[-2]])
        print(outstr)
        outfile.write(outstr + '\n')
        outfile.close()

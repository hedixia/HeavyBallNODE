from trainpv import *


class tempf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv = nn.ReLU()
        self.dense1 = nn.Linear(in_channels, in_channels)
        self.dense2 = nn.Linear(in_channels, out_channels)
        self.dense3 = nn.Linear(out_channels, out_channels)

    def forward(self, h, x):
        out = self.dense1(x)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        out = self.dense3(out)
        return out


class temprnn(nn.Module):
    def forward(self, h, x):
        return h


class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        nhid = 23
        self.cell = HeavyBallNODE(tempf(nhid, nhid), corr=0, corrf=False)
        # self.cell = HeavyBallNODE(tempf(nhid, nhid))
        self.ic = nn.Linear(5 * seqlen, 2 * nhid)
        self.rnn = temprnn()
        self.ode_rnn = ODE_RNN(self.cell, self.rnn, (2, nhid), self.ic, rnn_out=False, both=True, tol=1e-7)
        self.outlayer = nn.Linear(nhid, 5)

    def forward(self, t, x, multiforecast=None):
        out = self.ode_rnn(t, x, multiforecast=multiforecast, retain_grad=True)
        if multiforecast is not None:
            rnn, out, fore = out
            out = (self.outlayer(rnn[:, :, 0])[:-1], self.outlayer(out[:, :, 0])[1:], self.outlayer(fore[:, :, 0]))
        else:
            out = (self.outlayer(out[:, :, 0])[1:],)
        return out



if __name__ == '__main__':
    model = MODEL()
    trainpv(model, 'output/pv/log_hb_{}.csv'.format(count_parameters(model)), 'output/pv_hbnode.mdl', gradrec=True)

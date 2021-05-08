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
    def __init__(self, in_channels, out_channels, nhidden, res=False, cont=False):
        super().__init__()
        self.actv = nn.ReLU()
        self.dense1 = nn.Linear(in_channels,  out_channels)
        self.dense2 = nn.Linear(nhidden,  out_channels)
        self.dense2q = nn.Linear(nhidden, out_channels)
        # self.dense2y = nn.Linear(nhidden, nhidden)
        # self.dense3 = nn.Linear(nhidden, 2*out_channels)
        self.cont = cont
        self.res = res

    def forward(self, h, x):
        p, q = torch.split(h.clone(), 1, dim=1)
        y = self.dense1(x)
        # y = self.actv(y)
        m_ = self.dense2(p) + self.dense2q(q)
        m_ = m_ + y.view(m_.shape)  # self.dense2y(y.view(p.shape))
        # m_ = self.actv(m_)
        # m_ = self.dense3(m_)
        out = m_.view(q.shape) + q
        out = torch.cat([p, out], dim=1)
        return out


class MODEL(nn.Module):
    def __init__(self, res=False, cont=False):
        super(MODEL, self).__init__()
        nhid = 21
        self.cell = HeavyBallNODE(tempf(nhid, nhid), corr=0, corrf=False)
        # self.cell = HeavyBallNODE(tempf(nhid, nhid))
        self.ic = nn.Linear(5 * seqlen, 2 * nhid)
        self.rnn = temprnn(5, nhid, nhid, res=res, cont=cont)
        self.ode_rnn = ODE_RNN_with_Grad_Listener(self.cell, self.rnn, (2, nhid), self.ic, rnn_out=False, both=True, tol=1e-7)
        self.outlayer = nn.Linear(nhid, 5)

    def forward(self, t, x, multiforecast=None):
        out = self.ode_rnn(t, x, multiforecast=multiforecast)
        if multiforecast is not None:
            rnn, out, fore = out
            out = (self.outlayer(rnn[:, :, 0])[:-1], self.outlayer(out[:, :, 0])[1:], self.outlayer(fore[:, :, 0]))
        else:
            out = self.outlayer(out[:, :, 0])[1:]

        return out



if __name__ == '__main__':
    model = MODEL()
    trainpv(model, 'output/pv/log_hbc0_{}.csv'.format(count_parameters(model)), 'output/pv_hbnode_rnn.mdl')

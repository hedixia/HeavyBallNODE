from trainpv import *


class tempf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv = nn.ReLU()
        self.dense1 = nn.Linear(in_channels, 3*in_channels)
        self.dense2 = nn.Linear(3*in_channels, 4*out_channels)
        self.dense3 = nn.Linear(4*out_channels, out_channels)

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
        self.dense1 = nn.Linear(in_channels, nhidden)
        self.dense2 = nn.Linear(nhidden, nhidden)
        self.cont = cont
        self.res = res

    def forward(self, h, x):
        h = h.clone()
        y = self.dense1(x)
        # y = self.actv(y)
        m_ = self.dense2(h) + y  # self.dense2y(y.view(h.shape))
        # m_ = self.actv(m_)
        # m_ = self.dense3(m_)
        out = m_ + h
        return out


class MODEL(nn.Module):
    def __init__(self, res=False, cont=False):
        super(MODEL, self).__init__()
        nhid = 21
        self.cell = NODE(tempf(nhid, nhid))
        self.rnn = temprnn(5, nhid, nhid, res=res, cont=cont)
        self.ic = nn.Linear(5 * seqlen, nhid)
        self.ode_rnn = ODE_RNN(self.cell, self.rnn, nhid, self.ic, rnn_out=False, both=True, tol=1e-7)
        self.outlayer = nn.Linear(nhid, 5)

    def forward(self, t, x, multiforecast=None):
        out = self.ode_rnn(t, x, multiforecast=multiforecast)
        if multiforecast is not None:
            rnn, out, fore = out
            out = (self.outlayer(rnn)[:-1], self.outlayer(out)[1:], self.outlayer(fore))
        else:
            out = self.outlayer(out)[1:]
        return out



if __name__ == '__main__':
    model = MODEL()
    trainpv(model, 'output/pv/log_n0_{}.csv'.format(count_parameters(model)), 'output/pv_node_rnn.mdl')

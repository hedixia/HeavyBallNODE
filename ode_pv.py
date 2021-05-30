from deprecated.plane_vibration import *


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
    def forward(self, h, x):
        return h


class MODEL(nn.Module):
    def __init__(self):
        super(MODEL, self).__init__()
        nhid = 22
        self.cell = NODE(tempf(nhid, nhid))
        self.rnn = temprnn()
        self.ic = nn.Linear(5 * seqlen, nhid)
        self.ode_rnn = ODE_RNN_with_Grad_Listener(self.cell, self.rnn, nhid, self.ic, rnn_out=False, both=True, tol=1e-7)
        self.outlayer = nn.Linear(nhid, 5)

    def forward(self, t, x, multiforecast=None):
        out = self.ode_rnn(t, x, multiforecast=multiforecast, retain_grad=True)
        if multiforecast is not None:
            rnn, out, fore = out
            out = (self.outlayer(rnn)[:-1], self.outlayer(out)[1:], self.outlayer(fore))
        else:
            out = self.outlayer(out)[1:]
        return out



if __name__ == '__main__':
    model = MODEL()
    trainpv(model, 'results/plane_vibration/log_n_{}.csv'.format(count_parameters(model)), 'results/plane_vibration/pv_node.mdl', gradrec=True, pre_shrink=0.01)

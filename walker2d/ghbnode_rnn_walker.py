from base import *

from odelstm_data import Walker2dImitationData

seqlen = 64
data = Walker2dImitationData(seq_len=seqlen, device=0)


class tempf(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels, in_channels)
        # self.dense2 = nn.Linear(in_channels, out_channels)
        # self.dense3 = nn.Linear(out_channels, out_channels)

    def forward(self, h, x):
        out = self.dense1(x)
        # out = self.actv(out)
        # out = self.dense2(out)
        # out = self.actv(out)
        # out = self.dense3(out)
        return out


class temprnn(nn.Module):
    def __init__(self, in_channels, out_channels, nhidden, res=False, cont=False):
        super().__init__()
        self.actv = nn.Tanh()
        self.dense1 = nn.Linear(in_channels + 2 * nhidden, 2 * nhidden)
        self.dense2 = nn.Linear(2 * nhidden, 2 * nhidden)
        self.dense3 = nn.Linear(2 * nhidden, 2 * out_channels)
        self.cont = cont
        self.res = res

    def forward(self, h, x):
        out = torch.cat([h[:, 0], h[:, 1], x], dim=1)
        out = self.dense1(out)
        out = self.actv(out)
        out = self.dense2(out)
        out = self.actv(out)
        out = self.dense3(out).reshape(h.shape)
        out = out + h
        return out


class MODEL(nn.Module):
    def __init__(self, res=False, cont=False):
        super(MODEL, self).__init__()
        nhid = 24
        self.cell = HeavyBallNODE(tempf(nhid, nhid), corr=1, corrf=False, actv_h=nn.Tanh())
        # self.cell = HeavyBallNODE(tempf(nhid, nhid))
        self.rnn = temprnn(17, nhid, nhid, res=res, cont=cont)
        self.ode_rnn = ODE_RNN_with_Grad_Listener(self.cell, self.rnn, (2, nhid), None, tol=1e-7)
        self.outlayer = nn.Linear(nhid, 17)

    def forward(self, t, x):
        out = self.ode_rnn(t, x, retain_grad=True)[0]
        out = self.outlayer(out[:, :, 0])[1:]
        return out


def main():
    data = Walker2dImitationData(seq_len=seqlen, device=0)
    gradrec = True
    lr_dict = {0: 0.001, 50: 0.003}
    res = True
    cont = True
    torch.manual_seed(0)
    model = MODEL(res=res, cont=cont).to(0)
    modelname = 'GHBNODE'
    print(model.__str__())
    rec = Recorder()
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[0])
    print('Number of Parameters: {}'.format(count_parameters(model)))
    timelist = [time.time()]
    for epoch in range(500):
        rec['epoch'] = epoch
        if epoch in lr_dict:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[epoch])

        batchsize = 256
        train_start_time = time.time()
        for b_n in range(0, data.train_x.shape[1], batchsize):
            model.cell.nfe = 0
            predict = model(data.train_times[:, b_n:b_n + batchsize] / 64.0, data.train_x[:, b_n:b_n + batchsize])
            loss = criteria(predict, data.train_y[:, b_n:b_n + batchsize])
            rec['forward_nfe'] = model.cell.nfe
            rec['loss'] = loss

            # Gradient backprop computation
            if gradrec is not None:
                lossf = criteria(predict[-1], data.train_y[-1, b_n:b_n + batchsize])
                lossf.backward(retain_graph=True)
                vals = model.ode_rnn.h_rnn
                for i in range(len(vals)):
                    grad = vals[i].grad
                    rec['grad_{}'.format(i)] = 0 if grad is None else torch.norm(grad)
                model.zero_grad()

            model.cell.nfe = 0
            loss.backward()
            rec['backward_nfe'] = model.cell.nfe
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        rec['train_time'] = time.time() - train_start_time
        if epoch == 0 or (epoch + 1) % 1 == 0:
            model.cell.nfe = 0
            predict = model(data.valid_times / 64.0, data.valid_x)
            vloss = criteria(predict, data.valid_y)
            rec['va_nfe'] = model.cell.nfe
            rec['va_loss'] = vloss
        if epoch == 0 or (epoch + 1) % 20 == 0:
            model.cell.nfe = 0
            predict = model(data.test_times / 64.0, data.test_x)
            sloss = criteria(predict, data.test_y)
            sloss = sloss.detach().cpu().numpy()
            rec['ts_nfe'] = model.cell.nfe
            rec['ts_loss'] = sloss
        rec.capture(verbose=True)
        if (epoch + 1) % 20 == 0:
            torch.save(model, 'output/walker/{}_rnn_{}.mdl'.format(modelname, count_parameters(model)))
            rec.writecsv('output/walker/{}_rnn_{}.csv'.format(modelname, count_parameters(model)))

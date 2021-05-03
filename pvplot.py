import odernn_pv
import hbnode_rnn_pv
import hbnode_rnn_cont_pv
from trainpv import *
from base import *

spec = 10
obsnum = 1
x = data.test_x[:, obsnum:obsnum + 1]
y = data.test_y[:, obsnum:obsnum + 1]
t = data.test_times[:, obsnum:obsnum + 1]
p = data.tsext[:, obsnum:obsnum + 1, 0]
multiforecast = torch.linspace(0, forelen - 1, spec)
linsp = torch.linspace(0, 1, spec)
spec_t = torch.ones(len(t), spec)
spec_t[:, 0] = 0
spec_t = (spec_t * t) / (spec - 1)
spec_t = spec_t.flatten()
elapsed_time = torch.cumsum(spec_t, dim=0).detach().numpy()
forecast_time = elapsed_time[-1] + multiforecast

data_time = torch.cumsum(t.flatten(), dim=0).detach().numpy()
data_time = np.array([0, *data_time])
data_forecast_time = data_time[-1] + torch.arange(forelen)
truesol = [*x[:, 0, 0].numpy(), y[-1, 0, 0]]

print(elapsed_time[-5:])


def makeout(model):
    ode_rnn = model.ode_rnn
    n_t, n_b = t.shape
    h_ode = torch.zeros(n_t + 1, n_b, *ode_rnn.nhid, device=x.device)
    h_rnn = torch.zeros(n_t + 1, n_b, *ode_rnn.nhid, device=x.device)
    h_ode[0] = h_rnn[0] = ode_rnn.ic(rearrange(x, 't b c -> b (t c)')).view(h_ode[0].shape)
    odeval_list = []

    for i in range(n_t):
        ode_rnn.ode.update(t[i])
        h_rnn[i] = ode_rnn.rnn(h_ode[i], x[i])
        odevals = odeint(ode_rnn.ode, h_rnn[i], torch.linspace(0, 1, spec), atol=ode_rnn.tol, rtol=ode_rnn.tol)
        h_ode[i + 1] = odevals[-1]
        odeval_list.append(odevals.detach().cpu())

    ode_rnn.ode.update(torch.ones_like((t[0])))
    forecast = odeint(ode_rnn.ode, h_ode[-1], multiforecast * 1.0, atol=ode_rnn.tol, rtol=ode_rnn.tol)
    predict = model.outlayer(h_ode).detach().numpy()
    forecast = model.outlayer(forecast).detach().numpy()
    predict = rearrange(predict, 't ... -> t (...)')
    forecast = rearrange(forecast, 't ... -> t (...)')
    odeout = model.outlayer(torch.cat(odeval_list, dim=0)).detach().numpy()
    predict = rearrange(odeout, 't ... -> t (...)')
    return predict[:, 0], forecast[:, 0]


ode_rnn_pre_model = odernn_pv.MODEL()
ode_rnn_pre_model.load_state_dict(torch.load('output/pv_node_rnn.mdl'))
hbnode_rnn_pre_model = hbnode_rnn_pv.MODEL()
hbnode_rnn_pre_model.load_state_dict(torch.load('output/pv_hbnode_rnn.mdl'))
hbnode_c_rnn_pre_model = hbnode_rnn_cont_pv.MODEL()
hbnode_c_rnn_pre_model.load_state_dict(torch.load('output/pv_hbnode_rnn_cont.mdl'))

odepre, odefore = makeout(ode_rnn_pre_model)
hbnodepre, hbnodefore = makeout(hbnode_rnn_pre_model)
hbcpre, hbcfore = makeout(hbnode_c_rnn_pre_model)

plt.plot(elapsed_time, odepre, 'r')
plt.plot(elapsed_time, hbnodepre, 'g')
plt.plot(elapsed_time, hbcpre, 'c')
plt.plot(data_time, truesol, 'b')
plt.plot(forecast_time, odefore, 'r')
plt.plot(forecast_time, hbnodefore, 'g')
plt.plot(forecast_time, hbcfore, 'c')
plt.plot(data_forecast_time, p, 'b')
plt.legend(['ODE-RNN', 'HBNODE-RNN', 'HBNODE-RNN-cont', 'Observation'])
plt.show()

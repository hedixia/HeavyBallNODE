import node_rnn_pv
import hbnode_rnn_pv
import sonode_rnn_pv
import anode_rnn_pv
import ghbnode_rnn_pv
import hbnode_rnn_cont_pv
from trainpv import *
from base import *

spec = 10
obsnum = 10
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
elapsed_time = torch.cumsum(spec_t, dim=0).detach()
forecast_time = elapsed_time[-1] + multiforecast

data_time = torch.cumsum(t.flatten(), dim=0).detach()
data_time = torch.Tensor([0, *data_time])
data_forecast_time = data_time[-1] + torch.arange(forelen)
truesol = [*x[:, 0, 0].numpy(), y[-1, 0, 0]]


fine_time = torch.cat([elapsed_time, forecast_time], dim=0)
crude_time = torch.cat([data_time, data_forecast_time], dim=0)


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

modules = [node_rnn_pv,anode_rnn_pv, sonode_rnn_pv,  hbnode_rnn_pv, ghbnode_rnn_pv]
models = [m.MODEL() for m in modules]
modelnames = ['node', 'anode', 'sonode', 'hbnode', 'ghbnode']
colors = ['r', 'g', 'b', 'c', 'y']
state_dicts = [torch.load('output/pv_{}_rnn.mdl'.format(name)) for name in modelnames]
csvdat = torch.zeros(7, fine_time.shape[0])
for i in range(5):
    models[i].load_state_dict(state_dicts[i])
    prediction, forecast = makeout(models[i])
    seq = np.concatenate([prediction, forecast], axis=0)
    plt.plot(fine_time, seq, colors[i], label=modelnames[i])
    csvdat[i] = torch.Tensor(seq)

truesol = np.array(truesol)
p = p.flatten()
print(truesol.shape, p.shape)
truedat = np.concatenate([truesol, p], axis=0)
from scipy.interpolate import interp1d
f = interp1d(crude_time.numpy(), truedat)
truefine = f(fine_time.numpy())
plt.plot(fine_time, truefine, 'k', label='Truth')
plt.legend()
plt.show()

csvdat[-2] = torch.Tensor(truefine)
csvdat[-1] = torch.Tensor(fine_time)
print(data_time[-1])
print(csvdat)
np.savetxt('results/plane_vibration/sample_plot.csv', csvdat.numpy(), delimiter=',')
from misc import *


class Example_df(nn.Module):
    def __init__(self, input_dim, output_dim, nhid=20):
        super(Example_df, self).__init__()
        self.dense1 = nn.Linear(input_dim, nhid)
        self.dense2 = nn.Linear(nhid, output_dim)
        self.lrelu = nn.LeakyReLU(0.2)

    def forward(self, t, x):
        x = self.dense1(x)
        x = self.lrelu(x)
        x = self.dense2(x)
        return x


class Zeronet(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


class NODEintegrate(nn.Module):

    def __init__(self, df=None, x0=None):
        """
        Create an OdeRnnBase model
            x' = df(x)
            x(t0) = x0
        :param df: a function that computes derivative. input & output shape [batch, channel, feature]
        :param x0: initial condition.
            - if x0 is set to be nn.parameter then it can be trained.
            - if x0 is set to be nn.Module then it can be computed through some network.
        """
        super().__init__()
        self.df = df
        self.x0 = x0

    def forward(self, initial_condition, evaluation_times, x0stats=None):
        """
        Evaluate odefunc at given evaluation time
        :param initial_condition: shape [batch, channel, feature]. Set to None while training.
        :param evaluation_times: time stamps where method evaluates, shape [time]
        :param x0stats: statistics to compute x0 when self.x0 is a nn.Module, shape required by self.x0
        :return: prediction by ode at evaluation_times, shape [time, batch, channel, feature]
        """
        if initial_condition is None:
            initial_condition = self.x0
        if x0stats is not None:
            initial_condition = self.x0(x0stats)
        out = odeint(self.df, initial_condition, evaluation_times, rtol=tol, atol=tol)
        return out

    @property
    def nfe(self):
        return self.df.nfe


class NODElayer(nn.Module):
    def __init__(self, df, evaluation_times=(0.0, 1.0)):
        super(NODElayer, self).__init__()
        self.df = df
        self.evaluation_times = torch.as_tensor(evaluation_times)

    def forward(self, x0):
        out = odeint(self.df, x0, self.evaluation_times, rtol=tol, atol=tol)
        return out[1]

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.evaluation_times.to(device)


class NODE(nn.Module):
    def __init__(self, df=None, **kwargs):
        super(NODE, self).__init__()
        self.__dict__.update(kwargs)
        self.df = df
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        return self.df(t, x)


class SONODE(NODE):
    def forward(self, t, x):
        """
        Compute [y y']' = [y' y''] = [y' df(t, y, y')]
        :param t: time, shape [1]
        :param x: [y y'], shape [batch, 2, vec]
        :return: [y y']', shape [batch, 2, vec]
        """
        self.nfe += 1
        v = x[:, 1:, :]
        out = self.df(t, x)
        return torch.cat((v, out), dim=1)


class HeavyBallNODE(NODE):
    def __init__(self, df, thetaact=None, thetalin=None, gamma_guess=-3.0, gammaact='sigmoid', gamma_correction=False):
        super().__init__(df)
        self.gamma = nn.Parameter(torch.Tensor([gamma_guess]))
        self.gammaact = nn.Sigmoid() if gammaact == 'sigmoid' else gammaact
        self.thetaact = nn.Identity() if thetaact is None else thetaact
        self.thetalin = Zeronet() if thetalin is None else thetalin
        self.gamma_correction = gamma_correction

    def forward(self, t, x):
        """
        Compute [theta' m' v'] with heavy ball parametrization in
        $$ theta' = -m / sqrt(v + eps) $$
        $$ m' = h f'(theta) - rm $$
        $$ v' = p (f'(theta))^2 - qv $$
        https://www.jmlr.org/papers/volume21/18-808/18-808.pdf
        because v is constant, we change c -> 1/sqrt(v)
        c has to be positive
        :param t: time, shape [1]
        :param x: [theta m v], shape [batch, 3, dim]
        :return: [theta' m' v'], shape [batch, 3, dim]
        """
        self.nfe += 1
        theta, m = torch.split(x, 1, dim=1)
        dtheta = self.thetaact(self.thetalin(theta) - m)
        dm = self.df(t, theta) - torch.sigmoid(self.gamma) * m
        dm += self.gamma_correction * self.gamma * self.gamma * theta / 4
        return torch.cat((dtheta, dm), dim=1)


class ODERNN(nn.Module):
    def __init__(self, node, rnn, evaluation_times):
        super(ODERNN, self).__init__()
        self.t = evaluation_times
        self.node = node
        self.rnn = rnn

    def forward(self, x, rnn_feed):
        out = torch.zeros([len(self.t), *x.shape]).to(x.device())
        for i in range(1, len(self.t)):
            temp = self.node(self.t[i - 1:i + 1], out[i - 1])
            out[i] = self.rnn(temp, rnn_feed[i])
        return out

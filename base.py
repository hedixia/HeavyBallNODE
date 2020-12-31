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


class NODE(nn.Module):
    def __init__(self, df=None, **kwargs):
        super(NODE, self).__init__()
        self.__dict__.update(kwargs)
        self.df = df
        self.nfe = 0

    def forward(self, t, x):
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


class HeavyBallODE(NODE):
    def __init__(self, df, gamma):
        super().__init__(df)
        self.gamma = torch.as_tensor(gamma)

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
        theta, m, c = torch.split(x, 1, dim=1)
        dtheta = - m * c
        dm = self.df(t, theta) - torch.sigmoid(self.gamma) * m
        dc = 0 * c
        return torch.cat((dtheta, dm, dc), dim=1)

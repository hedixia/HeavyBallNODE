from base import *

class tempc(nn.Module):
    def forward(self, t, x):
        return torch.ones_like(x)

class tempr(nn.Module):
    def forward(self, h, x):
        return h

cell = NODE(tempc())
rnn = tempr()
model = ODE_RNN(cell, rnn, 5)
t = torch.arange(6).reshape(3, 2)
x = torch.arange(12).reshape(3, 2, 2)
out = model(t, x)
print(out.shape)


cell = HeavyBallNODE(tempc())
rnn = tempr()
model = ODE_RNN(cell, rnn, (2, 5))
t = torch.arange(6).reshape(3, 2)
x = torch.arange(12).reshape(3, 2, 2)
out = model(t, x)
print(out.shape)
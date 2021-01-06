# HeavyBallNODE

Data format shape: 
[timestamps, batch, channels (derivatives), feature dimension]

Usage:

First create a NODE type module by 

node = NODE(...)

And turn it into a time series model by 

model = NODEintegrate(node, initial_condition)

Comparably there is a residual network type model by

model = NODElayer(node)

Initial condition input are

1. constant input
2. nn.Parameter
3. nn.Module

In training phase, call model with parameters:

- initial_condition = None
- evaluation_times = evaluation_times, including the first one as starting time.
- x0stats = if initial condition is a nn.Module, then provide its input. Otherwise, set it to None (default).

In evaluating phase, if there is a initial condition given, then input initial_condition. Otherwise, set it to None.

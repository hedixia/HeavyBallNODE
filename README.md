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


Experiments:

- Silverbox initialization test in fig.3: silverbox_init.py
- Point cloud separation in sec 5.1: 
- MNIST in sec 5.2: mnist/mnist_full_run.py
- CIFAR in sec 5.2: HeavyBall_CIFAR.ipynb
- Plane Vibration in sec 5.3: plane_vibration
- Walker2D in sec 5.4: walker2d
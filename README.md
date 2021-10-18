# Heavy Ball Neural Ordinary Differential Equations

This is the official implementation of Heavy Ball Neural Ordinary Differential Equations. 
For any questions about the code, please correspond to hedixia@ucla.edu.
The code is based on Pytorch and torchdiffeq, and all default numerical solvers used in the experiments are dopri5.

## Usage

Download walker2d data by 

`python data_download.py`

Data format shape: 
[timestamps, batch, channels (derivatives), feature dimension]

Usage:

First create a NODE type module by 

`cell = NODE(...)`

Or a HBNODE by 

`cell = HBNODE(...)`

And turn it into a time series model by 

`model = NODEintegrate(cell)`

It can also be used as a residual network analogy by

`model = NODElayer(cell)`

For NODE-RNN type hybrids, use 

`model = ODE_RNN(ode, cell, nhid, ic)`

here `nhid` is the hidden shape (same shape as ode / cell input and output). `ic` is the initial conditions.


## Experiments

As Jupyter Notebooks:

- Point cloud separation in sec 5.1: point_cloud/nested_n_spheres_hbnode.ipynb
- CIFAR in sec 5.2: HeavyBall_CIFAR.ipynb

As Python files

- Silverbox initialization test in fig.3: python3 silverbox_init.py
- MNIST in sec 5.2: python3 mnist/mnist_full_run.py
- Plane Vibration in sec 5.3: python3 run.py pv hbnode
- Walker2D in sec 5.4: python3 run.py walker hbnode
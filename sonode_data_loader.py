from misc import *

def load_data(filename, skiprows=0, usecols=None, rescaling=1):
    sbdat = np.loadtxt(filename, skiprows=skiprows, delimiter=',', usecols=usecols)
    data = np.transpose(sbdat)
    v1_data = data[0]
    v2_data = data[1]
    v1_data = v1_data - np.full_like(v1_data, np.mean(v1_data))
    v2_data = v2_data - np.full_like(v2_data, np.mean(v2_data))
    v1_data = torch.Tensor(rescaling * v1_data)
    v2_data = torch.Tensor(rescaling * v2_data)
    return v1_data, v2_data
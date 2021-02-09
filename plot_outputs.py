import os
import re

from misc import *
from output_parser import OutputParser

output_data_folder = './hbnode_data'
# datdir = 'mnist_84k'
datdir = 'cifar_172k'
dsname = datdir[:5].upper()
root = '/'.join([output_data_folder, datdir])
methodnames = os.listdir(root)
print('Methods:', methodnames)


def plot(xmax, ds, attr, ylim=False, yextra=''):
    epoches = np.arange(xmax + 1)
    for method in methodnames:
        cnt = np.zeros(xmax + 1)
        vals = np.zeros(xmax + 1)
        for i in os.listdir('/'.join([root, method])):
            fname = '/'.join([root, method, i])
            op = OutputParser(fname)
            epoch, val = op.get(ds, attr, np.array)
            for j in range(len(epoch)):
                try:
                    cnt[epoch[j]] += 1
                    vals[epoch[j]] += val[j]
                except IndexError:
                    pass
        vals = vals / cnt
        index = np.argwhere(~np.isnan(vals))
        plt.plot(epoches[index], vals[index])
    if ylim:
        plt.ylim(*ylim)
    plt.legend(methodnames)
    plt.title('{} {} average {}'.format(dsname, ds, attr))
    plt.ylabel(attr + yextra)
    plt.xlabel('epoch')
    plt.savefig('{}/image/{}_{}_{}'.format(output_data_folder, dsname, ds, re.sub('/', '_', attr)))
    plt.show()


plot(40, 'train', 'nfe', ylim=(0, 100))
plot(40, 'test', 'acc', ylim=(0.8, 1))
plot(40, 'train', 'time', ylim=(0, 200), yextra=' (min)')

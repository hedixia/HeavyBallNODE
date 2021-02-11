import os
import re

from misc import *
from output_parser import OutputParser

output_data_folder = './hbnode_data'
datdir = 'mnist_84k'
#datdir = 'cifar_172k'
dsname = datdir[:5].upper()
root = '/'.join([output_data_folder, datdir])
methodnames = os.listdir(root)
print('Methods:', methodnames)


def plot(xmax, ds, attr, ylim=None, yextra='', psrate=1):
    epoches = np.arange(xmax + 1)
    for method in methodnames:
        cnt = np.zeros(xmax + 1)
        vals = np.zeros(xmax + 1)
        for i in os.listdir('/'.join([root, method])):
            fname = '/'.join([root, method, i])
            if fname[-4:] != '.txt':
                continue
            print(method, fname)
            op = OutputParser(fname)
            epoch, val = op.get(ds, attr, np.array)
            for j in range(len(epoch)):
                try:
                    cnt[epoch[j]] += 1
                    vals[epoch[j]] += val[j]
                except IndexError:
                    pass
        vals = np.divide(vals, cnt, out=-np.ones_like(cnt), where=(cnt != 0))
        index = np.argwhere(cnt != 0)
        plt.plot(epoches[index], vals[index])
    if ylim is not None:
        plt.ylim(*ylim)
    plt.legend(methodnames)
    plt.title('{} {} average {}'.format(dsname, ds, attr))
    plt.ylabel(attr + yextra)
    plt.xlabel('epoch')
    plt.savefig('{}/image/{}_{}_{}'.format(output_data_folder, dsname, ds, re.sub('/', '_', attr)))
    plt.show()

if datdir == 'cifar_172k':
    plot(40, 'train', 'nfe')
    plot(40, 'test', 'nfe')
    plot(40, 'test', 'acc', ylim=(0.45, 0.65))
    plot(40, 'train', 'time', yextra=' (min)')
    plot(40, 'train', 'loss')
    plot(40, 'test', 'loss', ylim=(1, 2))

if datdir == 'mnist_84k':
    plot(40, 'train', 'nfe', ylim=(35, 120))
    plot(40, 'test', 'nfe')
    plot(40, 'test', 'acc', ylim=(0.93, 0.99))
    plot(40, 'train', 'time', ylim=(0, 200), yextra=' (min)')
    plot(40, 'train', 'loss', ylim=(0, 0.03))
    plot(40, 'test', 'loss', ylim=(0, 0.25))

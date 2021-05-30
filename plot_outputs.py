import os
import re

from misc import *
from deprecated.output_parser import OutputParser

output_data_folder = './hbnode_data'
datdir = 'mnist_84k'
datdir = 'cifar_172k'
dsname = datdir[:5].upper()
root = '/'.join([output_data_folder, datdir])
methodnames = os.listdir(root)
print('Methods:', methodnames)


def plot(xmax, ds, attr, ylim=None, yextra='', psrate=1, ax=plt):
    epoches = np.arange(xmax + 1)
    for method in methodnames:
        cnt = np.zeros(xmax + 1)
        vals = np.zeros(xmax + 1)
        top = []
        bot = []
        fnames = os.listdir('/'.join([root, method]))
        for i in fnames:
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
            top.append(np.max(val))
            bot.append(np.min(val))
        print('{} & {:.1f} $\\pm$ {:.1f}'.format(method, 100*np.mean(top), 100*np.std(top)))
        print('{} & {:.3f} $\\pm$ {:.3f}'.format(method, np.mean(bot), np.std(bot)))
        vals = np.divide(vals, cnt, out=-np.ones_like(cnt), where=(cnt != 0))
        index = np.argwhere(cnt != 0)
        ax.plot(epoches[index], vals[index])
    if ylim is not None:
        ax.ylim(*ylim)
    ax.legend(methodnames)
    ax.title('{} {} average {}'.format(dsname, ds, attr))
    ax.ylabel(attr + yextra)
    ax.xlabel('epoch')
    ax.savefig('{}/image/{}_{}_{}.svg'.format(output_data_folder, dsname, ds, re.sub('/', '_', attr)))
    ax.show()


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

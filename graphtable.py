import os
import re

from misc import *
from output_parser import OutputParser

output_data_folder = './hbnode_data'
datdir = 'mnist_84k'
datdir = 'cifar_172k'
dsname = datdir[:5].upper()
methodnames = ['sonode', 'anode', 'ghbnode', 'hbnode', 'node']


def printstats(xmax, ds, attr, datdir='mnist_84k', **kwargs):
    out = []
    root = '/'.join([output_data_folder, datdir])
    for method in methodnames:
        top = []
        bot = []
        m = []
        try:
            fnames = os.listdir('/'.join([root, method]))
            for i in fnames:
                fname = '/'.join([root, method, i])
                if fname[-4:] != '.txt':
                    continue
                op = OutputParser(fname)
                epoch, val = op.get(ds, attr, np.array)
                top.append(np.max(val))
                bot.append(np.min(val))
                m.append(np.mean(val))
            if attr == 'acc':
                out.append('{:.1f} $\\pm$ {:.1f}'.format(100*np.mean(top), 100*np.std(top)))
            elif attr == 'loss':
                out.append('{:.3f} $\\pm$ {:.3f}'.format(np.mean(bot), np.std(bot)))
            else:
                out.append('{:.1f} $\\pm$ {:.1f}'.format(np.mean(m), np.std(m)))
        except FileNotFoundError:
            out.append(' ')
    return out


outs = [['SONODE', 'ANODE', 'GHBNODE', 'HBNODE', 'NODE'],
printstats(40, 'train', 'nfe'),
printstats(40, 'test', 'acc'),
printstats(40, 'test', 'loss'),
printstats(40, 'train', 'nfe', 'cifar_172k'),
printstats(40, 'test', 'acc', 'cifar_172k'),
printstats(40, 'test', 'loss', 'cifar_172k'),]
outs = list(zip(*outs))

print(outs)
for i in outs:
    print(' & '.join(i), ' \\\\')
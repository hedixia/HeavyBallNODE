import re
import numpy as np


class OutputParser:
    def __init__(self, fname, **kwargs):
        self.__dict__.update(kwargs)
        self.extra = []
        self.train = dict()
        self.test = dict()
        self.parse(fname)

    def parse(self, fname):
        rfile = open(fname, 'r')
        rlines = rfile.readlines()
        for i in rlines:
            self.parseline(i)

    def parseline(self, row):
        row = row.split('||')
        if len(row) <= 1:
            self.extra.append(row[0])
        else:
            ds, vals = row[:2]
            vals = vals.split(',')
            vals = [i.split(':') for i in vals]
            vals = [i for i in vals if len(i) == 2]
            for iter in vals:
                iter[0] = re.sub(' ', '', iter[0])
                iter[1] = re.sub('[^0-9.]', '', iter[1])
            epoch = int(vals[0][1])
            for (key, val) in vals:
                if val == '':
                    continue
                if ds[:2] == 'Tr':
                    self.train[(epoch, key)] = np.float(val)
                else:
                    self.test[(epoch, key)] = np.float(val)

    def get(self, ds, key, otype=None):
        outdict = {}
        selfdict = getattr(self, ds)
        for i in selfdict:
            if i[1] == key:
                outdict[i[0]] = selfdict[i]
        if otype is not None:
            key = sorted(outdict)
            val = [outdict[i] for i in key]
            try:
                outdict = (otype(key), otype(val))
            except TypeError:
                outdict = (key, val)
        return outdict

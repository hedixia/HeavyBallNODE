import numpy as np
import os
import csv

folder = 'imgdat'
dataset = 'mnist'

outfile = open(os.path.join(folder, 'outdat.csv'), 'w')
outcsv = csv.writer(outfile)
outcsv.writerow(['method', 'file', 'train/test', 'epoch', 'loss', 'acc', 'total_nfe', 'forward_nfe', 'time/epoch(s)',
                 'elapsed_time(min)'])
for method in os.listdir(os.path.join(folder, dataset)):
    method_path = os.path.join(folder, dataset, method)
    for file in os.listdir(method_path):
        print(method, file)
        with open(os.path.join(method_path, file), 'r') as infile:
            csvreader = csv.reader(infile)
            listdat = list(csvreader)
            for rows in listdat:
                if 'tensor' in rows[2]:
                    rows[2] = rows[2][7:-18]
                if rows[4] == '':
                    rows[4] = rows[3]
                    rows = ['test', *rows]
                else:
                    rows = ['train', *rows]
                rows = [method, file, *rows]
                outcsv.writerow(rows)

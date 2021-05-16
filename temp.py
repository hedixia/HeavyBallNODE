import os
import csv

rec_names = ["model", "test#", "train/test", "iter", "loss", "acc", "forwardnfe", "backwardnfe", "time/iter",
             "time_elapsed"]

root = "results/cifar/tol"
outfile = open('imgdat/cifar_5.csv', 'w')
csvwriter = csv.writer(outfile)
csvwriter.writerow(rec_names)

for model in os.listdir(root):
    fnames = os.listdir(os.path.join(root, model))
    fnames = [i for i in fnames if '5' in i]
    #fnames = sorted(fnames, key=lambda x: int(x[-5]))
    for fname in fnames:
        testnum = fname[-5]
        with open(os.path.join(root, model, fname), 'r') as infile:
            reader = csv.reader(infile)
            for obs in reader:
                dat = list(obs)
                print(fname, dat)
                if dat[4] == '':
                    dat[4] = 0
                    dat = ['test', *dat]
                else:
                    forward_nfe = float(dat[4])
                    backward_nfe = float(dat[3]) - forward_nfe
                    dat[3] = forward_nfe
                    dat[4] = backward_nfe
                    dat = ['train', *dat]
                dat = [model, testnum, *dat]
                csvwriter.writerow(dat)

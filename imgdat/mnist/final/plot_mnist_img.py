# -*- coding: utf-8 -*-
"""
Plot HBNODE MNIST
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import glob

methods = ['NODE', 'ANODE','SONODE','HBNODE','GHBNODE',]
colors = ['b','y','g','r','m']
path = '.'
all_files = glob.glob(path + "/*.csv")
print(all_files)
li = []

for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)

df = pd.concat(li, axis=0, ignore_index=True).sort_values('iter')
avedf = df.groupby(['model', 'train/test', 'iter']).mean().groupby(['model', 'train/test'])
avedict = dict(tuple(avedf))
#node_test_acc = avedf[(avedf['model']=='node') & (avedf['train/test'] == 'test')]
#print(node_test_acc)

'''
sonode = df[df['model'] == 'sonode']
anode = df[df['model'] == 'anode']
ghbnode = df[df['model'] == 'ghbnode']
hbnode = df[df['model'] == 'hbnode']
node = df[df['model'] == 'node']

sonode_train = sonode[sonode['train/test'] == 'train']
sonode_test = sonode[sonode['train/test'] == 'test']
anode_train = anode[anode['train/test'] == 'train']
anode_test = anode[anode['train/test'] == 'test']
ghbnode_train = ghbnode[ghbnode['train/test'] == 'train']
ghbnode_test = ghbnode[ghbnode['train/test'] == 'test']
hbnode_train = hbnode[hbnode['train/test'] == 'train']
hbnode_test = hbnode[hbnode['train/test'] == 'test']
node_train = node[node['train/test'] == 'train']
node_test = node[node['train/test'] == 'test']

sonode_train_1 = sonode_train[sonode_train['test#'] == '_']
sonode_train_2 = sonode_train[sonode_train['test#'] == '_']
sonode_train_3 = sonode_train[sonode_train['test#'] == '_']
sonode_test_1 = sonode_test[sonode_test['test#'] == '_']
sonode_test_2 = sonode_test[sonode_test['test#'] == '_']
sonode_test_3 = sonode_test[sonode_test['test#'] == '_']

anode_train_1 = anode_train[anode_train['test#'] == '_']
anode_train_2 = anode_train[anode_train['test#'] == '_']
anode_train_3 = anode_train[anode_train['test#'] == '_']
anode_test_1 = anode_test[anode_test['test#'] == '_']
anode_test_2 = anode_test[anode_test['test#'] == '_']
anode_test_3 = anode_test[anode_test['test#'] == '_']

ghbnode_train_1 = ghbnode_train[ghbnode_train['test#'] == '_']
ghbnode_train_2 = ghbnode_train[ghbnode_train['test#'] == '_']
ghbnode_train_3 = ghbnode_train[ghbnode_train['test#'] == '_']
ghbnode_test_1 = ghbnode_test[ghbnode_test['test#'] == '_']
ghbnode_test_2 = ghbnode_test[ghbnode_test['test#'] == '_']
ghbnode_test_3 = ghbnode_test[ghbnode_test['test#'] == '_']

hbnode_train_1 = hbnode_train[hbnode_train['test#'] == '_']
hbnode_train_2 = hbnode_train[hbnode_train['test#'] == '_']
hbnode_train_3 = hbnode_train[hbnode_train['test#'] == '_']
hbnode_test_1 = hbnode_test[hbnode_test['test#'] == '_']
hbnode_test_2 = hbnode_test[hbnode_test['test#'] == '_']
hbnode_test_3 = hbnode_test[hbnode_test['test#'] == '_']

node_train_1 = node_train[node_train['test#'] == '_']
node_train_2 = node_train[node_train['test#'] == '_']
node_train_3 = node_train[node_train['test#'] == '_']
node_test_1 = node_test[node_test['test#'] == '_']
node_test_2 = node_test[node_test['test#'] == '_']
node_test_3 = node_test[node_test['test#'] == '_']


############################### Loss ##########################################
sonode_train_1_loss = np.array(sonode_train_1['loss'])
sonode_train_2_loss = np.array(sonode_train_2['loss'])
sonode_train_3_loss = np.array(sonode_train_3['loss'])
sonode_test_1_loss = np.array(sonode_test_1['loss'])
sonode_test_2_loss = np.array(sonode_test_2['loss'])
sonode_test_3_loss = np.array(sonode_test_3['loss'])
min_len = min(len(sonode_train_1_loss), len(sonode_train_2_loss), len(sonode_train_3_loss))
sonode_train_loss_avg = (sonode_train_1_loss[:min_len] + sonode_train_2_loss[:min_len] + sonode_train_3_loss[:min_len])/3.
sonode_test_loss_avg = (sonode_test_1_loss[:min_len] + sonode_test_2_loss[:min_len] + sonode_test_3_loss[:min_len])/3.

anode_train_1_loss = np.array(anode_train_1['loss'])
anode_train_2_loss = np.array(anode_train_2['loss'])
anode_train_3_loss = np.array(anode_train_3['loss'])
anode_test_1_loss = np.array(anode_test_1['loss'])
anode_test_2_loss = np.array(anode_test_2['loss'])
anode_test_3_loss = np.array(anode_test_3['loss'])
min_len = min(len(anode_train_1_loss), len(anode_train_2_loss), len(anode_train_3_loss))
anode_train_loss_avg = (anode_train_1_loss[:min_len] + anode_train_2_loss[:min_len] + anode_train_3_loss[:min_len])/3.
anode_test_loss_avg = (anode_test_1_loss[:min_len] + anode_test_2_loss[:min_len] + anode_test_3_loss[:min_len])/3.

hbnode_train_1_loss = np.array(hbnode_train_1['loss'])
hbnode_train_2_loss = np.array(hbnode_train_2['loss'])
hbnode_train_3_loss = np.array(hbnode_train_3['loss'])
hbnode_test_1_loss = np.array(hbnode_test_1['loss'])
hbnode_test_2_loss = np.array(hbnode_test_2['loss'])
hbnode_test_3_loss = np.array(hbnode_test_3['loss'])
min_len = min(len(hbnode_train_1_loss), len(hbnode_train_2_loss), len(hbnode_train_3_loss))
hbnode_train_loss_avg = (hbnode_train_1_loss[:min_len] + hbnode_train_2_loss[:min_len] + hbnode_train_3_loss[:min_len])/3.
hbnode_test_loss_avg = (hbnode_test_1_loss[:min_len] + hbnode_test_2_loss[:min_len] + hbnode_test_3_loss[:min_len])/3.

ghbnode_train_1_loss = np.array(ghbnode_train_1['loss'])
ghbnode_train_2_loss = np.array(ghbnode_train_2['loss'])
ghbnode_train_3_loss = np.array(ghbnode_train_3['loss'])
ghbnode_test_1_loss = np.array(ghbnode_test_1['loss'])
ghbnode_test_2_loss = np.array(ghbnode_test_2['loss'])
ghbnode_test_3_loss = np.array(ghbnode_test_3['loss'])
min_len = min(len(ghbnode_train_1_loss), len(ghbnode_train_2_loss), len(ghbnode_train_3_loss))
ghbnode_train_loss_avg = (ghbnode_train_1_loss[:min_len] + ghbnode_train_2_loss[:min_len] + ghbnode_train_3_loss[:min_len])/3.
ghbnode_test_loss_avg = (ghbnode_test_1_loss[:min_len] + ghbnode_test_2_loss[:min_len] + ghbnode_test_3_loss[:min_len])/3.

node_train_1_loss = np.array(node_train_1['loss'])
node_test_1_loss = np.array(node_test_1['loss'])
min_len = len(node_train_1_loss)
node_train_loss_avg = (node_train_1_loss[:min_len])/1.
node_test_loss_avg = (node_test_1_loss[:min_len])/1.



################################## Acc ########################################
sonode_train_1_acc = np.array(sonode_train_1['acc'])
sonode_train_2_acc = np.array(sonode_train_2['acc'])
sonode_train_3_acc = np.array(sonode_train_3['acc'])
sonode_test_1_acc = np.array(sonode_test_1['acc'])
sonode_test_2_acc = np.array(sonode_test_2['acc'])
sonode_test_3_acc = np.array(sonode_test_3['acc'])
min_len = min(len(sonode_train_1_acc), len(sonode_train_2_acc), len(sonode_train_3_acc))
sonode_train_acc_avg = (sonode_train_1_acc[:min_len] + sonode_train_2_acc[:min_len] + sonode_train_3_acc[:min_len])/3.
sonode_test_acc_avg = (sonode_test_1_acc[:min_len] + sonode_test_2_acc[:min_len] + sonode_test_3_acc[:min_len])/3.

anode_train_1_acc = np.array(anode_train_1['acc'])
anode_train_2_acc = np.array(anode_train_2['acc'])
anode_train_3_acc = np.array(anode_train_3['acc'])
anode_test_1_acc = np.array(anode_test_1['acc'])
anode_test_2_acc = np.array(anode_test_2['acc'])
anode_test_3_acc = np.array(anode_test_3['acc'])
min_len = min(len(anode_train_1_acc), len(anode_train_2_acc), len(anode_train_3_acc))
anode_train_acc_avg = (anode_train_1_acc[:min_len] + anode_train_2_acc[:min_len] + anode_train_3_acc[:min_len])/3.
anode_test_acc_avg = (anode_test_1_acc[:min_len] + anode_test_2_acc[:min_len] + anode_test_3_acc[:min_len])/3.

hbnode_train_1_acc = np.array(hbnode_train_1['acc'])
hbnode_train_2_acc = np.array(hbnode_train_2['acc'])
hbnode_train_3_acc = np.array(hbnode_train_3['acc'])
hbnode_test_1_acc = np.array(hbnode_test_1['acc'])
hbnode_test_2_acc = np.array(hbnode_test_2['acc'])
hbnode_test_3_acc = np.array(hbnode_test_3['acc'])
min_len = min(len(hbnode_train_1_acc), len(hbnode_train_2_acc), len(hbnode_train_3_acc))
hbnode_train_acc_avg = (hbnode_train_1_acc[:min_len] + hbnode_train_2_acc[:min_len] + hbnode_train_3_acc[:min_len])/3.
hbnode_test_acc_avg = (hbnode_test_1_acc[:min_len] + hbnode_test_2_acc[:min_len] + hbnode_test_3_acc[:min_len])/3.

ghbnode_train_1_acc = np.array(ghbnode_train_1['acc'])
ghbnode_train_2_acc = np.array(ghbnode_train_2['acc'])
ghbnode_train_3_acc = np.array(ghbnode_train_3['acc'])
ghbnode_test_1_acc = np.array(ghbnode_test_1['acc'])
ghbnode_test_2_acc = np.array(ghbnode_test_2['acc'])
ghbnode_test_3_acc = np.array(ghbnode_test_3['acc'])
min_len = min(len(ghbnode_train_1_acc), len(ghbnode_train_2_acc), len(ghbnode_train_3_acc))
ghbnode_train_acc_avg = (ghbnode_train_1_acc[:min_len] + ghbnode_train_2_acc[:min_len] + ghbnode_train_3_acc[:min_len])/3.
ghbnode_test_acc_avg = (ghbnode_test_1_acc[:min_len] + ghbnode_test_2_acc[:min_len] + ghbnode_test_3_acc[:min_len])/3.

node_train_1_acc = np.array(node_train_1['acc'])
node_test_1_acc = np.array(node_test_1['acc'])
min_len = len(node_train_1_acc)
node_train_acc_avg = (node_train_1_acc[:min_len])/1.
node_test_acc_avg = (node_test_1_acc[:min_len])/1.


############################# Forward NFE #####################################
sonode_train_1_fnfe = np.array(sonode_train_1['forwardnfe'])
sonode_train_2_fnfe = np.array(sonode_train_2['forwardnfe'])
sonode_train_3_fnfe = np.array(sonode_train_3['forwardnfe'])
sonode_test_1_fnfe = np.array(sonode_test_1['forwardnfe'])
sonode_test_2_fnfe = np.array(sonode_test_2['forwardnfe'])
sonode_test_3_fnfe = np.array(sonode_test_3['forwardnfe'])
min_len = min(len(sonode_train_1_fnfe), len(sonode_train_2_fnfe), len(sonode_train_3_fnfe))
sonode_train_fnfe_avg = (sonode_train_1_fnfe[:min_len] + sonode_train_2_fnfe[:min_len] + sonode_train_3_fnfe[:min_len])/3.
sonode_test_fnfe_avg = (sonode_test_1_fnfe[:min_len] + sonode_test_2_fnfe[:min_len] + sonode_test_3_fnfe[:min_len])/3.

anode_train_1_fnfe = np.array(anode_train_1['forwardnfe'])
anode_train_2_fnfe = np.array(anode_train_2['forwardnfe'])
anode_train_3_fnfe = np.array(anode_train_3['forwardnfe'])
anode_test_1_fnfe = np.array(anode_test_1['forwardnfe'])
anode_test_2_fnfe = np.array(anode_test_2['forwardnfe'])
anode_test_3_fnfe = np.array(anode_test_3['forwardnfe'])
min_len = min(len(anode_train_1_fnfe), len(anode_train_2_fnfe), len(anode_train_3_fnfe))
anode_train_fnfe_avg = (anode_train_1_fnfe[:min_len] + anode_train_2_fnfe[:min_len] + anode_train_3_fnfe[:min_len])/3.
anode_test_fnfe_avg = (anode_test_1_fnfe[:min_len] + anode_test_2_fnfe[:min_len] + anode_test_3_fnfe[:min_len])/3.

hbnode_train_1_fnfe = np.array(hbnode_train_1['forwardnfe'])
hbnode_train_2_fnfe = np.array(hbnode_train_2['forwardnfe'])
hbnode_train_3_fnfe = np.array(hbnode_train_3['forwardnfe'])
hbnode_test_1_fnfe = np.array(hbnode_test_1['forwardnfe'])
hbnode_test_2_fnfe = np.array(hbnode_test_2['forwardnfe'])
hbnode_test_3_fnfe = np.array(hbnode_test_3['forwardnfe'])
min_len = min(len(hbnode_train_1_fnfe), len(hbnode_train_2_fnfe), len(hbnode_train_3_fnfe))
hbnode_train_fnfe_avg = (hbnode_train_1_fnfe[:min_len] + hbnode_train_2_fnfe[:min_len] + hbnode_train_3_fnfe[:min_len])/3.
hbnode_test_fnfe_avg = (hbnode_test_1_fnfe[:min_len] + hbnode_test_2_fnfe[:min_len] + hbnode_test_3_fnfe[:min_len])/3.

ghbnode_train_1_fnfe = np.array(ghbnode_train_1['forwardnfe'])
ghbnode_train_2_fnfe = np.array(ghbnode_train_2['forwardnfe'])
ghbnode_train_3_fnfe = np.array(ghbnode_train_3['forwardnfe'])
ghbnode_test_1_fnfe = np.array(ghbnode_test_1['forwardnfe'])
ghbnode_test_2_fnfe = np.array(ghbnode_test_2['forwardnfe'])
ghbnode_test_3_fnfe = np.array(ghbnode_test_3['forwardnfe'])
min_len = min(len(ghbnode_train_1_fnfe), len(ghbnode_train_2_fnfe), len(ghbnode_train_3_fnfe))
ghbnode_train_fnfe_avg = (ghbnode_train_1_fnfe[:min_len] + ghbnode_train_2_fnfe[:min_len] + ghbnode_train_3_fnfe[:min_len])/3.
ghbnode_test_fnfe_avg = (ghbnode_test_1_fnfe[:min_len] + ghbnode_test_2_fnfe[:min_len] + ghbnode_test_3_fnfe[:min_len])/3.

node_train_1_fnfe = np.array(node_train_1['forwardnfe'])
node_test_1_fnfe = np.array(node_test_1['forwardnfe'])
min_len = len(node_train_1_fnfe)
node_train_fnfe_avg = (node_train_1_fnfe[:min_len])/1.
node_test_fnfe_avg = (node_test_1_fnfe[:min_len])/1.



############################# Backward NFE ####################################
sonode_train_1_bnfe = np.array(sonode_train_1['backwardnfe'])
sonode_train_2_bnfe = np.array(sonode_train_2['backwardnfe'])
sonode_train_3_bnfe = np.array(sonode_train_3['backwardnfe'])
sonode_test_1_bnfe = np.array(sonode_test_1['backwardnfe'])
sonode_test_2_bnfe = np.array(sonode_test_2['backwardnfe'])
sonode_test_3_bnfe = np.array(sonode_test_3['backwardnfe'])
min_len = min(len(sonode_train_1_bnfe), len(sonode_train_2_bnfe), len(sonode_train_3_bnfe))
sonode_train_bnfe_avg = (sonode_train_1_bnfe[:min_len] + sonode_train_2_bnfe[:min_len] + sonode_train_3_bnfe[:min_len])/3.
sonode_test_bnfe_avg = (sonode_test_1_bnfe[:min_len] + sonode_test_2_bnfe[:min_len] + sonode_test_3_bnfe[:min_len])/3.

anode_train_1_bnfe = np.array(anode_train_1['backwardnfe'])
anode_train_2_bnfe = np.array(anode_train_2['backwardnfe'])
anode_train_3_bnfe = np.array(anode_train_3['backwardnfe'])
anode_test_1_bnfe = np.array(anode_test_1['backwardnfe'])
anode_test_2_bnfe = np.array(anode_test_2['backwardnfe'])
anode_test_3_bnfe = np.array(anode_test_3['backwardnfe'])
min_len = min(len(anode_train_1_bnfe), len(anode_train_2_bnfe), len(anode_train_3_bnfe))
anode_train_bnfe_avg = (anode_train_1_bnfe[:min_len] + anode_train_2_bnfe[:min_len] + anode_train_3_bnfe[:min_len])/3.
anode_test_bnfe_avg = (anode_test_1_bnfe[:min_len] + anode_test_2_bnfe[:min_len] + anode_test_3_bnfe[:min_len])/3.

hbnode_train_1_bnfe = np.array(hbnode_train_1['backwardnfe'])
hbnode_train_2_bnfe = np.array(hbnode_train_2['backwardnfe'])
hbnode_train_3_bnfe = np.array(hbnode_train_3['backwardnfe'])
hbnode_test_1_bnfe = np.array(hbnode_test_1['backwardnfe'])
hbnode_test_2_bnfe = np.array(hbnode_test_2['backwardnfe'])
hbnode_test_3_bnfe = np.array(hbnode_test_3['backwardnfe'])
min_len = min(len(hbnode_train_1_bnfe), len(hbnode_train_2_bnfe), len(hbnode_train_3_bnfe))
hbnode_train_bnfe_avg = (hbnode_train_1_bnfe[:min_len] + hbnode_train_2_bnfe[:min_len] + hbnode_train_3_bnfe[:min_len])/3.
hbnode_test_bnfe_avg = (hbnode_test_1_bnfe[:min_len] + hbnode_test_2_bnfe[:min_len] + hbnode_test_3_bnfe[:min_len])/3.

ghbnode_train_1_bnfe = np.array(ghbnode_train_1['backwardnfe'])
ghbnode_train_2_bnfe = np.array(ghbnode_train_2['backwardnfe'])
ghbnode_train_3_bnfe = np.array(ghbnode_train_3['backwardnfe'])
ghbnode_test_1_bnfe = np.array(ghbnode_test_1['backwardnfe'])
ghbnode_test_2_bnfe = np.array(ghbnode_test_2['backwardnfe'])
ghbnode_test_3_bnfe = np.array(ghbnode_test_3['backwardnfe'])
min_len = min(len(ghbnode_train_1_bnfe), len(ghbnode_train_2_bnfe), len(ghbnode_train_3_bnfe))
ghbnode_train_bnfe_avg = (ghbnode_train_1_bnfe[:min_len] + ghbnode_train_2_bnfe[:min_len] + ghbnode_train_3_bnfe[:min_len])/3.
ghbnode_test_bnfe_avg = (ghbnode_test_1_bnfe[:min_len] + ghbnode_test_2_bnfe[:min_len] + ghbnode_test_3_bnfe[:min_len])/3.

node_train_1_bnfe = np.array(node_train_1['backwardnfe'])
node_test_1_bnfe = np.array(node_test_1['backwardnfe'])
min_len = len(node_train_1_bnfe)
node_train_bnfe_avg = (node_train_1_bnfe[:min_len])/1.
node_test_bnfe_avg = (node_test_1_bnfe[:min_len])/1.




################################## Time #######################################
sonode_train_1_time = np.array(sonode_train_1['time/iter'])
sonode_train_2_time = np.array(sonode_train_2['time/iter'])
sonode_train_3_time = np.array(sonode_train_3['time/iter'])
sonode_test_1_time = np.array(sonode_test_1['time/iter'])
sonode_test_2_time = np.array(sonode_test_2['time/iter'])
sonode_test_3_time = np.array(sonode_test_3['time/iter'])
min_len = min(len(sonode_train_1_time), len(sonode_train_2_time), len(sonode_train_3_time))
sonode_train_time_avg = (sonode_train_1_time[:min_len] + sonode_train_2_time[:min_len] + sonode_train_3_time[:min_len])/3.
sonode_test_time_avg = (sonode_test_1_time[:min_len] + sonode_test_2_time[:min_len] + sonode_test_3_time[:min_len])/3.

anode_train_1_time = np.array(anode_train_1['time/iter'])
anode_train_2_time = np.array(anode_train_2['time/iter'])
anode_train_3_time = np.array(anode_train_3['time/iter'])
anode_test_1_time = np.array(anode_test_1['time/iter'])
anode_test_2_time = np.array(anode_test_2['time/iter'])
anode_test_3_time = np.array(anode_test_3['time/iter'])
min_len = min(len(anode_train_1_time), len(anode_train_2_time), len(anode_train_3_time))
anode_train_time_avg = (anode_train_1_time[:min_len] + anode_train_2_time[:min_len] + anode_train_3_time[:min_len])/3.
anode_test_time_avg = (anode_test_1_time[:min_len] + anode_test_2_time[:min_len] + anode_test_3_time[:min_len])/3.

hbnode_train_1_time = np.array(hbnode_train_1['time/iter'])
hbnode_train_2_time = np.array(hbnode_train_2['time/iter'])
hbnode_train_3_time = np.array(hbnode_train_3['time/iter'])
hbnode_test_1_time = np.array(hbnode_test_1['time/iter'])
hbnode_test_2_time = np.array(hbnode_test_2['time/iter'])
hbnode_test_3_time = np.array(hbnode_test_3['time/iter'])
min_len = min(len(hbnode_train_1_time), len(hbnode_train_2_time), len(hbnode_train_3_time))
hbnode_train_time_avg = (hbnode_train_1_time[:min_len] + hbnode_train_2_time[:min_len] + hbnode_train_3_time[:min_len])/3.
hbnode_test_time_avg = (hbnode_test_1_time[:min_len] + hbnode_test_2_time[:min_len] + hbnode_test_3_time[:min_len])/3.

ghbnode_train_1_time = np.array(ghbnode_train_1['time/iter'])
ghbnode_train_2_time = np.array(ghbnode_train_2['time/iter'])
ghbnode_train_3_time = np.array(ghbnode_train_3['time/iter'])
ghbnode_test_1_time = np.array(ghbnode_test_1['time/iter'])
ghbnode_test_2_time = np.array(ghbnode_test_2['time/iter'])
ghbnode_test_3_time = np.array(ghbnode_test_3['time/iter'])
min_len = min(len(ghbnode_train_1_time), len(ghbnode_train_2_time), len(ghbnode_train_3_time))
ghbnode_train_time_avg = (ghbnode_train_1_time[:min_len] + ghbnode_train_2_time[:min_len] + ghbnode_train_3_time[:min_len])/3.
ghbnode_test_time_avg = (ghbnode_test_1_time[:min_len] + ghbnode_test_2_time[:min_len] + ghbnode_test_3_time[:min_len])/3.

node_train_1_time = np.array(node_train_1['time/iter'])
node_test_1_time = np.array(node_test_1['time/iter'])
min_len = len(node_train_1_time)
node_train_time_avg = (node_train_1_time[:min_len])/1.
node_test_time_avg = (node_test_1_time[:min_len])/1.



################################## Cum Time ###################################
sonode_train_1_cum_time = np.array(sonode_train_1['time_elapsed'])*60
sonode_train_2_cum_time = np.array(sonode_train_2['time_elapsed'])*60
sonode_train_3_cum_time = np.array(sonode_train_3['time_elapsed'])*60
sonode_test_1_cum_time = np.array(sonode_test_1['time_elapsed'])*60
sonode_test_2_cum_time = np.array(sonode_test_2['time_elapsed'])*60
sonode_test_3_cum_time = np.array(sonode_test_3['time_elapsed'])*60
min_len = min(len(sonode_train_1_cum_time), len(sonode_train_2_cum_time), len(sonode_train_3_cum_time))
sonode_train_cum_time_avg = (sonode_train_1_cum_time[:min_len] + sonode_train_2_cum_time[:min_len] + sonode_train_3_cum_time[:min_len])/3.
sonode_test_cum_time_avg = (sonode_test_1_cum_time[:min_len] + sonode_test_2_cum_time[:min_len] + sonode_test_3_cum_time[:min_len])/3.

anode_train_1_cum_time = np.array(anode_train_1['time_elapsed'])*60
anode_train_2_cum_time = np.array(anode_train_2['time_elapsed'])*60
anode_train_3_cum_time = np.array(anode_train_3['time_elapsed'])*60
anode_test_1_cum_time = np.array(anode_test_1['time_elapsed'])*60
anode_test_2_cum_time = np.array(anode_test_2['time_elapsed'])*60
anode_test_3_cum_time = np.array(anode_test_3['time_elapsed'])*60
min_len = min(len(anode_train_1_cum_time), len(anode_train_2_cum_time), len(anode_train_3_cum_time))
anode_train_cum_time_avg = (anode_train_1_cum_time[:min_len] + anode_train_2_cum_time[:min_len] + anode_train_3_cum_time[:min_len])/3.
anode_test_cum_time_avg = (anode_test_1_cum_time[:min_len] + anode_test_2_cum_time[:min_len] + anode_test_3_cum_time[:min_len])/3.

hbnode_train_1_cum_time = np.array(hbnode_train_1['time_elapsed'])*60
hbnode_train_2_cum_time = np.array(hbnode_train_2['time_elapsed'])*60
hbnode_train_3_cum_time = np.array(hbnode_train_3['time_elapsed'])*60
hbnode_test_1_cum_time = np.array(hbnode_test_1['time_elapsed'])*60
hbnode_test_2_cum_time = np.array(hbnode_test_2['time_elapsed'])*60
hbnode_test_3_cum_time = np.array(hbnode_test_3['time_elapsed'])*60
min_len = min(len(hbnode_train_1_cum_time), len(hbnode_train_2_cum_time), len(hbnode_train_3_cum_time))
hbnode_train_cum_time_avg = (hbnode_train_1_cum_time[:min_len] + hbnode_train_2_cum_time[:min_len] + hbnode_train_3_cum_time[:min_len])/3.
hbnode_test_cum_time_avg = (hbnode_test_1_cum_time[:min_len] + hbnode_test_2_cum_time[:min_len] + hbnode_test_3_cum_time[:min_len])/3.

ghbnode_train_1_cum_time = np.array(ghbnode_train_1['time_elapsed'])*60
ghbnode_train_2_cum_time = np.array(ghbnode_train_2['time_elapsed'])*60
ghbnode_train_3_cum_time = np.array(ghbnode_train_3['time_elapsed'])*60
ghbnode_test_1_cum_time = np.array(ghbnode_test_1['time_elapsed'])*60
ghbnode_test_2_cum_time = np.array(ghbnode_test_2['time_elapsed'])*60
ghbnode_test_3_cum_time = np.array(ghbnode_test_3['time_elapsed'])*60
min_len = min(len(ghbnode_train_1_cum_time), len(ghbnode_train_2_cum_time), len(ghbnode_train_3_cum_time))
ghbnode_train_cum_time_avg = (ghbnode_train_1_cum_time[:min_len] + ghbnode_train_2_cum_time[:min_len] + ghbnode_train_3_cum_time[:min_len])/3.
ghbnode_test_cum_time_avg = (ghbnode_test_1_cum_time[:min_len] + ghbnode_test_2_cum_time[:min_len] + ghbnode_test_3_cum_time[:min_len])/3.

node_train_1_cum_time = np.array(node_train_1['time_elapsed'])*60
node_test_1_cum_time = np.array(node_test_1['time_elapsed'])*60
min_len = len(node_train_1_cum_time)
node_train_cum_time_avg = (node_train_1_cum_time[:min_len])/1.
node_test_cum_time_avg = (node_test_1_cum_time[:min_len])/1.
'''

plot_len = 10
'''
plt.figure(figsize=(12,12))
axes = plt.gca()
#axes.set_ylim([0.0015, 0.5])
#axes.set_ylim([0.0015, 0.4])
axes.tick_params(axis='x', labelsize=50)
axes.tick_params(axis='y', labelsize=50)
#axes.set_yscale('log')
plt.plot(node_train_loss_avg[:plot_len], linewidth=5, color='b', label = "NODE")
plt.plot(anode_train_loss_avg[:plot_len], linewidth=5, color='y', label = "ANODE")
#plt.plot(sonode_train_loss_avg[:plot_len], linewidth=5, color='g', label = "SONODE")
#plt.plot(hbnode_train_loss_avg[:plot_len], linewidth=5, color='r', label = "HBNODE")
plt.plot(hbnode_train_loss_avg[:plot_len], linewidth=5, color='y', label = "ANODE") ##
plt.plot(sonode_train_loss_avg[:plot_len], linewidth=5, color='g', label = "SONODE") ##
plt.plot(anode_train_loss_avg[:plot_len], linewidth=5, color='r', label = "HBNODE") ##
plt.plot(ghbnode_train_loss_avg[:plot_len], linewidth=5, color='m', label = "GHBNODE")
plt.xlabel("Epoch", fontsize=50)
plt.ylabel("Training loss", fontsize=50)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend(loc='upper right', fontsize=50)
plt.tight_layout()
plt.savefig('MNIST_train_loss.pdf')
plt.show()
'''
'''
plt.figure(figsize=(12,12))
axes = plt.gca()
axes.set_ylim([0.05, 0.3])
axes.tick_params(axis='x', labelsize=50)
axes.tick_params(axis='y', labelsize=50)
plt.plot(node_test_loss_avg[:plot_len], linewidth=5, color='b', label = "NODE")
plt.plot(anode_test_loss_avg[:plot_len], linewidth=5, color='y', label = "ANODE")
plt.plot(sonode_test_loss_avg[:plot_len], linewidth=5, color='g', label = "SONODE")
plt.plot(hbnode_test_loss_avg[:plot_len], linewidth=5, color='r', label = "HBNODE")
plt.plot(ghbnode_test_loss_avg[:plot_len], linewidth=5, color='m', label = "GHBNODE")
plt.xlabel("Epoch", fontsize=50)
plt.ylabel("Test loss", fontsize=50)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.legend(loc='bottom left', fontsize=50)
plt.tight_layout()
plt.savefig('MNIST_test_loss.pdf')
plt.show()
'''

plt.figure(figsize=(12,12))
axes = plt.gca()
axes.set_ylim([89, 98.5])
axes.tick_params(axis='x', labelsize=50)
axes.tick_params(axis='y', labelsize=50)
for i in range(5):
    data = np.array(avedict[(methods[i].lower(), 'test')]['acc'])
    plt.plot(data[:plot_len]*100, linewidth=5, color=colors[i], label = methods[i])
plt.xlabel("Epoch", fontsize=50)
plt.ylabel("Test acc (%)", fontsize=50)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.legend(loc='upper left', fontsize=50)
plt.tight_layout()
plt.savefig('MNIST_test_acc.pdf')
plt.show()


plt.figure(figsize=(12,12))
axes = plt.gca()
#axes.set_ylim([18, 60])
#axes.set_ylim([18, 45])
axes.tick_params(axis='x', labelsize=50)
axes.tick_params(axis='y', labelsize=50)
print(avedict.keys())
for i in range(5):
    data = np.array(avedict[(methods[i].lower(), 'train')]['forwardnfe'])
    plt.plot(data[:plot_len], linewidth=5, color=colors[i], label = methods[i])
plt.xlabel("Epoch", fontsize=50)
plt.ylabel("NFE (forward)", fontsize=50)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.legend(loc='bottom left', fontsize=50)
plt.tight_layout()
plt.savefig('MNIST_forward_nfe.pdf')
plt.show()


plt.figure(figsize=(12,12))
axes = plt.gca()
#axes.set_ylim([18, 80])
#axes.set_ylim([18, 45])
axes.tick_params(axis='x', labelsize=50)
axes.tick_params(axis='y', labelsize=50)
for i in range(5):
    data = np.array(avedict[(methods[i].lower(), 'train')]['backwardnfe'])
    plt.plot(data[:plot_len], linewidth=5, color=colors[i], label = methods[i])
plt.xlabel("Epoch", fontsize=50)
plt.ylabel("NFE (backward)", fontsize=50)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.legend(loc='bottom left', fontsize=50)
plt.tight_layout()
plt.savefig('MNIST_backward_nfe.pdf')
plt.show()
'''
plt.figure(figsize=(12,12))
axes = plt.gca()
#axes.set_ylim([18, 60])
axes.set_ylim([18, 45])
axes.tick_params(axis='x', labelsize=50)
axes.tick_params(axis='y', labelsize=50)
plt.plot(node_test_fnfe_avg[:plot_len], linewidth=5, color='b', label = "NODE")
plt.plot(anode_test_fnfe_avg[:plot_len], linewidth=5, color='y', label = "ANODE")
plt.plot(sonode_test_fnfe_avg[:plot_len], linewidth=5, color='g', label = "SONODE")
plt.plot(hbnode_test_fnfe_avg[:plot_len], linewidth=5, color='r', label = "HBNODE")
plt.plot(ghbnode_test_fnfe_avg[:plot_len], linewidth=5, color='m', label = "GHBNODE")
plt.xlabel("Epoch", fontsize=50)
plt.ylabel("NFE (test)", fontsize=50)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.legend(loc='bottom left', fontsize=50)
plt.tight_layout()
plt.savefig('MNIST_test_nfe.pdf')
plt.show()
'''
plt.figure(figsize=(12,12))
axes = plt.gca()
#axes.set_ylim([0, 15000])
#axes.set_ylim([0, 15])
#axes.set_ylim([0, 2.5])
axes.tick_params(axis='x', labelsize=50)
axes.tick_params(axis='y', labelsize=50)
for i in range(5):
    data = np.array(avedict[(methods[i].lower(), 'train')]['time_elapsed'])
    plt.plot(data[:plot_len]/1000. * 60., linewidth=5, color=colors[i], label = methods[i])
plt.xlabel("Epoch", fontsize=50)
plt.ylabel("Training time (x1000s)", fontsize=50)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
plt.legend(loc='upper left', fontsize=50)
plt.tight_layout()
plt.savefig('MNIST_training_time.pdf')
plt.show()


'''
plt.figure(figsize=(12,12))
axes = plt.gca()
#axes.set_ylim([5, 18])
axes.set_ylim([5, 14])
axes.tick_params(axis='x', labelsize=50)
axes.tick_params(axis='y', labelsize=50)
plt.plot(node_test_cum_time_avg[:plot_len]-node_train_cum_time_avg[:plot_len], linewidth=5, color='b', label = "NODE")
plt.plot(anode_test_cum_time_avg[:plot_len]-anode_train_cum_time_avg[:plot_len], linewidth=5, color='y', label = "ANODE")
plt.plot(sonode_test_cum_time_avg[:plot_len]-sonode_train_cum_time_avg[:plot_len], linewidth=5, color='g', label = "SONODE")
plt.plot(hbnode_test_cum_time_avg[:plot_len]-hbnode_train_cum_time_avg[:plot_len], linewidth=5, color='r', label = "HBNODE")
plt.plot(ghbnode_test_cum_time_avg[:plot_len]-ghbnode_train_cum_time_avg[:plot_len], linewidth=5, color='m', label = "GHBNODE")
plt.xlabel("Epoch", fontsize=50)
plt.ylabel("Test time (s)", fontsize=50)
plt.grid(b=True, which='major', color='#666666', linestyle='-')
plt.minorticks_on()
plt.grid(b=True, which='minor', color='#999999', linestyle='-', alpha=0.2)
#plt.legend(loc='bottom left', fontsize=50)
plt.tight_layout()
plt.savefig('MNIST_test_time.pdf')
plt.show()
'''
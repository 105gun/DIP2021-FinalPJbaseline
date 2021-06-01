import os

lst = range(1,183)
tp = 2  # 0: train 1: val 2: test

if tp == 0:
    name = './filelists/ShanghaiA_train.txt'
if tp == 1:
    name = './filelists/ShanghaiA_val.txt'
if tp == 2:
    name = './filelists/ShanghaiA_test.txt'

with open(name, 'w') as f:
    for i in lst:
        if tp == 2:
            f.write('C:/Users/ooo69/CrowdCountingDatasets/ShanghaiTechPartA/fullresolution/origin/Test/IMG_' + str(i) + '.jpg\n')
        else:
            f.write('C:/Users/ooo69/CrowdCountingDatasets/ShanghaiTechPartA/fullresolution/origin/Train/IMG_' + str(i) + '.jpg\n')

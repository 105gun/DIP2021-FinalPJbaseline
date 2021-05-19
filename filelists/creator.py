import os

lst=range(1,10)
with open('./filelists/ShanghaiA_test.txt', 'w') as f:
    for i in lst:
        f.write('C:/Users/ooo69/CrowdCountingDatasets/ShanghaiTechPartA/fullresolution/origin/Test/IMG_' + str(i) + '.jpg\n')
        # f.write('C:/Users/ooo69/CrowdCountingDatasets/ShanghaiTechPartA/fullresolution/origin/Train/IMG_' + str(i) + '.jpg\n')

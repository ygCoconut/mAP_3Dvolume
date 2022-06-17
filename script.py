import sys
import numpy as np
import h5py

opt = sys.argv[1]

if opt == '0':
    # python script.py 0 demo_data/rat_gt_stats.csv
    fn = sys.argv[2]
    aa = open(fn)
    bb = aa.readlines()
    result = np.zeros([len(bb)-1,2])
    tt = {'small':0,'medium':1,'large':2}
    for i in range(1,len(bb)):
        line = bb[i].replace('\n','').split(',')
        result[i-1, 0] = int(line[0])
        result[i-1, 1] = tt[line[-1]]
    aa.close()
    np.savetxt(fn[:fn.rfind('.')]+'_group.txt', result, '%d')
elif opt == '1': # generate lucchi grouping file
    gt = np.array(h5py.File('demo_data/lucchi_gt_test.h5','r')['main'])
    uid = np.unique(gt)
    uid = uid[uid>0]
    label = np.zeros(len(uid))
    label[::2] = 1
    label[::3] = 2
    np.savetxt('demo_data/lucchi_test_group.txt', np.vstack([uid,label]).T, '%d')

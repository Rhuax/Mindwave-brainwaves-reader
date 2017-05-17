import numpy as np


dataset = np.genfromtxt('keirnAunonDataset.csv', delimiter=',', dtype=np.float32)



nof_seq=np.zeros(5)


for row in dataset:
    if np.array_equal(row[-5:],np.array([1,0,0,0,0],dtype=np.float32)):
        nof_seq[0]+=1
    elif np.array_equal(row[-5:],np.array([0,1,0,0,0],dtype=np.float32)):
        nof_seq[1]+=1
    elif np.array_equal(row[-5:], np.array([0, 0, 1, 0, 0], dtype=np.float32)):
        nof_seq[2] += 1
    elif np.array_equal(row[-5:], np.array([0, 0, 0,1, 0], dtype=np.float32)):
        nof_seq[3] += 1
    elif np.array_equal(row[-5:], np.array([0, 0, 0, 0, 1], dtype=np.float32)):
        nof_seq[4] += 1



print('somma delle istanze: '+str(np.sum(nof_seq)))
nof_seq/=2500
print(nof_seq)


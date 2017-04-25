from cdtw import pydtw

import os
import numpy as np
import itertools
np.set_printoptions(linewidth=250)

names = ['stefano','roberta','angelo','claudio','gianluca','michel',
       'asia','emanuele','milad','giulia','mirella','monica','fabiola']

for name in names:
    guyzz = None #Build an array of csv for every person
    for filename in os.listdir("records/"):
        if name in filename:
            if guyzz is None:
                guyzz = []
                guyzz.append(np.array(np.genfromtxt('records/'+filename,delimiter=',',dtype=None)))
            else:
                guyzz.append(np.genfromtxt('records/'+filename,delimiter=',',dtype=None))



    n_waves=np.shape(guyzz[0])[1]

    comb=list(itertools.combinations(range(np.shape(guyzz)[0]),2))
    # ^^this is so fucking brutal^^
    matrix=np.zeros([np.shape(guyzz)[0],np.shape(guyzz)[0]])
    for i in comb:
        correlation_vector=0
        firsttask=guyzz[i[0]]
        secondtask=guyzz[i[1]]
        minValue= np.min([np.shape(firsttask)[0],np.shape(secondtask)[0]])
        firsttask2=firsttask[0:minValue][:]
        secondtask2=secondtask[0:minValue][:]

        for waves in range(n_waves):
            d= pydtw.dtw(firsttask[:][waves], secondtask[:][waves],
                         pydtw.Settings(dist='manhattan',step='dp2',window='nowindow',
                                        compute_path=True,norm=True))
            correlation_vector+=d.get_dist()

        correlation_vector/=n_waves
        matrix[i[0]][i[1]]=correlation_vector

    print(name + " : ")
    print(matrix)
    print('---------------')

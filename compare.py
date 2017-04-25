from cdtw import pydtw

import os
import numpy as np
import itertools
np.set_printoptions(linewidth=250)

tasks=['logica','memoria','musica_metal','rilassamento']

for item in tasks:
    guyzz=None #Build an array of csv for every person
    for filename in os.listdir("records/"):
        if item in filename:
            if guyzz is None:
                guyzz=[]
                guyzz.append(np.array(np.genfromtxt('records/'+filename,delimiter=',',dtype=None)))
            else:
                guyzz.append(np.genfromtxt('records/'+filename,delimiter=',',dtype=None))



    n_waves=np.shape(guyzz[0])[1]

    comb=list(itertools.combinations(range(np.shape(guyzz)[0]),2))
    # ^^this is so fucking brutal^^
    matrix=np.zeros([np.shape(guyzz)[0],np.shape(guyzz)[0]])
    for i in comb:
        correlation_vector=0
        firstguy=guyzz[i[0]]
        secondguy=guyzz[i[1]]
        minValue= np.min([np.shape(firstguy)[0],np.shape(secondguy)[0]])
        firstguy2=firstguy[0:minValue][:]
        secondguy2=secondguy[0:minValue][:] #trim the fuck out

        for waves in range(n_waves):
            d= pydtw.dtw(firstguy[:][waves], secondguy[:][waves],
                         pydtw.Settings(dist='manhattan',step='dp2',window='nowindow',
                                        compute_path=True,norm=True))
            correlation_vector+=d.get_dist()

        correlation_vector/=n_waves
        matrix[i[0]][i[1]]=correlation_vector

    print(item+":")
    print(matrix)
    print('---------------')

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
tasks=['memoria','logica','rilassamento','musica_metal']
waves=['delta','theta','lowAlpha','highAlpha','lowBeta','highBeta','lowGamma','midGamma','attention'
                    ,'meditation','rawValue']
"""
for task in tasks:
    for filename in os.listdir('records/by_task/'+task):
        dataset = np.genfromtxt('records/by_task/'+task+'/'+filename, delimiter=',', dtype=np.float)
        dataset=dataset[:,0:-1]
        for i in range(np.shape(dataset)[1]):
            plt.plot(dataset[:,i])
        plt.legend(waves)
        pylab.savefig('plots/'+task+'/'+filename[0:filename.index('_')]+'.svg',format='svg', bbox_inches='tight')
        plt.clf()"""


for task in tasks:
    for wave in range(len(waves)):
        names=[]
        for filename in os.listdir('records/by_task/'+task):
            dataset=np.genfromtxt('records/by_task/'+task+'/'+filename,delimiter=',',dtype=np.float)
            plt.plot(dataset[:,wave],lw=0.5)
            names+=[filename[0:filename.index('_')]]
        #plt.legend(names)
        pylab.savefig('plots/by_wave/'+task+'/'+waves[wave]+'.svg',format='svg', bbox_inches='tight')
        plt.clf()
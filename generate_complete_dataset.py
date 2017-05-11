#
# This script creates a unique csv file (from files in "record") to be used as training and test set
# Format:
#   delta,theta,lowalpha,highalpha,lowbeta,highbeta,lowgamma,midgamma,attention,meditation,rawvalue,task
#
# 0-rilassamento
# 1-musica metal
# 2-logica
# 3-memoria
#
import os
import numpy as np
dataset=None

np.set_printoptions(linewidth=250)



for file in sorted(os.listdir('records/')):
    matrix=np.array(np.genfromtxt('records/' + file, delimiter=',', dtype=None))
    matrix=np.delete(matrix,-1,1) #Remove the blink column
    desired_output=np.zeros(4)
    if 'rilassamento' in file:
        desired_output[0]=1
    elif 'musica_metal' in file:
        desired_output[1]=1
    elif 'logica' in file:
        desired_output[2]=1
    elif 'memoria' in file:
        desired_output[3]=1

    matrix=np.concatenate((matrix,np.tile(desired_output,(np.shape(matrix)[0],1))),axis=1)
    if dataset is None:
        dataset=np.array(matrix)
    else:
        dataset=np.append(dataset,matrix,axis=0)

np.savetxt('ioio.csv',dataset,delimiter=',',fmt='%i')

#
# npersone x ntask x onde righe
# punti+[persona,onda,task]
#


import numpy as np
import os

names = ['stefano', 'mirella', 'claudio', 'roberta', 'gianluca', 'michel', 'asia', 'milad',
         'angelo', 'fabiola', 'monica', 'giulia', 'emanuele']

task = ['rilassamento', 'musica_metal', 'logica', 'memoria']

spikeBounds = [25000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]

dataset = None

np.set_printoptions(linewidth=250)

for file in sorted(os.listdir('records/')):
    matrix = np.array(np.genfromtxt('records/' + file, delimiter=',', dtype=None))
    matrix = matrix[:, 0:7]
    desired_output = np.zeros(4)
    if 'rilassamento' in file:
        desired_output[0] = 1
    elif 'musica_metal' in file:
        desired_output[1] = 1
    elif 'logica' in file:
        desired_output[2] = 1
    elif 'memoria' in file:
        desired_output[3] = 1
        
    for i in range(np.shape(record)[1]):
        for j in range(np.shape(matrix)[0]):
            if matrix[i][j] > spikeBounds[i]:
                matrix[i][j] = 1
            else:
                matrix[i][j] = 0

    matrix = np.concatenate((matrix, np.tile(desired_output, (np.shape(matrix)[0], 1))), axis=1)
    if dataset is None:
        dataset = np.array(matrix)
    else:
        dataset = np.append(dataset, matrix, axis=0)

n.savetxt('spikeDataset.csv', dataset, fmt='%i', delimiter=',')

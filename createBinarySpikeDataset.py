import numpy as np
import os
import matplotlib.pyplot as plt

names = ['stefano', 'mirella', 'claudio', 'roberta', 'gianluca', 'michel', 'asia', 'milad',
         'angelo', 'fabiola', 'monica', 'giulia', 'emanuele']

task = ['rilassamento', 'musica_metal', 'logica', 'memoria']

spikeBounds = [25000, 5000, 5000, 5000, 5000, 5000, 5000, 5000]

dataset = None

np.set_printoptions(linewidth=250)

for file in sorted(os.listdir('records/')):
    if "csv" in file:
        matrix = np.array(np.genfromtxt('records/' + file, delimiter=',', dtype=int))
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

        crop = np.shape(matrix)[0] % 5
        if crop > 0:
            for i in range(1, crop + 1):
                np.delete(matrix, (np.shape(matrix)[0] - i), axis=0)

        print(matrix)

        inc_bounds = np.zeros((np.shape(matrix)[1], 5000))
        increment = 500
        for bound in range(0, 1500000, increment):
            for j in range(np.shape(matrix)[1]):
                for i in range(np.shape(matrix)[0]):
                    if matrix[i][j] > spikeBounds[j]:
                        matrix[i][j] = 1
                    else:
                        matrix[i][j] = 0
                    inc_bounds[j][bound % increment] += matrix[i][j]

        print(inc_bounds)

        for j in range(np.shape(matrix)[1]):
            plt.plot(inc_bounds[j][:])
            plt.show()

        matrix = np.concatenate((matrix, np.tile(desired_output, (np.shape(matrix)[0], 1))), axis=1)

        if dataset is None:
            dataset = np.array(matrix)
        else:
            dataset = np.append(dataset, matrix, axis=0)

np.savetxt('spikeDataset.csv', dataset, fmt='%i', delimiter=',')

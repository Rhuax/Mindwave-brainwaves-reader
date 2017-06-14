import numpy as np
import os
import matplotlib.pyplot as plt

names = ['stefano', 'mirella', 'claudio', 'roberta', 'gianluca', 'michel', 'asia', 'milad',
         'angelo', 'fabiola', 'monica', 'giulia', 'emanuele']

task = ['rilassamento', 'musica_metal', 'logica', 'memoria']

spikeBounds = [945*500, 130*500, 43*500, 38*500, 27*500, 22.5*500, 10.97*500, 10.95*500]

dataset = None

np.set_printoptions(linewidth=250)
"""
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
            for i in range(1, crop):
                np.delete(matrix, (np.shape(matrix)[0] - i - 1), axis=0)

        print(matrix)

        inc_bounds = np.zeros((np.shape(matrix)[1], 5000))
        increment = 500
        for bound in range(0, 1500000, increment):
            for j in range(np.shape(matrix)[1]):
                for i in range(np.shape(matrix)[0]):
                    if matrix[i][j] > bound:
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


"""

dataset=np.genfromtxt('eegdataset.csv',delimiter=',',dtype=np.int32)

dataset=np.concatenate((dataset[:,0:8],dataset[:,-4:]),axis=1)
def plot_spikes_distribution():
    for wave in range(np.shape(dataset)[1]):
        cazzo=np.zeros(3000)
        i=0
        for bound in range(500, 1500000, 500):
            col = np.copy(dataset[:, wave])
            for mamt in range(np.shape(col)[0]):
                if col[mamt]>=bound:
                    col[mamt]=1
                else:
                    col[mamt]=0

            cazzo[i]=np.sum(col)
            i+=1
        plt.plot(cazzo)
        plt.show()

bounds=[]

def create_spike_dataset():
    for c in range(8):
        for lamadonnaeputtana in range(np.shape(dataset)[0]):
            if dataset[lamadonnaeputtana][c] > spikeBounds[c]:
                dataset[lamadonnaeputtana][c]=1
            else:
                dataset[lamadonnaeputtana][c]=0

#plot_spikes_distribution()
create_spike_dataset()
np.savetxt('spikeDataset.csv', dataset, fmt='%i', delimiter=',')

import os

import matplotlib as plt
import numpy as np

# import seaborn as sns

tasks = ['memoria', 'logica', 'rilassamento', 'musica_metal']
waves = ['delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'midGamma']

means = []
corrMat = []



def mean_wave(i_wave, squared_waves):
    s = np.sum(squared_waves[:, i_wave])
    s = np.sqrt(s)
    return s.astype(int)


def correlate_waves(person, wave_1, wave_2):
    for filename in os.listdir('records/by_person/' + person):
        dataset = np.genfromtxt('records/by_person/' + filename, delimiter=',', dtype=int)
        dataset = dataset[:, :-4]
        datasets.append(dataset)
        print(datasets)
    for i in np.shape(datasets)[0]:
        corrMat[wave_1][wave_2] += datasets[i][wave_1]*datasets[i][wave_2] / means[wave_1] * means[wave_2]

total_waves = np.zeros((0, 8))

for task in tasks:
    ds = np.zeros((0, 8))
    dims = []
    datasets = []
    for filename in os.listdir('records/by_task/' + task):
        dataset = np.genfromtxt('records/by_task/' + task + '/' + filename, delimiter=',', dtype=int)
        dataset = dataset[:, :-4]
        datasets.append(dataset)
        dims.append(np.shape(dataset)[0])
    mindim = np.min(dims)
    for dataset in datasets:
        total_waves = np.vstack([total_waves, dataset[:mindim, :]])
        # corr(np.min(dims), ds)

squared_waves = np.square(total_waves)
for i in range(np.shape(waves)[0]):
    means.append(mean_wave(i, squared_waves))

print(means)

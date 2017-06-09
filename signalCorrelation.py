import os

import matplotlib as plt
import numpy as np

# import seaborn as sns

tasks = ['memoria', 'logica', 'rilassamento', 'musica_metal']
waves = ['delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'midGamma']

means = []


def mean_wave(i_wave, squared_waves):
    s = np.sum(squared_waves[:, i_wave])
    return np.sqrt(s)


total_waves = np.zeros((0, 8))

for task in tasks:
    ds = np.zeros((0, 8))
    datasets = []
    dims = []
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
for i in range(waves.__len__()):
    means.append(mean_wave(i, squared_waves))

print(means)

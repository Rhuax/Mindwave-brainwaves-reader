import os
import numpy as np
import pandas as pd
import matplotlib as plt
import pylab
import seaborn as sns
np.set_printoptions(precision=7, suppress=True)
tasks=['memoria','logica','rilassamento','musica_metal']
waves=['delta','theta','lowAlpha','highAlpha','lowBeta','highBeta','lowGamma','midGamma']

def mean(wave):
    return np.sqrt(np.sum(np.square(wave)))

def correlation(dataset, means):
    df = pd.DataFrame(index = waves, columns = waves)
    for i in range(waves.__len__()):
        for j in range(waves.__len__()):
            df[waves[i]][waves[j]] = correlate(means[i], means[j], dataset[:, i], dataset[:, j])
    return df.astype(float)


def correlate(mean1, mean2, wave1, wave2):
    a = np.divide(wave1, mean1)
    b = np.divide(wave2, mean2)
    return np.dot(a, b)

for task in tasks:
    dims = []
    for filename in os.listdir('records/by_task/'+task):
        dataset = np.genfromtxt('records/by_task/'+task+'/'+filename,delimiter=',',dtype=np.float64)
        dataset = dataset[:, :-4]
        dims.append(np.shape(dataset)[0])
    mindim = np.min(dims)
    for filename in os.listdir('records/by_task/'+task):
        means = []
        dataset = np.genfromtxt('records/by_task/'+task+'/'+filename,delimiter=',',dtype=np.float64)
        dataset = dataset[:mindim, :-4]
        for i in range(waves.__len__()):
            means.append(mean(dataset[:,i]))
        sns.plt.figure(figsize = (12, 9))
        sns.heatmap(correlation(dataset, means))
        sns.plt.title(filename)
        sns.plt.savefig('plots/correlation/'+filename+'.svg', dpi = 100)
        sns.plt.clf()

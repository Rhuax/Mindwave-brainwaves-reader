import os

import matplotlib as plt
import numpy as np
import pandas as pd
import pylab
import seaborn as sns

np.set_printoptions(precision=7, suppress=True)

tasks = ['memoria', 'logica', 'rilassamento', 'musica_metal']
waves = ['delta', 'theta', 'lowAlpha', 'highAlpha', 'lowBeta', 'highBeta', 'lowGamma', 'midGamma']
persons = ['angelo', 'asia', 'claudio', 'emanuele', 'fabiola', 'gianluca', 'giulia', 'michel', 'milad', 'mirella', 'monica', 'roberta', 'stefano']

def mean(wave):
    return np.sqrt(np.sum(np.square(wave)))


def correlation(dataset, means):
    df = pd.DataFrame(index=waves, columns=waves)
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
    complete_ds = np.zeros((0, 8))
    for filename in os.listdir('records/by_task/' + task):
        print(filename)
        dataset = np.genfromtxt('records/by_task/' + task + '/' + filename, delimiter=',', dtype=np.int)
        dataset = dataset[:, :-4]
        dims.append(np.shape(dataset)[0])
    mindim = np.min(dims)
    means = []
    for filename in os.listdir('records/by_task/' + task):
        dataset = np.genfromtxt('records/by_task/' + task + '/' + filename, delimiter=',', dtype=np.int)
        dataset = dataset[:mindim, :-4]
        complete_ds = np.vstack([complete_ds, dataset])

    for i in range(waves.__len__()):
        means.append(mean(complete_ds[:, i]))
    sns.plt.figure(figsize=(12, 9))
    corr = correlation(complete_ds, means)
    sns.heatmap(corr)
    sns.plt.title(task)
    np.savetxt('plots/correlation_by_task/'+task+'_dataset.csv', complete_ds, fmt = '%1.10f')
    np.savetxt('plots/correlation_by_task/'+task+'_means.csv', np.array(means), fmt = '%1.10f')
    np.savetxt('plots/correlation_by_task/'+task+'_correlation.csv', corr, fmt = '%1.10f')

    sns.plt.savefig('plots/correlation_by_task/' + task + '.svg', dpi=100)
    sns.plt.clf()

for person in persons:
    dims = []
    complete_ds = np.zeros((0, 8))
    for filename in os.listdir('records/by_person/' + person):
        print(filename)
        dataset = np.genfromtxt('records/by_person/' + person + '/' + filename, delimiter=',', dtype=np.int)
        dataset = dataset[:, :-4]
        dims.append(np.shape(dataset)[0])
    mindim = np.min(dims)
    means = []
    for filename in os.listdir('records/by_person/' + person):
        dataset = np.genfromtxt('records/by_person/' + person + '/' + filename, delimiter=',', dtype=np.int)
        dataset = dataset[:mindim, :-4]
        complete_ds = np.vstack([complete_ds, dataset])
    for i in range(waves.__len__()):
        means.append(mean(complete_ds[:, i]))
    sns.plt.figure(figsize=(12, 9))
    corr = correlation(complete_ds, means)
    sns.heatmap(corr)
    sns.plt.title(person)
    np.savetxt('plots/correlation_by_person/'+person+'_dataset.csv', complete_ds, fmt = '%1.10f')
    np.savetxt('plots/correlation_by_person/'+person+'_means.csv', np.array(means), fmt = '%1.10f')
    np.savetxt('plots/correlation_by_person/'+person+'_correlation.csv', corr, fmt = '%1.10f')

    sns.plt.savefig('plots/correlation_by_person/' + person + '.svg', dpi=100)
    sns.plt.clf()
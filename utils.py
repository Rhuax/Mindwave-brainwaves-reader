import numpy as np
import random

dataset=np.genfromtxt('eegdataset.csv',delimiter=',',dtype=np.int32)

def create_array_task(dataset, sequences):
    current_output = dataset[0][-4:]
    array_task = np.zeros(sequences,dtype=np.int32)
    line = 0
    i = 1
    for row in dataset:
        actual = row[-4:]

        if not np.array_equal(actual, current_output):
            array_task[i] = line
            i += 1
            current_output = actual
        line += 1
    return array_task

def two_over_thirteen_fold_CV(dataset, increment, test_people):
    seq_step = 4
    i = 0

    seqs = create_array_task(dataset, 52)
    train_all = [[-1 for x in range(11)] for y in range(len(dataset))]
    validate_all = [[-1 for x in range(11)] for y in range(len(dataset))]

    while i < (11 * seq_step):
        val_start = seqs[i]
        val_end = seqs[i + (test_people * 4)]
        train_all[int(i / (seq_step * increment))][:val_start] = dataset[:val_start - 1]
        train_all[int(i / (seq_step * increment))][val_start + 1:] = dataset[val_end + 1:]
        validate_all[int(i / (seq_step * increment))] = dataset[val_start:val_end]
        i += (seq_step * increment)
    return train_all, validate_all


train, val = two_over_thirteen_fold_CV(dataset, 1, 2)   #test_people default is 2, increment can be 1 or 2.
for i in range(11):
    np.savetxt('cross_validation/training_'+str(i)+'.csv', train[i], fmt='%i', delimiter=',')
    np.savetxt('cross_validation/testing_'+str(i)+'.csv', val[i], fmt='%i', delimiter=',')


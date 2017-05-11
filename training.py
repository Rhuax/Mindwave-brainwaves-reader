import numpy as np

from keras.layers import LSTM,Dense
from keras.models import Sequential

dataset=np.genfromtxt('eegdataset.csv',delimiter=',',dtype=np.int32)


def calculate_max_sequence_length(dataset):
    max_seq_length = 0
    cur_seq_length = 0
    current_output = dataset[0][-4:]
    seqs = 1
    for row in dataset:
        o=row[-4:]
        if not np.array_equal(o, current_output):
            seqs += 1
            if cur_seq_length > max_seq_length:

                max_seq_length = cur_seq_length
            current_output = o
            cur_seq_length = 1
        else:
            cur_seq_length += 1
    return max_seq_length, seqs


def create_array_task(dataset,sequences):
    current_output = dataset[0][-4:]
    array_task = np.zeros(sequences)
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

print(create_array_task(dataset, 52))

"""
It builds the network, defining its structure
"""
def create_model():
    model=Sequential()
    model.add(LSTM(11,return_sequences=True))




max_length,sequences=calculate_max_sequence_length(dataset)
sequences_indices=create_array_task(dataset,sequences=sequences)

import numpy as np

from keras.layers import LSTM,Dense,Dropout
from keras.models import Sequential


epochs=10
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



"""
It builds the network, defining its structure
"""
def create_model():
    model=Sequential()
    model.add(LSTM(11,stateful=True,return_sequences=True,input_shape=(None,11),batch_size=5))
    model.add(LSTM(11,return_sequences=True))
    model.add(Dropout(0.3))
    model.add(LSTM(11))
    model.add(Dense(4,activation='softmax'))
    return model




max_length,sequences=calculate_max_sequence_length(dataset)
sequences_indices=create_array_task(dataset,sequences=sequences)

network=create_model()
print('Compiling model..')
network.compile(optimizer='rmsprop',loss='categorical_crossentropy')
print('Model compiled. Specs:')
network.summary()
for e in range(epochs):
    current_sequence=0
    for seq in range(sequences-1):
        start=sequences_indices[current_sequence]
        end=sequences_indices[current_sequence+1]
        Train=dataset[start:end]
        #Split the sequence in batches of 5 instances
        for batch in np.split(Train,5,axis=0):
            X_Train=batch[:][0:-4]
            Y_Train=batch[:][-4:]



        network.reset_states()



        current_sequence+=1

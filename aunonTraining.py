import numpy as np

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from keras.optimizers import RMSprop, SGD, Adam

"""
sequenze per ogni task
[ 65.  65.  64.  65.  65.]
il terzo task ha 64 sequenze

"""

sequence_length = 250*10 # 10 seconds sampling at 250 Hz




np.set_printoptions(linewidth=200)
epochs = 5

batch_size = 5
dataset = np.genfromtxt('keirnAunonDataset.csv', delimiter=',', dtype=np.float32)


def calculate_sequences(dataset):
    return np.split(dataset,range(sequence_length,np.shape(dataset)[0]-sequence_length,sequence_length),axis=0)
                            #^^This is so brutal^^


def create_array_task(dataset, sequences):
    current_output = dataset[0][-5:]
    array_task = np.zeros(sequences, dtype=np.float32)
    line = 0
    i = 1
    for row in dataset:
        actual = row[-5:]

        if not np.array_equal(actual, current_output):
            array_task[i] = line
            i += 1
            current_output = actual
        line += 1
    return array_task


def k_fold_CV(dataset, folds):
    folds += 1
    fold_len = int(len(dataset) / folds)
    train_all = [[0 for x in range(folds)] for y in range(len(dataset) - fold_len)]
    validate_all = [[0 for x in range(folds)] for y in range(fold_len)]
    for i in range(1, folds):
        val_start = i * fold_len
        val_end = (i + 1) * fold_len
        train_all[i][:val_start] = dataset[:val_start - 1]
        train_all[i][val_start + 1:] = dataset[val_end + 1:]
        validate_all[i] = dataset[val_start:val_end]
    return train_all, validate_all


# train, val = k_fold_CV(dataset, 10)
# for i in range(1, 11):
#    np.savetxt('cross_validation/training_'+str(i)+'.csv', train[i], fmt='%i', delimiter=',')
#    np.savetxt('cross_validation/testing_'+str(i)+'.csv', val[i], fmt='%i', delimiter=',')


"""
It builds the network, defining its structure
"""


def create_model():
    model = Sequential()
    model.add(LSTM(11, stateful=True, return_sequences=True, batch_input_shape=(1, batch_size, 7)))
    model.add(LSTM(11, return_sequences=True))
    model.add(LSTM(11, return_sequences=True))
    model.add(LSTM(11))
    model.add(Dense(11))
    model.add(Dense(5, activation='softmax'))
    return model


def calculate_accuracy(set, model):
    total_accuracy = 0
    for seqa in set:

        y_true_test = np.reshape(seqa[0, -5:], (1, 5))
        # print('y è :', end='')
        # print(y_true)

        k = 0
        true_positives = 0
        for mini_batch_test in range(0,sequence_length,batch_size):
            batch_x = seqa[mini_batch_test:mini_batch_test+batch_size, 0:-5]
            batch_x=np.expand_dims(batch_x,0)
            output = model.predict_on_batch(batch_x)
            r_output = np.zeros((1, 5))
            r_output[0][np.argmax(output)] = 1
            if np.array_equal(r_output, y_true_test):
                true_positives += 1

            k += 1
        # print('accuracy : ',end='')
        total_accuracy += true_positives/(sequence_length/batch_size)
        model.reset_states()

    return total_accuracy /( np.shape(set)[0])


sequences = calculate_sequences(dataset)
network = create_model()
print('Compiling model..')
opt = RMSprop(lr=0.0005)
network.compile(optimizer=opt, loss='categorical_crossentropy',
                metrics=['accuracy'])
print('Model compiled. Specs:')
network.summary()

#there are 324 sequences

training_set=sequences[:300]
test_set=sequences[300:]
network = create_model()

opt = RMSprop(lr=0.00005)
network.compile(optimizer=opt, loss='categorical_crossentropy',
                metrics=['accuracy'])

for e in range(epochs):
    epoch_accuracy = 0
    print('Epoch :' + str(e))
    for seq in training_set:
        y_true = np.reshape(seq[0,-5:], (1, 5))
        print('task: ',end='')
        print(y_true)
        j = 0
        avg_loss=0
        acc = 0
        for mini_batch in range(0,sequence_length,batch_size):

            batch_x = seq[mini_batch:mini_batch+batch_size,0:-5]
            batch_x = np.expand_dims(batch_x, 0)
            loss, accuracy = network.train_on_batch(batch_x, y_true)
            # print('Accuracy: '+str(accuracy))
            avg_loss += loss
            acc += accuracy
            # print('loss on batch '+str(j),end=' ')
            # print(':'+str(loss))
            j += 1
        print('accuratezza media per questa sequenza '+str(acc /(sequence_length/batch_size)))
        print('avg loss :'+str(avg_loss/(sequence_length/batch_size)))
        print('accuratezza sul test set :' + str(calculate_accuracy(test_set, network)))
        epoch_accuracy += acc /(sequence_length/batch_size)
        # print('accuratezza media per questa sequenza '+str(acc/((end-start)/5)))
        network.reset_states()
    print('accuratezza media per ques\'epoca sul training set: ' + str(epoch_accuracy / np.shape(training_set)[0]))
    print('accuratezza sul test set :' + str(calculate_accuracy(test_set, network)))

f = open('architecture', 'w')
f.write(network.to_json())
f.close()
network.save('network.h5')

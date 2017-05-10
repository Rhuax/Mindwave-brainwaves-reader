import pandas as pd
import seaborn as sea
import matplotlib as plt
from keras.models import Sequential
from keras.layers import Dense

from sklearn.decomposition import PCA

waves = pd.read_csv('preprocessed/eeg.csv', sep = ',')
labels = waves['Task']
training = waves.drop(labels = 'Task', axis = 1)
labels = labels.DataFrame.as_matrix()
training = training.values

#Model definition
model = Sequential()
model.add(Dense(12, input_dim=439, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics = ['accuracy', 'mae', 'mse'])
model.fit(training, labels, epochs=500, batch_size=10)
scores = model.evaluate(training, labels)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))





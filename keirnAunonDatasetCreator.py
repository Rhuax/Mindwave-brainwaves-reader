#data is a cell array of cell arrays.  Each individual cell array is made up of a
    # subject string, task string, trial string, and data array.
# Each data array is 7 rows by 2500 columns.
# The 7 rows correspond to channels c3, c4, p3, p4, o1, o2, and EOG.
# Across columns are samples taken at 250 Hz for 10 seconds, for 2500 samples.
# For example, the first cell array looks like
#       'subject 1'   'baseline'   'trial 1'[7x2500 single].

# Recordings were made with reference to electrically linked mastoids A1 and A2.
# EOG was recorded  between the forehead above the left browline and another on the left cheekbone.
# Recording was performed with a bank of Grass 7P511 amplifiers whose bandpass analog filters were set at 0.1 to 100 Hz.
#
# Subjects 1 and 2 were employees of a university and were left-handed age 48 and right-handed age 39, respectively.
# Subjects 3 through 7 were right-handed college students between the age of 20 and 30 years old.
# All were mail subjects with the exception of Subject 5.
# Subjects performed five trials of each task in one day. They returned to do a second five trials on another day.
# Subjects 2 and 7 completed only one 5-trial session. Subject 5 completed three sessions.
# For more information see Alternative Modes of Communication Between Man and Machine, Zachary A. Keirn, Masters Thesis in Electrical Engineering, Purdue University, December, 1988.
import numpy as np

tasks = {'baseline':0, 'counting': 1, 'letter-composing':2, 'multiplication':3, 'rotation':4}

dataset = None
eegFile = open('eegdata.ascii', 'r')
rowCounter = 0
channelMatrix = np.zeros(shape=(2500, 7))
desiredOutput = np.zeros(5)
for rows in eegFile:
    currentTask = None
    if ',' in rows:
        tokenizedRow = rows.split(', ')
        currentTask = tasks[tokenizedRow[1]]
        desiredOutput = np.zeros(5)
        desiredOutput[currentTask] = 1
        desiredOutput = np.tile(desiredOutput, (2500, 1))
        channelMatrix = np.hstack((channelMatrix, desiredOutput))
        rowCounter = 0
        #print(tokenizedRow[1], tasks[tokenizedRow[1]])
        #print(desiredOutput)
    elif rows == '\n':
        if dataset is None:
            dataset = np.array(channelMatrix)
        else:
            dataset = np.append(dataset, channelMatrix, axis=0)
        print(dataset)
        channelMatrix = np.zeros(shape=(2500, 7))
        desiredOutput = np.zeros(5)
        continue
    else:
        values = rows.split(' ')
        values = values[:-1]
        channelMatrix[:,rowCounter] = np.asarray(values)
        rowCounter += 1
        print(channelMatrix)
        #outFile.write(values)
        #outFile.write('\n')


np.savetxt('keirnAunonDataset.csv', dataset, delimiter=',', fmt='%1.3f')


names = ['stefano', 'mirella', 'claudio', 'roberta', 'gianluca', 'michel', 'asia', 'milad',
         'angelo', 'fabiola', 'monica', 'giulia', 'emanuele']

task = ['rilassamento', 'musica_metal', 'logica', 'memoria']

import os
import numpy as n

# 13persone x 4 task x 11
okok = 0
dataset = n.zeros(shape=[572, 437 + 3])
dataset.fill(-n.inf)
index = 0

import glob

for name in names:
    for filename in sorted(os.listdir('records/')):
        if name in filename:
            name_index = names.index(name)
            task_index = None
            for i in task:
                if i in filename:
                    task_index = task.index(i)

            record = n.genfromtxt('records/' + filename, delimiter=',')
            record = n.delete(record, -1, 1)  # Del the muthafucking blink strenght
            for i in range(n.shape(record)[1]):  # For every brain wave

                dataset[index][0 : n.shape(n.array(record[:, i]))[0]] = n.array(record[:, i])
                dataset[index][-3] = name_index
                dataset[index][-2] = task_index
                dataset[index][-1] = i
                index += 1

n.savetxt('superdataset.csv', dataset, fmt='%.3f', delimiter=',')
#
# npersone x ntask x onde righe
# punti+[persona,task,onda]
#

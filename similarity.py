import numpy as np
from cdtw import pydtw
from matplotlib import pyplot as p

matrix1 = np.genfromtxt('records/gianluca_logica_200.csv', delimiter=',', dtype=int)
matrix2 = np.genfromtxt('records/claudio_logica_200.csv', delimiter=',', dtype=int)

n_waves = np.shape(matrix1)[1]
correlation_vector = np.zeros(n_waves)

minValue = np.min([np.shape(matrix1)[0], np.shape(matrix2)[0]])
#15540 ril
#6627
#46644

#13130
#23815
#30386
#61583


# Smooth values
#matrix1_smooth = np.sqrt(matrix1[0:minValue][:])
#matrix2_smooth = np.sqrt(matrix2[0:minValue][:])
matrix1_smooth=matrix1[0:minValue][:]
matrix2_smooth=matrix2[0:minValue][:]
'''
Old cross correlation code

for i in range(n_waves):  # For every brain wave
    c_cor = np.correlate(matrix1[0:minValue][i], matrix2[0:minValue][i])
    ilmassimodioporco = np.max([np.correlate(matrix1[0:minValue][i], matrix1[0:minValue][i]),
                                np.correlate(matrix2[0:minValue][i], matrix2[0:minValue][i])])
    norm_val = c_cor / ilmassimodioporco
    correlation_vector[i] = norm_val

print('Correlation for every brain wave:',end='')
print( correlation_vector)
print('Average correlation value: ', end='')
print(correlation_vector.mean())'''

# print(np.correlate(matrix1[0:minValue][0], matrix1[0:minValue][0]))
# print(np.correlate(matrix2[0:minValue][0], matrix2[0:minValue][0]))

for i in range(n_waves):
    d = pydtw.dtw(matrix1_smooth[:][i], matrix2_smooth[:][i],
                  pydtw.Settings(dist='manhattan', step='dp2', window='nowindow',
                                 compute_path=True, norm=True))
    #p.plot(matrix1[:minValue][i])
    #p.plot(matrix2[:minValue][i])
    #p.show()
    correlation_vector[i] = d.get_dist()
    #d.plot_alignment()

print('Correlation for every brain wave:', end='')
print(correlation_vector)
print('Average correlation value: ', end='')
print(correlation_vector.mean())

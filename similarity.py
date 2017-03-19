import numpy as np

matrix1 = np.genfromtxt('stefano_relax.csv', delimiter=',', dtype=int)
matrix2 = np.genfromtxt('michel_relax.csv', delimiter=',', dtype=int)

n_waves = np.shape(matrix1)[1]
correlation_vector=np.zeros(n_waves)
i = 6
minValue = np.min([np.shape(matrix1)[0], np.shape(matrix2)[0]])

# Smooth values
matrix1 = np.sqrt(matrix1)
matrix2 = np.sqrt(matrix2)

for i in range(n_waves):  # For every brain wave
    c_cor = np.correlate(matrix1[0:minValue][i], matrix2[0:minValue][i])
    ilmassimodioporco = np.max([np.correlate(matrix1[0:minValue][i], matrix1[0:minValue][i]),
                                np.correlate(matrix2[0:minValue][i], matrix2[0:minValue][i])])
    norm_val = c_cor / ilmassimodioporco
    correlation_vector[i] = norm_val

print('Correlation for every brain wave:',end='')
print( correlation_vector)
print('Average correlation value: ',end='')
print(correlation_vector.mean())

# print(np.correlate(matrix1[0:minValue][0], matrix1[0:minValue][0]))
# print(np.correlate(matrix2[0:minValue][0], matrix2[0:minValue][0]))

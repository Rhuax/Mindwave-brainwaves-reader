import numpy as np
dataset=np.genfromtxt('ioio.csv',delimiter=',',dtype=np.int32)




def calculate_max_sequence_length(dataset):
    max_seq_length=0
    cur_seq_length=0
    current_output=dataset[0][-4:]
    seqs=1
    for row in dataset:
        o=row[-4:]
        if not np.array_equal(o,current_output):
            seqs+=1
            if cur_seq_length>max_seq_length:

                max_seq_length=cur_seq_length
            current_output=o
            cur_seq_length=1
        else:
            cur_seq_length+=1
    return max_seq_length,seqs


print(calculate_max_sequence_length(dataset))
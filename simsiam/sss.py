import numpy as np

n=-1
record = np.zeros([32,32])
for i in range(8,40):
    n += 1
    n_a = -1
    for j in range(1,39):
        mul = i*j
        div = i/j
        if 250 < mul and 500 > mul:
            if div > 0.3 and div < 3:
                n_a += 1
                record[n,n_a] = j

np.savetxt('record.csv',record,delimiter = ',')




import numpy as np

a = [[1,2,3],[4,5,6]]
a_arr = np.array(a)
# print(a_arr.shape)
c = [1,2]
for lab in a_arr:
    print(lab[c])

import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

a = np.asarray([0,1,2,3,2,1,0])
a = np.asarray(list(range(4)))
print(a)
enc = OneHotEncoder()
enc.fit(a.reshape(-1,1))
print(enc.n_values_)
b = np.asarray([[2,1],[3,2]])
print(enc.transform(b.reshape(-1,1)).toarray())

'''
d = {1:0,2:1}
print(len(d))
e = ([1,2],[2,3],[4,5])
f = np.asarray(e)
print(f)
'''
e = ['positive','neutral','negative']
enc = LabelEncoder()
enc.fit(e)
f = ['neutral','negative','neutral','positive']
g = enc.transform(f)
h = enc.inverse_transform(g).tolist()
print(h)


d = np.asarray([[0,0,1],[1,0,0]])
print(np.argmax(d, axis=1))

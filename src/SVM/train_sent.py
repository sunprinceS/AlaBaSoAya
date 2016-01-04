#!/usr/bin/python2
# -*- coding: utf-8 -*-
from tgrocery import Grocery
import sys
pred = open('prediction_{}'.format(sys.argv[1]),'w')
golden = open('golden_{}'.format(sys.argv[1]),'w')
train_fn = Grocery(sys.argv[1])

labels=[]
# comments=[]
data=[]

with open('../data/{}.dat'.format(sys.argv[1])) as comment_data:
    comments = comment_data.read().splitlines()
with open('../data/{}.sent'.format(sys.argv[1])) as lab_file:
    lab_tmp = lab_file.read().splitlines()
    for lab in lab_tmp:
        labels.append(lab.split(',')[1])
for comment , lab in zip(comments,labels):
    data.append([lab,comment])
train_fn.train(data[:int(len(data)*4/5)])
train_fn.save()

valid_fn = Grocery(sys.argv[1])
valid_fn.load()

for sent in data[int(len(data)*4/5):]:
    pred.write('{}\n'.format(valid_fn.predict(sent[1])))
    golden.write('{}\n'.format(sent[0]))
        
pred.close()
golden.close()



# Predict
acc = valid_fn.test(data[int(len(data)*4/5):])
print("Out-sample accuracy : {}\n".format(acc))

#!/usr/bin/python2
# -*- coding: utf-8 -*-

from tgrocery import Grocery
import sys

# build map
# lab_map={}
# with open('misc_data/categoryMap/{}.category'.format(sys.argv[1])) as category_file:
    # categories = category_file.read().splitlines()
    # for idx,category in enumerate(categories):
        # lab_map[category]=idx

# loading training data
train_data=[]
with open('misc_data/{}_train.asp.dat'.format(sys.argv[1])) as train_data_file:
    train_data = train_data_file.read().splitlines()

# loading label
labs=[]
with open('misc_data/{}_train.asp.label'.format(sys.argv[1])) as lab_file:
    labs = lab_file.read().splitlines()
tr = []
for y,x in zip(labs,train_data):
    tr.append((y.split(',')[1],x))
# build SVM model
train_fn = Grocery(sys.argv[1])
train_fn.train(tr)
train_fn.save()

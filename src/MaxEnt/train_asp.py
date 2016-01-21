#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
File: train_asp.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/sunprinceS
Description: 
"""
import joblib as jl
import numpy as np
import scipy as sp
from sklearn.linear_model import LogisticRegression
import sys

from util import *

def main():

    data_matrix=[]
    train_labels=[]

    #load category map
    lab_map=io.loadMap(sys.argv[1],'train')

    #Loading training data
    train_corpus = io.loadDat(sys.argv[1],sys.argv[2],'train','asp')

    #Load label
    train_labels=io.loadLabel(sys.argv[1],sys.argv[2],'train','asp',lab_map)

    if sys.argv[3] == 'bow':

        data_matrix = transform.BOWtransform(train_corpus,'train',sys.argv[1],sys.argv[3],sys.argv[2])

    elif sys.argv[3] == 'wv':

        data_matrix = transform.gloveTransform(train_corpus)

    elif sys.argv[3] == 'bow+wv':

        data_matrix = transform.BOWtransform(train_corpus,'train',sys.argv[1],sys.argv[3],sys.argv[2])

        wv_matrix = transform.gloveTransform(train_corpus)

        #concat bow and glove vector
        for vec,wv_vec in zip(data_matrix,wv_matrix):
            vec.extend(wv_vec)

    else:
        print('Unexpected type!',file=sys.stderr)
        sys.exit()

    print('Data preprocessing end...\nStart training...')

    

if __name__ == "__main__":
    main()

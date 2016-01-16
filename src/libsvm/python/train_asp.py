#!/usr/bin/env python
# -*- coding: utf-8 -*-

from svmutil import *
from util import *
import sys

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

    # print('Data preprocessing end...\nStart training SVM...')
    svm_classify_model = svm_train(train_labels,data_matrix,'-t 0 -b 1 -q')
    svm_save_model('{}/{}.model_{}.{}.{}'.format(marcos.SVM_MODEL_DIR,sys.argv[1],'asp',sys.argv[3],sys.argv[2]),svm_classify_model)

if __name__ == "__main__":
    main()

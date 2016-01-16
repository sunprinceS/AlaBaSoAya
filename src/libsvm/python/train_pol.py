#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: train_sent.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/yourname
Description: train ABSA slot3 in SVM
"""

from svmutil import *
from util import *
import sys

lab_map={'positive':1,'negative':-1,'neutral':0}

def main():
    asp_map = io.loadMap(sys.argv[1],'train')
    asp_list = io.loadAsp(sys.argv[1],'train',sys.argv[2])
    train_labels = io.loadLabel(sys.argv[1],sys.argv[2],'train','pol',lab_map)
    data_matrix_tmp = io.loadTreeLSTMVec(sys.argv[1],'train',sys.argv[2])
    data_matrix = transform.addAspect(data_matrix_tmp,asp_map,asp_list)
    svm_classify_model = svm_train(train_labels,data_matrix,'-t 0 -b 1 -q')
    svm_save_model('{}/{}.model_{}.{}'.format(marcos.SVM_SENT_MODEL_DIR,sys.argv[1],'pol',sys.argv[2]),svm_classify_model)

if __name__ == "__main__":
    main()

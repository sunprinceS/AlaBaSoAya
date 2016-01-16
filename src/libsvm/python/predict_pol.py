#!/usr/bin/env python
# -*- coding: utf-8 -*-

from svmutil import *
from util import *
import sys
import numpy as np

lab_map={1:'positive',-1:'negative',0:'neutral'}

def main():
    data_matrix = []

    #load category map
    asp_map = io.loadMap(sys.argv[1],'test_pol')
    asp_list = io.loadAsp(sys.argv[1],'te',sys.argv[2])
    data_matrix_tmp = io.loadTreeLSTMVec(sys.argv[1],'te',sys.argv[2])
    data_matrix = transform.addAspect(data_matrix_tmp,asp_map,asp_list)
    con_labels = [0]*len(asp_list)
    svm_classify_model = svm_load_model('{}/{}.model_{}.{}'.format(marcos.SVM_SENT_MODEL_DIR,sys.argv[1],'pol',sys.argv[2]))
    lab_set = svm_classify_model.get_labels()
    p_labels,p_acc,p_val=svm_predict(con_labels,data_matrix,svm_classify_model,'-b 1')
    with open('{}/{}.pol.pred.{}'.format(marcos.MISC_DIR,sys.argv[1],sys.argv[2]),'w') as ans_file:
        for label in p_labels:
            ans_file.write('{}\n'.format(lab_map[label]))
if __name__ == "__main__":
    main()

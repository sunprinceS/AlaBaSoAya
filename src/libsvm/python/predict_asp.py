#!/usr/bin/env python
# -*- coding: utf-8 -*-

from svmutil import *
from util import *
import sys

def main():

    data_matrix=[]

    #load category map
    lab_map=io.loadMap(sys.argv[1],'te')

    # load testing data
    test_corpus = io.loadDat(sys.argv[1],sys.argv[2],'te','asp')

    if sys.argv[4] == 'bow':

        data_matrix = transform.BOWtransform(test_corpus,'te',sys.argv[1],sys.argv[4],sys.argv[2])

    elif sys.argv[4] == 'wv':

        data_matrix = transform.gloveTransform(test_corpus)


    elif sys.argv[4] == 'bow+wv':

        data_matrix = transform.BOWtransform(test_corpus,'te',sys.argv[1],sys.argv[4],sys.argv[2])
        wv_matrix = transform.gloveTransform(test_corpus)

        #concat bow and glove vector
        for vec,wv_vec in zip(data_matrix,wv_matrix):
            vec.extend(wv_vec)

    else:
        print("unexpected type!",file=sys.stderr)

    #use SVM to classify
    #dummy label
    con_labels=[0]*len(test_corpus)

    svm_classify_model= svm_load_model('{}/{}.model_{}.{}.{}'.format(marcos.SVM_MODEL_DIR,sys.argv[1],'asp',sys.argv[4],sys.argv[2]))
    lab_set = svm_classify_model.get_labels()
    p_labels,p_acc,p_val=svm_predict(con_labels,data_matrix,svm_classify_model,'-b 1')


    #write ans
    with open('output/{}_asp.out.{}'.format(sys.argv[1],sys.argv[2]),'w') as ans:
        for p_dis in p_val:
            ans_list=[]
            for idx,p in enumerate(p_dis):
                if(p >= float(sys.argv[3])): # threshold
                    ans_list.append(lab_map[lab_set[idx]])
            ans.write('{}\n'.format(' '.join(ans_list)))


if __name__ == "__main__":
    main()

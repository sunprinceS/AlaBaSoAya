#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from svmutil import *
from sklearn.feature_extraction.text import CountVectorizer
import sys

lab_map={}
lab_set=[]

with open('misc_data/categoryMap/{}.category'.format(sys.argv[1])) as category_file:
    categories = category_file.read().splitlines()
    for idx,category in enumerate(categories):
        lab_map[idx]=category

# build bag of words analyzer
vectorizer = CountVectorizer(min_df=1)
with open('misc_data/{}_te.asp.dat'.format(sys.argv[1])) as test_data:
    test_corpus = test_data.read().splitlines()
    bow_matrix = vectorizer.fit_transform(test_corpus)
    
    bow_corpus_numeric = bow_matrix.toarray()

    con_labels=[0]*len(test_corpus)

    bow_corpus_numeric = bow_corpus_numeric.tolist()
    svm_classify_model= svm_load_model('src/libsvm/SVMmodel/{}.model'.format(sys.argv[1]))
    lab_set = svm_classify_model.get_labels()
    p_labels,p_acc,p_val=svm_predict(con_labels,bow_corpus_numeric,svm_classify_model,'-b 1')

with open('output/{}_asp.out'.format(sys.argv[1]),'w') as ans:
    # print(len(lab_set))
    # print(len(lab_map))
    # Among this label , a label is only occur at once (MULTIMEDIA_DEVICE#MISC) , drop it!
    for p_dis in p_val:
        ans_list=[]
        for i,p in enumerate(p_dis):
            if(p >= float(sys.argv[2])): # threshold
                # print(lab_set[i])
                ans_list.append(lab_map[lab_set[i]])
        ans.write('{}\n'.format(' '.join(ans_list)))

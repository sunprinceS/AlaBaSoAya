#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from svmutil import *
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import sys

lab_map={}
lab_set=[]

with open('misc_data/categoryMap/{}.category'.format(sys.argv[1])) as category_file:
    categories = category_file.read().splitlines()
    for idx,category in enumerate(categories):
        lab_map[idx]=category

# build bag of words analyzer
vectorizer = joblib.load('src/libsvm/python/transformModel/vec.{}.{}'.format(sys.argv[1],sys.argv[2])
)
tfidf_transformer = joblib.load('src/libsvm/python/transformModel/tfidf.{}.{}'.format(sys.argv[1],sys.argv[2]))

# pca = joblib.load('src/libsvm/python/transformModel/pca.{}.{}'.format(sys.argv[1],sys.argv[2]))


#Predicting stage
with open('misc_data/{}_te.asp.dat.{}'.format(sys.argv[1],sys.argv[2])) as test_data:
    test_corpus = test_data.read().splitlines()
    bow_matrix = vectorizer.transform(test_corpus)

    bow_matrix = tfidf_transformer.transform(bow_matrix)
    bow_corpus_numeric = bow_matrix.toarray()
    # bow_corpus_numeric = pca_transformer.transform(bow_corpus_numeric)

    con_labels=[0]*len(test_corpus)

    bow_corpus_numeric = bow_corpus_numeric.tolist()
    svm_classify_model= svm_load_model('src/libsvm/python/SVMmodel/{}.model.{}'.format(sys.argv[1],sys.argv[2]))
    lab_set = svm_classify_model.get_labels()
    p_labels,p_acc,p_val=svm_predict(con_labels,bow_corpus_numeric,svm_classify_model,'-b 1')

with open('output/{}_asp.out.{}'.format(sys.argv[1],sys.argv[2]),'w') as ans:
    for p_dis in p_val:
        ans_list=[]
        for i,p in enumerate(p_dis):
            if(p >= float(sys.argv[3])): # threshold
                ans_list.append(lab_map[lab_set[i]])
        ans.write('{}\n'.format(' '.join(ans_list)))
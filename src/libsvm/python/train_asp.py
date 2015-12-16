#!/usr/bin/env python
# -*- coding: utf-8 -*-

from svmutil import *
from sklearn.feature_extraction.text import CountVectorizer
import sys

lab_map={}
lab_set=[]

with open('misc_data/categoryMap/{}.category'.format(sys.argv[1])) as category_file:
    categories = category_file.read().splitlines()
    for idx,category in enumerate(categories):
        lab_map[category]=idx

# build bag of words analyzer
vectorizer = CountVectorizer(min_df=1)
with open('misc_data/{}_train.asp.dat'.format(sys.argv[1])) as train_data:
    train_corpus = train_data.read().splitlines()
    bow_matrix = vectorizer.fit_transform(train_corpus)
    
    bow_corpus_numeric = bow_matrix.toarray()

    train_labels=[]
    with open('misc_data/{}_train.asp.label'.format(sys.argv[1])) as label_file:
        labels = label_file.read().splitlines()
        for label in labels:
            train_labels.append(lab_map[label.split(',')[1]])

    bow_corpus_numeric = bow_corpus_numeric.tolist()
    problem = svm_problem(train_labels,bow_corpus_numeric)
    svm_classify_model = svm_train(train_labels,bow_corpus_numeric,'-t 0 -b 1 -q')
    svm_save_model('src/libsvm/SVMmodel/{}.model'.format(sys.argv[1]),svm_classify_model)

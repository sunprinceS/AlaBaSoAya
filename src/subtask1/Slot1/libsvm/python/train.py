#!/usr/bin/env python
# -*- coding: utf-8 -*-

from svmutil import *
from sklearn.feature_extraction.text import CountVectorizer
import sys

DATA_PATH = '../../data'
lab = open('{}/{}.lab'.format(DATA_PATH,sys.argv[1]))
lab_map={}

with open('{}/{}.category'.format(DATA_PATH,sys.argv[1])) as category_file:
    categories = category_file.read().splitlines()
    for idx,category in enumerate(categories):
        lab_map[category]=idx

# build bag of words analyzer
vectorizer = CountVectorizer(min_df=1)
with open('{}/{}.dat'.format(DATA_PATH,sys.argv[1])) as train_data:
    train_corpus = train_data.read().splitlines()
    bow_matrix = vectorizer.fit_transform(train_corpus)
    print("Total words in {}.dat : {}".format(sys.argv[1],len(vectorizer.get_feature_names())))
    print("Total aspect in {}.lab : {}".format(sys.argv[1],len(lab_map)))
    
    bow_corpus_numeric = bow_matrix.toarray()

    train_labels=[]
    with open('{}/{}.lab'.format(DATA_PATH,sys.argv[1])) as label_file:
        labels = label_file.read().splitlines()
        for label in labels:
            train_labels.append(lab_map[label.split(',')[1]])

    bow_corpus_numeric = bow_corpus_numeric.tolist()
    problem = svm_problem(train_labels,bow_corpus_numeric)
    svm_classify_model = svm_train(train_labels[:int(4*len(train_labels)/5)],bow_corpus_numeric[:int(4*len(train_labels)/5)],'-t 0 -b 1 -q')
    svm_save_model('model/{}.model'.format(sys.argv[1]),svm_classify_model)
    p_label,p_acc,p_val=svm_predict(train_labels[int(4*len(train_labels)/5):],bow_corpus_numeric[int(4*len(train_labels)/5):],svm_classify_model,'-b 1')

with open('prediction/{}.prob'.format(sys.argv[1]),'w') as ans:
    for p_dis in p_val:
        for p in p_dis:
            ans.write('{} '.format(p))
        ans.write('\n')

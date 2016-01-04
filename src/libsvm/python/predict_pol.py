#!/usr/bin/env python
# -*- coding: utf-8 -*-

from svmutil import *
from svm import *
from sklearn.feature_extraction.text import CountVectorizer
import sys

DATA_PATH = 'Files'
lab_map={'positive':1,'negative':2,'neutral':3,'conflict':4}
lab_set=[]

# build bag of words analyzer
vectorizer = CountVectorizer(min_df=1)
with open('{}/tr.pol.dat'.format(DATA_PATH)) as train_data:
    train_corpus = train_data.read().splitlines()
    bow_matrix = vectorizer.fit_transform(train_corpus)
    # print("Total words in {}.dat : {}".format(sys.argv[1],len(vectorizer.get_feature_names())))
    # print("Total aspect in {}.lab : {}".format(sys.argv[1],len(lab_map)))
    
    bow_corpus_numeric = bow_matrix.toarray()

    train_labels=[]
    with open('{}/tr.pol.label'.format(DATA_PATH)) as label_file:
        labels = label_file.read().splitlines()
        for label in labels:
            train_labels.append(lab_map[label.split(',')[1]])

    bow_corpus_numeric = bow_corpus_numeric.tolist()

    problem = svm_problem(train_labels,bow_corpus_numeric)

    svm_classify_model = svm_train(train_labels,bow_corpus_numeric,'-t 0 -b 1 -q')

    svm_save_model('SVMmodel/{}.model'.format(sys.argv[1]),svm_classify_model)

    p_labels,p_acc,p_val=svm_predict(train_labels,bow_corpus_numeric,svm_classify_model,'-b 1')

with open('{}/Out.pol'.format(DATA_PATH),'w') as ans:
    for p_label in p_labels:
        if int(p_label) not in lab_set:
            lab_set.append(int(p_label))
    ans.write('labels '+' '.join(str(x) for x in lab_set)+'\n')

    for p_label,p_dis in zip(p_labels,p_val):
        ans.write('{} {}\n'.format(int(p_label),' '.join(str(p) for p in p_dis)))


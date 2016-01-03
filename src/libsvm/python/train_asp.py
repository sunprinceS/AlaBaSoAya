#!/usr/bin/env python
# -*- coding: utf-8 -*-

from svmutil import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import re
import sys

##Info##
lab_map={}
lab_set=[]

##Load categoryMap##
with open('misc_data/categoryMap/{}.category'.format(sys.argv[1])) as category_file:
    categories = category_file.read().splitlines()
    for idx,category in enumerate(categories):
        lab_map[category]=idx

# build bag of words analyzer
# vectorizer = CountVectorizer(min_df=1,stop_words='english',ngram_range=(1,2))
vectorizer = CountVectorizer(min_df=1,stop_words='english')
tfidf_transformer = TfidfTransformer()
# pca_transformer = PCA(n_components=3000)

##Loading training data##
with open('misc_data/{}_train.asp.dat.{}'.format(sys.argv[1],sys.argv[2])) as train_data:

    train_corpus = train_data.read().splitlines()

    #Build transformer
    bow_matrix = vectorizer.fit_transform(train_corpus)
    bow_matrix = tfidf_transformer.fit_transform(bow_matrix)
    bow_corpus_numeric = bow_matrix.toarray()
    # bow_corpus_numeric = pca_transformer.fit_transform(bow_corpus_numeric)
    print ("The shape of dt matrix is {}\n".format(bow_matrix.shape))

    #Load label
    train_labels=[]
    with open('misc_data/{}_train.asp.label.{}'.format(sys.argv[1],sys.argv[2])) as label_file:
        labels = label_file.read().splitlines()
        for label in labels:
            train_labels.append(lab_map[label.split(',')[1]])

    #start train SVM
    bow_corpus_numeric = bow_corpus_numeric.tolist()
    problem = svm_problem(train_labels,bow_corpus_numeric)
    svm_classify_model = svm_train(train_labels,bow_corpus_numeric,'-t 0 -b 1 -q')
    svm_save_model('src/libsvm/python/SVMmodel/{}.model.{}'.format(sys.argv[1],sys.argv[2]),svm_classify_model)

    #save transform model for further usage
    joblib.dump(vectorizer,'src/libsvm/python/transformModel/vec.{}.{}'.format(sys.argv[1],sys.argv[2]))
    joblib.dump(tfidf_transformer,'src/libsvm/python/transformModel/tfidf.{}.{}'.format(sys.argv[1],sys.argv[2]))
    # joblib.dump(pca_transformer,'src/libsvm/python/transformModel/pca.{}.{}'.format(sys.argv[1],sys.argv[2]))


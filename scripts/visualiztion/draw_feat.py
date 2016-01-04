#!/usr/bin/env python
# -*- coding: utf-8 -*-

#for restaurant visualization

from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
import re
import sys
import matplotlib.pyplot as pl
# import pylab


lab_map_train={}
lab_map_test={}
lab_set=[]

with open('misc_data/categoryMap/{}.category'.format(sys.argv[1])) as category_file:
    categories = category_file.read().splitlines()
    for idx,category in enumerate(categories):
        lab_map_train[category]=idx

# build bag of words analyzer
# vectorizer = CountVectorizer(min_df=1,stop_words=stopwords.words('english'))
vectorizer = CountVectorizer(min_df=1,stop_words='english')
tfidf_transformer = TfidfTransformer()
pca = PCA(n_components=2)

with open('misc_data/{}_train.asp.dat'.format(sys.argv[1])) as train_data:
    train_corpus = train_data.read().splitlines()
    bow_matrix = vectorizer.fit_transform(train_corpus)

    #add TF-idf
    bow_matrix = tfidf_transformer.fit_transform(bow_matrix)
    bow_corpus_numeric = bow_matrix.toarray()
    bow_corpus_numeric = pca.fit_transform(bow_corpus_numeric)

    train_labels=[]
    with open('misc_data/{}_train.asp.label'.format(sys.argv[1])) as label_file:
        labels = label_file.read().splitlines()
        for label in labels:
            train_labels.append(lab_map_train[label.split(',')[1]])
    
    colors =['b','g','g','g','r','r','r','c','y','y','y','k']
    markers=['+','o','^','D','o','^','D','+','+','s','o','+']

    for idx,(lab,color,mark) in enumerate(zip(train_labels,colors,markers)):
        pl.scatter(bow_corpus_numeric[idx][0],bow_corpus_numeric[idx][1],c=color,marker=mark,label=lab,alpha=0.5)
    pl.xlim(-0.15,0.15)
    pl.ylim(-0.4,0.2)
    show()
    # pl.savefig("{}.png".format(sys.argv[1]))

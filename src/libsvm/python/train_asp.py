#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from svmutil import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.externals import joblib
import numpy as np
import joblib as jl
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

if sys.argv[3] == 'bow':

    # build bag of words analyzer
    # vectorizer = CountVectorizer(min_df=1,stop_words='english',ngram_range=(1,2))
    vectorizer = CountVectorizer(min_df=1,stop_words='english')
    tfidf_transformer = TfidfTransformer()
    # pca_transformer = PCA(n_components=3000)
    # lda_transformer = LDA(n_topics=1000)

    ##Loading training data##
    with open('misc_data/{}_train.asp.dat.{}'.format(sys.argv[1],sys.argv[2])) as train_data:

        train_corpus = train_data.read().splitlines()

        #Build transformer
        bow_matrix = vectorizer.fit_transform(train_corpus)
        bow_matrix = tfidf_transformer.fit_transform(bow_matrix)
        # bow_matrix = lda_transformer.fit_transform(bow_matrix)
        # bow_corpus_numeric = pca_transformer.fit_transform(bow_corpus_numeric)
        bow_corpus_numeric = bow_matrix.toarray()
        # print ("The shape of dt matrix is {}\n".format(bow_matrix.shape))

        #Load label
        train_labels=[]
        with open('misc_data/{}_train.asp.label.{}'.format(sys.argv[1],sys.argv[2])) as label_file:
            labels = label_file.read().splitlines()
            for label in labels:
                train_labels.append(lab_map[label.split(',')[1]])

        #start train SVM
        bow_corpus_numeric = bow_corpus_numeric.tolist()
        problem = svm_problem(train_labels,bow_corpus_numeric)
        svm_classify_model = svm_train(train_labels,bow_corpus_numeric,'-s 0 -c 5 -t 2 -g 0.5 -e 0.1 -b 1 -q')
        svm_save_model('src/libsvm/python/SVMmodel/{}.model.{}.{}'.format(sys.argv[1],sys.argv[2],sys.argv[3]),svm_classify_model)

        #save transform model for further usage
        joblib.dump(vectorizer,'src/libsvm/python/transformModel/vec.{}.{}'.format(sys.argv[1],sys.argv[2]))
        joblib.dump(tfidf_transformer,'src/libsvm/python/transformModel/tfidf.{}.{}'.format(sys.argv[1],sys.argv[2]))
        # joblib.dump(pca_transformer,'src/libsvm/python/transformModel/pca.{}.{}'.format(sys.argv[1],sys.argv[2]))
        # joblib.dump(lda_transformer,'src/libsvm/python/transformModel/lda.{}.{}'.format(sys.argv[1],sys.argv[2]))

elif sys.argv[3] == 'wv':

    ##load glove vector
    vocab_dict={}
    with open('glove/vocab.txt') as vocab_file:
        vocab_data = vocab_file.read().splitlines()
        for line_no,vocab in enumerate(vocab_data):
            vocab_dict[vocab] = line_no

    glove_matrix = jl.load('glove/glove.840B.float32.emb')
    oov_vec = np.mean(glove_matrix,axis=0)
    glove_matrix = np.vstack([glove_matrix,oov_vec])
    # print(glove_matrix.shape)
    ##Loading training data##
    with open('misc_data/{}_train.asp.dat.{}'.format(sys.argv[1],sys.argv[2])) as train_data:

        train_corpus = train_data.read().splitlines()

        sent_vecs=[]
        for opinion in train_corpus:
            sent_vec = np.array([.0]*300)
            opinion_seg = opinion.split(' ')
            for word in opinion_seg:
                if(word in vocab_dict):
                    sent_vec += glove_matrix[vocab_dict[word]]
                else:
                    sent_vec += glove_matrix[len(glove_matrix)-1]
            sent_vecs.append((sent_vec/(len(opinion_seg))).tolist())


        #Load label
        train_labels=[]
        with open('misc_data/{}_train.asp.label.{}'.format(sys.argv[1],sys.argv[2])) as label_file:
            labels = label_file.read().splitlines()
            for label in labels:
                train_labels.append(lab_map[label.split(',')[1]])

        #start train SVM
        problem = svm_problem(train_labels,sent_vecs)
        svm_classify_model = svm_train(train_labels,sent_vecs,'-t 0 -b 1 -q')
        svm_save_model('src/libsvm/python/SVMmodel/{}.model.{}.{}'.format(sys.argv[1],sys.argv[2],sys.argv[3]),svm_classify_model)

elif sys.argv[3] == 'bow+wv':
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
        print ("The shape of dt matrix is {}\n".format(bow_matrix.shape))
        bow_corpus_numeric = bow_matrix.toarray().tolist()
        # bow_corpus_numeric = pca_transformer.fit_transform(bow_corpus_numeric)

        ##load glove vector
        vocab_dict={}
        with open('glove/vocab.txt') as vocab_file:
            vocab_data = vocab_file.read().splitlines()
            for line_no,vocab in enumerate(vocab_data):
                vocab_dict[vocab] = line_no

        glove_matrix = jl.load('glove/glove.840B.float32.emb')
        oov_vec = np.mean(glove_matrix,axis=0)
        glove_matrix = np.vstack([glove_matrix,oov_vec])


        sent_vecs=[]
        for opinion,bow_vec in zip(train_corpus,bow_corpus_numeric):
            sent_vec = np.array([.0]*300)
            opinion_seg = opinion.split(' ')
            for word in opinion_seg:
                if(word in vocab_dict):
                    sent_vec += glove_matrix[vocab_dict[word]]
                else:
                    sent_vec += glove_matrix[len(glove_matrix)-1]
            sent_vec = (sent_vec/(len(opinion_seg))).tolist()
            sent_vec.extend(bow_vec)
            sent_vecs.append(sent_vec)

        #Load label
        train_labels=[]
        with open('misc_data/{}_train.asp.label.{}'.format(sys.argv[1],sys.argv[2])) as label_file:
            labels = label_file.read().splitlines()
            for label in labels:
                train_labels.append(lab_map[label.split(',')[1]])

        #start train SVM
        problem = svm_problem(train_labels,sent_vecs)
        svm_classify_model = svm_train(train_labels,sent_vecs,'-t 0 -b 1 -q')
        svm_save_model('src/libsvm/python/SVMmodel/{}.model.{}.{}'.format(sys.argv[1],sys.argv[2],sys.argv[3]),svm_classify_model)

else:
    print('Unexpected type!',file=sys.stderr)
    sys.exit()

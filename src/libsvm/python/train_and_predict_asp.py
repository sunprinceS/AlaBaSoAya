#!/usr/bin/env python
# -*- coding: utf-8 -*-

from svmutil import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import PCA
from nltk.corpus import stopwords
import re
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import sys

class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self,doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

# nltk.download()
lab_map_train={}
lab_map_test={}
lab_set=[]

#Training stage
with open('misc_data/categoryMap/{}.category'.format(sys.argv[1])) as category_file:
    categories = category_file.read().splitlines()
    for idx,category in enumerate(categories):
        lab_map_train[category]=idx

with open('misc_data/categoryMap/{}.category'.format(sys.argv[1])) as category_file:
    categories = category_file.read().splitlines()
    for idx,category in enumerate(categories):
        lab_map_test[idx]=category
# build bag of words analyzer
# vectorizer = CountVectorizer(min_df=1,stop_words=stopwords.words('english'))
vectorizer = CountVectorizer(min_df=1,stop_words='english')
tfidf_transformer = TfidfTransformer()
pca = PCA(n_components=1000)

with open('misc_data/{}_train.asp.dat'.format(sys.argv[1])) as train_data:
    train_corpus = train_data.read().splitlines()
    # train_corpus=[]
    # for sentence in tmp_train_corpus:
        # print(sentence)
        # train_corpus.append(re.sub("[0-9]","",sentence))
    # train_corpus  = re.sub("[0-9]","",train_corpus)
    bow_matrix = vectorizer.fit_transform(train_corpus)
    #add TF-idf
    bow_matrix = tfidf_transformer.fit_transform(bow_matrix)
    with open('{}.vocab'.format(sys.argv[1]),'w') as vocab:
        vocab.write('\n'.join(vectorizer.get_feature_names()))
    bow_corpus_numeric = bow_matrix.toarray()
    bow_corpus_numeric = pca.fit_transform(bow_corpus_numeric)

    train_labels=[]
    with open('misc_data/{}_train.asp.label'.format(sys.argv[1])) as label_file:
        labels = label_file.read().splitlines()
        for label in labels:
            train_labels.append(lab_map_train[label.split(',')[1]])

    bow_corpus_numeric = bow_corpus_numeric.tolist()
    problem = svm_problem(train_labels,bow_corpus_numeric)
    svm_classify_model = svm_train(train_labels,bow_corpus_numeric,'-t 0 -b 1 -q')
    svm_save_model('src/libsvm/SVMmodel/{}.model'.format(sys.argv[1]),svm_classify_model)


#Predicting stage
with open('misc_data/{}_te.asp.dat'.format(sys.argv[1])) as test_data:
    test_corpus = test_data.read().splitlines()
    bow_matrix = vectorizer.transform(test_corpus)

    bow_matrix = tfidf_transformer.transform(bow_matrix)
    bow_corpus_numeric = bow_matrix.toarray()
    bow_corpus_numeric = pca.transform(bow_corpus_numeric)

    con_labels=[0]*len(test_corpus)

    bow_corpus_numeric = bow_corpus_numeric.tolist()
    svm_classify_model= svm_load_model('src/libsvm/SVMmodel/{}.model'.format(sys.argv[1]))
    lab_set = svm_classify_model.get_labels()
    p_labels,p_acc,p_val=svm_predict(con_labels,bow_corpus_numeric,svm_classify_model,'-b 1')
with open('{}_asp.prob'.format(sys.argv[1]),'w') as prob:
    prob.write('labels {}\n'.format(' '.join(str(i) for i in lab_set)))
    for label , val in zip(p_labels,p_val):
        prob.write('{} {}\n'.format(label,' '.join(str(i) for i in val)))

# with open('output/{}_asp.out'.format(sys.argv[1]),'w') as ans:
    # # Among this label , a label is only occur at once (MULTIMEDIA_DEVICE#MISC) , drop it!
    # for lab in p_labels:
        # ans.write('{}\n'.format(lab_map_test[int(lab)]))

with open('output/{}_asp.out'.format(sys.argv[1]),'w') as ans:
    # Among this label , a label is only occur at once (MULTIMEDIA_DEVICE#MISC) , drop it!
    for p_dis in p_val:
        ans_list=[]
        for i,p in enumerate(p_dis):
            if(p >= float(sys.argv[2])): # threshold
                # print(lab_set[i])
                ans_list.append(lab_map_test[lab_set[i]])
        ans.write('{}\n'.format(' '.join(ans_list)))

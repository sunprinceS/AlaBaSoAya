"""
File: transformer.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/sunprinceS
Description: turn text into numric matrix
"""

import sys
import joblib as jl
import numpy as np
import scipy as sp

from . import io,marcos


def BOWTransformer():

    preprocess_pipe = marcos.PREPROCESS_PIPE

    #set bag of word's param
    preprocess_pipe.set_params(bow__min_df=1)
    preprocess_pipe.set_params(bow__stop_words='english')

    #set tf-idf param
    # feat_pipe.set_params(tfidf__?=)

    #set PCA transformer
    # dim_reduc_pipe.set__params(pca__n_components=3000)

    #set LDA transformer
    # dim_reduc_pipe.set__params(lda__n_topics=1000)

    return preprocess_pipe

def BOWtransform(corpus,mode,domain,trans_type,cross_val):
    data_matrix=[]

    if mode == 'train':
        bow_transformer = BOWTransformer()
        data_matrix = bow_transformer.fit_transform(corpus)

        #save transform model
        jl.dump(bow_transformer,'{}/{}.{}_model.{}'.format(marcos.TRANSFORM_MODEL_DIR,domain,trans_type,cross_val))

    elif mode == 'te':
        bow_transformer = jl.load('{}/{}.{}_model.{}'.format(marcos.TRANSFORM_MODEL_DIR,domain,trans_type,cross_val))
        data_matrix = bow_transformer.transform(corpus)

    else:
        print("Unexpected mode in BOWtransform",file=sys.stderr)

    # turn dt matrix to list
    print ("The shape of dt matrix is {}\n".format(data_matrix.shape))
    if sp.sparse.isspmatrix_csr(data_matrix):
        data_matrix = data_matrix.toarray().tolist()
    else: #pass through dimension reduction pipe
        data_matrix = data_matrix.tolist()

    return data_matrix

def addAspect(treelstm_vec_array,asp_map,asp_list):
    # print(treelstm_vec_array.shape)
    # print(len(asp_list))
    asp_array = np.zeros((len(asp_list),1), 'float32')
    for idx,asp in enumerate(asp_list):
        asp_array[idx] += float(asp_map[asp])
    ret = np.hstack((treelstm_vec_array,asp_array))
    return (np.hstack((treelstm_vec_array,asp_array))).tolist()


def gloveTransform(corpus):

    vocab_dict = io.loadVocabDict()
    glove_matrix = io.loadGloveVec()

    data_matrix=[]

    for sentence in corpus:
        sent_vec= np.array([.0]*300)
        sentence_seg = sentence.split(' ')
        for word in sentence_seg:
            if word in vocab_dict:
                sent_vec += glove_matrix[vocab_dict[word]]
            else:
                sent_vec += glove_matrix[-1]
        data_matrix.append((sent_vec/len(sentence_seg)).tolist())

    return data_matrix

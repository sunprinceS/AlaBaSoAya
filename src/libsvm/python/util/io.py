"""
File: io.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/sunprinces
Description: basic i/o
"""

import joblib as jl
import numpy as np
import scipy as sp
import sys

from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer

from . import marcos

def loadMap(domain,mode):
    with open('{}/{}.category'.format(marcos.CAT_DIR,domain)) as category_file:
        categories = category_file.read().splitlines()
        lab_map={}
        if mode == 'train' or mode == 'test_pol' or mode=='valid':
            for idx,category in enumerate(categories):
                lab_map[category]=idx
        elif mode == 'te':
            for idx,category in enumerate(categories):
                lab_map[idx]=category
        else:
            print('Unexpected mode in loadMap {}'.format(mode),file=sys.stderr)
            sys.exit()

        return lab_map

def loadDat(domain,cross_val,mode,task):
    corpus=[]
    with open('{}/{}_{}.{}.dat.{}'.format(marcos.MISC_DIR,domain,mode,task,cross_val)) as file:
        corpus=file.read().splitlines()
    return corpus

def loadLabel(domain,cross_val,mode,task,lab_map):
    golden=[]
    with open('{}/{}_{}.{}.label.{}'.format(marcos.MISC_DIR,domain,mode,task,cross_val)) as file:
        labels=file.read().splitlines()
        if task == 'asp':
            for label in labels:
                golden.append(lab_map[label.split(',')[1]])
        elif task =='pol':
            for label in labels:
                golden.append(lab_map[label.split(',')[1]])
        else:
            print('Unexpected task {}!'.format(task),file=sys.stderr)
    return golden

def loadVocabDict():
    vocab_dict={}
    with open('glove/vocab.txt') as vocab_file:
        vocab_data = vocab_file.read().splitlines()
        for line_no,vocab in enumerate(vocab_data):
            vocab_dict[vocab] = line_no

    return vocab_dict

def loadGloveVec():

    glove_matrix = jl.load('glove/glove.840B.float32.emb')
    oov_vec = np.mean(glove_matrix,axis=0)
    glove_matrix = np.vstack([glove_matrix,oov_vec])

    return glove_matrix
def loadTreeLSTMVec(domain,mode,idx):
    treelstm_matrix=[]
    with open('{}/{}_{}.pol.feat.{}'.format(marcos.MISC_DIR,domain,mode,idx)) as treelstm_dat:
        treelstm = treelstm_dat.read().splitlines()
        for vec in treelstm:
            tmp = np.array(vec.split(' ')).astype(np.float32)
            treelstm_matrix.append(tmp)
    return np.array(treelstm_matrix)

def loadAsp(domain,mode,idx):
    asp_list = []
    with open('{}/{}_{}.pol.goldenAsp.{}'.format(marcos.MISC_DIR,domain,mode,idx)) as asp_file:
        asp_list = asp_file.read().splitlines()
    return asp_list

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

from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer

from . import marcos

def loadMap(domain,mode):
    with open('{}/{}.category'.format(marcos.CAT_DIR,domain)) as category_file:
        categories = category_file.read().splitlines()
        lab_map={}
        if mode == 'train':
            for idx,category in enumerate(categories):
                lab_map[category]=idx
        elif mode == 'te':
            for idx,category in enumerate(categories):
                lab_map[idx]=category
        else:
            print('Unexpected mode in loadMap',file=sys.stderr)
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

from __future__ import print_function
import numpy as np
from itertools import zip_longest
from . import marcos

def loadDat(domain,cross_val,mode,task):
    corpus=[]
    if mode == 'train' or mode == 'valid':
        with open('{}/data/{}_{}.{}.extractDat.{}'.format(marcos.NN_DIR, \
                domain,mode,task,cross_val)) as file:
            corpus=file.read().splitlines()
    elif mode == 'te':
        with open('{}/{}_te.{}.dat.{}'.format(marcos.MISC_DIR, domain,task,cross_val)) as file:
            corpus=file.read().splitlines()
    else:
        print("Unexpected mode : {}".format(mode))
    return corpus

def makeBatch(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def loadLabel(domain,cross_val,mode,task,lab_map):
    golden=[]
    with open('{}/data/{}_{}.{}.extractLab.{}'.format(marcos.NN_DIR, \
            domain,mode,task,cross_val)) as file:

        labels_list=file.read().splitlines()

        for labels_str in labels_list:
            labels = labels_str.split(' ')
            label_encoding = np.zeros(len(lab_map))
            for label in labels:
                label_encoding[int(label)] += 1
            golden.append(label_encoding)
    return golden

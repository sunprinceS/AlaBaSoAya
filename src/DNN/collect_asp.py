#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from util import *

def main():
    corpus_extract = []
    labels_extract = []
    # collect training data
    lab_map = io.loadMap(sys.argv[1],sys.argv[3])
    corpus = io.loadDat(sys.argv[1],sys.argv[2],sys.argv[3],'asp')
    labels = io.loadLabel(sys.argv[1],sys.argv[2],sys.argv[3],'asp',lab_map)

    assert (len(corpus) == len(labels))

    prev_sentence=corpus[0]
    lab_set=[]
    lab_set.append(labels[0])
    for sentence,label in zip(corpus,labels):
        if sentence != prev_sentence:
            corpus_extract.append(prev_sentence)
            labels_extract.append(lab_set)
            lab_set = []
            lab_set.append(label)
        else:
            if label not in lab_set:
                lab_set.append(label)
        prev_sentence = sentence
    with open('{}/data/{}_{}.asp.extractDat.{}'.format(marcos.NN_DIR, \
            sys.argv[1],sys.argv[3],sys.argv[2]),'w') as data:
        data.write('\n'.join(corpus_extract))
    with open('{}/data/{}_{}.asp.extractLab.{}'.format(marcos.NN_DIR, \
            sys.argv[1],sys.argv[3],sys.argv[2]),'w') as label:
        for l in labels_extract :
            lala = ' '.join(str(x) for x in l)
            label.write('{}\n'.format(lala))

if __name__ == "__main__":
    main()

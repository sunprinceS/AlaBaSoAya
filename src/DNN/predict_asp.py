#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys
import argparse
import joblib
import time
import signal
import random

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.utils import generic_utils
from keras.optimizers import Adagrad

from util import *

def main():

    start_time = time.time()

    # argument parser
    parser = argparse.ArgumentParser(prog='predict_asp.py',
            description='predict NN model for ABSA multi-category classification')
    parser.add_argument('--transform',type=str,required=True,choices=['bow','wv','bow+wv'],metavar='<transform method>')
    parser.add_argument('--threshold',type=float,required=True,metavar='<threshold to choose>')
    parser.add_argument('--domain', type=str, required=True, choices=['rest','lapt'], metavar='<domain>')
    parser.add_argument('--cross_val', type=str, required=True, metavar='<cross-validation-index>')
    args = parser.parse_args()
    batch_size=1
    #############
    # transform #
    #############
    te_matrix=[] #actually,it's the valid matrix used in training

    #load category map
    lab_map=io.loadMap(args.domain,'te')

    #Loading data
    te_corpus = nn.loadDat(args.domain,args.cross_val,'te','asp')

    if args.transform == 'bow':
        te_matrix = transform.BOWtransform(te_corpus,'te',\
                args.domain,args.transform,args.cross_val)

    elif args.transform == 'wv':

        te_matrix = transform.gloveTransform(te_corpus)

    elif args.transform == 'bow+wv':

        te_matrix = transform.BOWtransform(te_corpus,'train',\
                args.domain,args.transform,args.cross_val)
        wv_matrix = transform.gloveTransform(te_corpus)
        for vec,wv_vec in zip(te_matrix,wv_matrix):
            vec.extend(wv_vec)
    else:
        print('Unexpected transform method {}'.format(args.transform),file=sys.stderr)
        sys.exit(-1)

    #######################
    #      Load Model     #
    #######################
    print('Loading model and weights...')
    model = model_from_json(open('{}/aspModel/{}/{}.{}.json'.format(marcos.NN_DIR,args.domain, \
        marcos.MODEL,args.cross_val),'r').read())
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    model.load_weights('{}/aspModel/{}/{}.{}_best.hdf5'.format(marcos.NN_DIR,args.domain,\
        marcos.MODEL,args.cross_val))
    print('Model and weights loaded.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #    Make Batches    #
    ######################
    print('Making batches...')
    # te batches
    te_batches = [ b for b in nn.makeBatch(te_matrix, batch_size, fillvalue=te_matrix[-1]) ]

    print('Finished making batches.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #      Testing       #
    ######################

    # start testing
    print('Testing started...')
    pbar = generic_utils.Progbar(len(te_matrix)*batch_size)

    predictions = []
    # testing feedforward
    for i in range(len(te_batches)):
        X_sent_batch = np.array(te_batches[i])
        pval = model.predict_proba(X_sent_batch,batch_size,verbose=0)
        pred=[]
        for cnt,p in enumerate(pval[0]):
            if p >= args.threshold:
                pred.append(lab_map[cnt])
        predictions.append(pred)
        pbar.add(batch_size)

    with open('output/{}_asp.out.{}'.format(args.domain,args.cross_val),'w') as ans:
        for line in predictions:
            ans.write('{}\n'.format(' '.join(line)))

    print('Testing finished.')
    print('Time: %f s' % (time.time()-start_time))

if __name__ == "__main__":
    main()


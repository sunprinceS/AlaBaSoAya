#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
import sys
import argparse
import joblib
import time
import signal
import random

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.utils import generic_utils
from keras.optimizers import Adagrad, RMSprop
from util import *

def main():
    start_time = time.time()

    # argument parser
    parser = argparse.ArgumentParser(prog='train_asp.py',
            description='Train NN model for ABSA multi-category classification')
    parser.add_argument('--num_neurons', type=int, default=200, metavar='<mlp-hidden-units>')
    parser.add_argument('--num_layers', type=int, default=2, metavar='<mlp-hidden-layers>')
    parser.add_argument('--dropout', type=float, default=0.5, metavar='<dropout-rate>')
    parser.add_argument('--activation', type=str, default='relu', metavar='<activation-function>')
    parser.add_argument('--transform',type=str,required=True,choices=['bow','wv','bow+wv'],metavar='<transform method>')
    parser.add_argument('--num_epochs', type=int, default=100, metavar='<num-epochs>')
    parser.add_argument('--batch_size', type=int, default=64, metavar='<batch-size>')
    parser.add_argument('--learning_rate', type=float, default=0.0001, metavar='<learning-rate>')
    parser.add_argument('--domain', type=str, required=True, choices=['rest','lapt'], metavar='<domain>')
    parser.add_argument('--cross_val', type=str, required=True, metavar='<cross-validation-index>')
    args = parser.parse_args()

    #############
    # transform #
    #############

    train_matrix=[]
    valid_matrix=[]
    train_labels=[]
    valid_labels=[]

    #load category map
    lab_map=io.loadMap(args.domain,'train')

    #Loading data
    train_corpus = nn.loadDat(args.domain,args.cross_val,'train','asp')
    valid_corpus = nn.loadDat(args.domain,args.cross_val,'valid','asp')

    #Load label
    train_labels=nn.loadLabel(args.domain,args.cross_val,'train','asp',lab_map)
    valid_labels=nn.loadLabel(args.domain,args.cross_val,'valid','asp',lab_map)

    if args.transform == 'bow':
        train_matrix = transform.BOWtransform(train_corpus,'train',\
                args.domain,args.transform,args.cross_val)
        valid_matrix = transform.BOWtransform(valid_corpus,'valid',\
                args.domain,args.transform,args.cross_val)

    elif args.transform == 'wv':

        train_matrix = transform.gloveTransform(train_corpus)
        valid_matrix = transform.gloveTransform(valid_corpus)

    elif args.transform == 'bow+wv':

        train_matrix = transform.BOWtransform(train_corpus,'train',\
                args.domain,args.transform,args.cross_val)
        wv_matrix = transform.gloveTransform(train_corpus)
        for vec,wv_vec in zip(train_matrix,wv_matrix):
            vec.extend(wv_vec)

        valid_matrix = transform.BOWtransform(valid_corpus,'valid',\
                args.domain,args.transform,args.cross_val)
        wv_matrix = transform.gloveTransform(valid_corpus)
        for vec,wv_vec in zip(valid_matrix,wv_matrix):
            vec.extend(wv_vec)
    else:
        print('Unexpected transform method {}'.format(args.transform),file=sys.stderr)
        sys.exit(-1)

    ######################
    # Model Descriptions #
    ######################
    feat_num = len(train_matrix[0])
    num_class = len(lab_map)

    # print('Generating and compiling model...')

    # feedforward model (MLP)
    model = Sequential()
    model.add(Dense(args.num_neurons,input_dim=feat_num,init='uniform',activation=args.activation))
    model.add(Dropout(args.dropout))
    for l in range(args.num_layers):
        model.add(Dense(args.num_neurons,init='uniform',activation=args.activation))
        model.add(Dropout(args.dropout))
    model.add(Dense(num_class,init='uniform',activation='softmax'))

    # save model configuration
    json_string = model.to_json()
    model_filename='{}/aspModel/{}/{}layers_{}neurons_{}dropout_{}activation.{}'.format( \
            marcos.NN_DIR,args.domain,args.num_layers,args.num_neurons,\
            args.dropout,args.activation,args.cross_val)
    open(model_filename + '.json', 'w').write(json_string)

    # loss and optimizer
    rmsprop = RMSprop(lr=args.learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=rmsprop)
    # print('Compilation finished.')
    # print('Time: %f s' % (time.time()-start_time))

    ######################
    #    Make Batches    #
    ######################
    # print('Making batches...')

    # training batches
    train_batches = [ b for b in nn.makeBatch(train_matrix, \
            args.batch_size, fillvalue=train_matrix[-1]) ]
    train_lab_batches = [ b for b in nn.makeBatch(train_labels, \
            args.batch_size, fillvalue=train_labels[-1]) ]
    train_indices = list(range(len(train_batches)))

    # validation batches
    valid_batches = [ b for b in nn.makeBatch(valid_matrix,\
            args.batch_size, fillvalue=valid_matrix[-1]) ]
    valid_lab_batches = [ b for b in nn.makeBatch(valid_labels,\
            args.batch_size, fillvalue=valid_labels[-1]) ]
    
    # print('Finished making batches.')
    # print('Time: %f s' % (time.time()-start_time))

    ######################
    #      Training      #
    ######################
    dev_biases = []
    min_bias = 1e9
    min_bias_epoch = -1

    # define interrupt handler
    def PrintDevBias():
        print('Min validation Bias epoch: %i' % min_bias_epoch)
        print(dev_biases)

    def InterruptHandler(sig, frame):
        print(str(sig))
        PrintDevBias()
        sys.exit(-1)

    signal.signal(signal.SIGINT, InterruptHandler)
    signal.signal(signal.SIGTERM, InterruptHandler)

    # print training information
    # print('-'*80)
    # print('Training Information')
    # print('# of MLP hidden units: %i' % args.num_neurons)
    # print('# of MLP hidden layers: %i' % args.num_layers)
    # print('Dropout: %f' % args.dropout)
    # print('MLP activation function: %s' % args.activation)
    # print('# of training epochs: %i' % args.num_epochs)
    # print('Batch size: %i' % args.batch_size)
    # print('Learning rate: %f' % args.learning_rate)
    # print('-'*80)

    # start training
    # print('Training started...')
    for k in range(args.num_epochs):
        # print('-'*80)
        # print('Epoch %i' % (k+1))
        # progbar = generic_utils.Progbar(len(train_indices)*args.batch_size)
        # shuffle batch indices
        random.shuffle(train_indices)
        for i in train_indices:
            X_sent_batch = np.array(train_batches[i])
            Y_lab_batch =  np.array(train_lab_batches[i])
            loss = model.train_on_batch(X_sent_batch,Y_lab_batch)
            loss = loss[0].tolist()
            # progbar.add(args.batch_size, values=[('train loss', loss)])
        # print('Time: %f s' % (time.time()-start_time))

        # evaluate on dev set
        # pbar = generic_utils.Progbar(len(valid_batches)*args.batch_size)

        # validation feedforward
        dev_bias = 0
        for i in range(len(valid_batches)):
            X_sent_batch = np.array(valid_batches[i])
            Y_lab_batch = np.array(valid_lab_batches[i])
            pred = model.predict_proba(X_sent_batch,args.batch_size,verbose=0)

            if i != (len(valid_batches)-1):
                for one_pred,lab in zip(pred,Y_lab_batch):
                    zero_idx=[]
                    for idx,ii in enumerate(lab):
                        if lab[ii] == 0:
                            zero_idx.append(idx)
                    # print("ONE {}".format(one_pred.shape))
                    # print("LAB {}".format(lab.shape))
                    dev_bias += np.sum((one_pred[zero_idx] - lab[zero_idx])**2)

            # pbar.add(args.batch_size)

        # calculate validation accuracy
        dev_bias = float(dev_bias)/len(valid_matrix)
        dev_biases.append(dev_bias)
        # print('Validation Bias: %f' % dev_bias)
        # print('Time: %f s' % (time.time()-start_time))

        # save best weights
        if dev_bias < min_bias:
            min_bias = dev_bias
            min_bias_epoch = k
            model.save_weights(model_filename + '_best.hdf5', overwrite=True)

    # print(dev_biases)
    # print('Best Valid Biases: %f; epoch#%i' % (min_bias, (min_bias_epoch+1)))
    # print('Training finished.')
    print('Time: %f s' % (time.time()-start_time))

if __name__ == "__main__":
    main()

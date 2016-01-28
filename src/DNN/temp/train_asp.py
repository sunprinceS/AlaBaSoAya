#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
from io import StringIO
from settings import *
import sys
import lasagne

def main():

    data_matrix=[]
    train_labels=[]

    #load category map
    lab_map=io.loadMap(sys.argv[1],'train')

    #Loading training data
    train_corpus = nn.loadDat(sys.argv[1],sys.argv[2],'train','asp')

    #Load label
    train_labels=nn.loadLabel(sys.argv[1],sys.argv[2],'train','asp',lab_map)

    if sys.argv[3] == 'bow':

        data_matrix = transform.BOWtransform(train_corpus,'train',sys.argv[1],sys.argv[3],sys.argv[2])

    elif sys.argv[3] == 'wv':

        data_matrix = transform.gloveTransform(train_corpus)

    elif sys.argv[3] == 'bow+wv':

        data_matrix = transform.BOWtransform(train_corpus,'train',sys.argv[1],sys.argv[3],sys.argv[2])

        wv_matrix = transform.gloveTransform(train_corpus)

        #concat bow and glove vector
        for vec,wv_vec in zip(data_matrix,wv_matrix):
            vec.extend(wv_vec)

    else:
        print('Unexpected type!',file=sys.stderr)
        sys.exit()



    #neural network
    input_var = T.matrix('inputs')
    target_var = T.matrix('labels')

    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_model(inputVar=input_var)

    # sys.argv[1] : dom
    # sys.argv[2] : cross_val
    # sys.argv[3] : transform type
    # sys.argv[4] : acc_model
    # load previous model
    if len(sys.argv) == 5:
        print("loading model {}".format(sys.argv[4]))
        with np.load('{}/aspModel/{}/{}.npz.{}.{}'.format(marcos.NN_DIR,\
                sys.argv[1],sys.argv[4],sys.argv[3],sys.argv[2])) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    pred = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(pred, target_var)
    loss = loss.mean()
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=LEARNING_RATE, momentum=MOMENTUM)


    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    # test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      # dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, shuffle=True):
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1

        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        for batch in iterate_minibatches(X_val, y_val,200 , shuffle=False):
            inputs, targets = batch
            err, acc = val_fn(inputs, targets)
            val_err += err
            val_acc += acc
            val_batches += 1

        # Then we print the results for this epoch:
        print("Epoch {} took {:.3f}s".format(epoch + 1, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(val_acc / val_batches * 100))
        if(epoch%10==0):
            np.savez('model/{}/tmp/{}x{:.2f}.npz'.format(sys.argv[1],epoch,val_acc / val_batches * 100), *lasagne.layers.get_all_param_values(network))


if __name__ == '__main__':
    main()


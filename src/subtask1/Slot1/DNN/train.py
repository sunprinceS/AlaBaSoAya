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


def load_dataset():

    train_data = open("DATA/{}.in".format(sys.argv[1]))
    train_lab = open("DATA/{}.lab".format(sys.argv[1]))
    X_train=[]
    y_train=[]
    for x,y in zip(train_data.read().splitlines(),train_lab.read().splitlines()):
        X_train.append(np.loadtxt(StringIO(x),dtype=np.float32))
        y_train.append(np.loadtxt(StringIO(y),dtype=np.int32))

    X_train = np.asarray(X_train)
    y_train = np.array(y_train)
    assert len(X_train) == len(y_train)
    X_train, X_val = X_train[:int(len(X_train)*4/5)], X_train[int(-len(X_train)*4/5):]
    y_train, y_val = y_train[:int(len(y_train)*4/5)], y_train[int(-len(y_train)*4/5):]

    train_data.close()
    train_lab.close()
    return X_train, y_train, X_val, y_val


def iterate_minibatches(inputs, targets, batchsize=BATCH_SIZE, shuffle=False):
    assert len(inputs) == len(targets)
    inputs = np.array(inputs)
    targets = np.array(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield (inputs[excerpt], targets[excerpt])


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main():
    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val, y_val = load_dataset()
    

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('labels')
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_model(inputVar=input_var)
    
    #load previous model
    if len(sys.argv) == 3:
        print("loading model".format(sys.argv[2]))
        with np.load('model/{}/{}.npz'.format(sys.argv[1],sys.argv[2])) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)

    # Create a loss expression for training, i.e., a scalar objective we want
    # to minimize (for our multi-class problem, it is the cross-entropy loss):
    pred = lasagne.layers.get_output(network)
    loss = lasagne.objectives.categorical_crossentropy(pred, target_var)
    loss = loss.mean()
    # We could add some weight decay as well here, see lasagne.regularization.

    # Create update expressions for training, i.e., how to modify the
    # parameters at each training step. Here, we'll use Stochastic Gradient
    # Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=LEARNING_RATE, momentum=MOMENTUM)

    # Create a loss expression for validation/testing. The crucial difference
    # here is that we do a deterministic forward pass through the network,
    # disabling dropout layers.
    test_prediction = lasagne.layers.get_output(network, deterministic=True)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    # As a bonus, also create an expression for the classification accuracy:
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)

    train_fn = theano.function([input_var, target_var], loss, updates=updates)

    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])

    # Finally, launch the training loop.
    print("Starting training...")
    # We iterate over epochs:
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
        print("Epoch {} took {:.3f}s".format(
            epoch + 1, time.time() - start_time))
        print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
        print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
        print("  validation accuracy:\t\t{:.2f} %".format(
            val_acc / val_batches * 100))
        if(epoch%10==0):
            np.savez('model/{}/tmp/{}x{:.2f}.npz'.format(sys.argv[1],epoch,val_acc / val_batches * 100), *lasagne.layers.get_all_param_values(network))


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on ABSA subtask1 using Lasagne.")
        print("Usage: %s " % sys.argv[0])
        print()
    else:
        main()


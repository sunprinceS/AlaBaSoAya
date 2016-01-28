#!/usr/bin/env python

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T
from io import StringIO
from settings_sent import *
import sys
import lasagne
import argparse


def load_dataset(train_dat_path, train_lab_path, test_dat_path):
    with open(train_dat_path, 'r') as train_data, \
         open(train_lab_path, 'r') as train_lab, \
         open(test_dat_path, 'r') as test_data:
        X_train = []
        y_train = []
        X_test = []
        for x,y in zip(train_data.read().splitlines(),train_lab.read().splitlines()):
            X_train.append(np.loadtxt(StringIO(x),dtype=np.float32))
            y = y.split(',')[1]
            if y == 'positive':
                lab = '2'
            elif y == 'neutral':
                lab = '1'
            elif y == 'negative':
                lab = '0'
            y_train.append(np.loadtxt(StringIO(lab),dtype=np.int32))
        for line in test_data.read().splitlines():
            X_test.append(np.loadtxt(StringIO(line), dtype=np.float32))

        X_train = np.asarray(X_train)
        y_train = np.asarray(y_train)
        X_test = np.asarray(X_test)
        assert len(X_train) == len(y_train)
        X_train, X_val = X_train[:int(len(X_train)*4/5)], X_train[int(-len(X_train)*4/5):]
        y_train, y_val = y_train[:int(len(y_train)*4/5)], y_train[int(-len(y_train)*4/5):]

        return X_train, y_train, X_val, y_val, X_test


def iterate_minibatches(inputs, targets=None, batchsize=BATCH_SIZE, shuffle=False):
    if targets is None:
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
    else:
        inputs = np.array(inputs)
        for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
            excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main():
    parser = argparse.ArgumentParser(prog='train_sent.py', description='Train DNN for ABSA sentiment subtask')
    parser.add_argument('--train-dat-path', type=str, required=True, metavar='<training data path>')
    parser.add_argument('--train-lab-path', type=str, required=True, metavar='<training label path>')
    parser.add_argument('--test-dat-path', type=str, required=True, metavar='<testing data path>')
    parser.add_argument('--pred-path', type=str, required=True, metavar='<predictions path>')
    args = parser.parse_args()

    # Load the dataset
    print("Loading data...")
    X_train, y_train, X_val = load_dataset(ars.train_dat_path, args.train_lab_path, args.test_dat_path)

    # Prepare Theano variables for inputs and targets
    input_var = T.matrix('inputs')
    target_var = T.ivector('labels')
    # Create neural network model (depending on first command line parameter)
    print("Building model and compiling functions...")
    network = build_model(inputVar=input_var)

    #load previous model
    if len(sys.argv) == 3:
        print("loading model".format(sys.argv[2]))
        with np.load('model_sent/{}/{}.npz'.format(sys.argv[1],sys.argv[2])) as f:
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
    test_fn = theano.function([input_var], test_prediction)

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
            np.savez('model_sent/{}/tmp/{}x{:.2f}.npz'.format(sys.argv[1],epoch,val_acc / val_batches * 100), *lasagne.layers.get_all_param_values(network))

    # testing data
    predictions = []
    for batch in iterate_minibatches(X_test, 200 , shuffle=False):
        inputs, targets = batch
        pred = test_fn(inputs, targets)
        predictions.extend(pred.tolist())
    with open(args.pred_path, 'w') as pred_file:
        for pred in predictions:
            if pred == 0:
                pred_file.write('negative\n')
            elif pred == 1:
                pred_file.write('neutral\n')
            else:
                pred_file.write('positive\n')


if __name__ == '__main__':
    if ('--help' in sys.argv) or ('-h' in sys.argv):
        print("Trains a neural network on ABSA subtask1 using Lasagne.")
        print("Usage: {} <category> [model]" .format(sys.argv[0]))
        print()
    else:
        main()


#!/usr/bin/env python
"""
File: train.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/sunprinceS
Description: train sentence embedding vector from to auto_encoder
"""

from __future__ import print_function
import sys
import numpy as np
import theano
import theano.tensor as T
import lasagne
import time
import random
from io import StringIO
from settings import *


def load_data(fin, data):
    OK = True
    lines = []
    for z in range(BATCH_SIZE):
        line = fin.readline()
        if not line:
            fin = open(data, "r")
            line = fin.readline()
            OK = False
        lines.append(np.loadtxt(StringIO(line), dtype=np.float32))

    lines = np.asarray(lines)
    Y = lines[:, 0].astype(np.int32)
    X = lines[:, 1:]

    return X, Y, OK, fin

def create_iter_functions(output_layer):
    X_batch = T.matrix('x')
    Y_batch = T.ivector('y')

    # objective = lasagne.objectives.categorical_crossentropy(output_layer)
    # objective = lasagne.objectives.Objective(output_layer, loss_function=lasagne.objectives.categorical_crossentropy)
    # loss_train = objective.get_loss(X_batch, target=Y_batch)

    all_params = lasagne.layers.get_all_params(output_layer,trainable=True)
    pred = T.argmax(lasagne.layers.get_output(output_layer, X_batch), axis = 1)
    loss_train = T.mean(T.nnet.categorical_crossentropy(pred,Y_batch))

    accuracy = T.mean(T.eq(pred,Y_batch), dtype=theano.config.floatX)
    # accuracy = T.eq(pred,Y_batch)

    # updates = lasagne.updates.rmsprop(loss_train, all_params, LEARNING_RATE)
    updates_sgd = lasagne.updates.sgd(loss_train,all_params,LEARNING_RATE)
    updates = lasagne.updates.apply_momentum(updates_sgd,all_params,MOMENTUM)

    iter_train = theano.function(
        [X_batch, Y_batch], accuracy, updates=updates,
    )

    iter_valid = theano.function(
        [X_batch, Y_batch], accuracy,
    )

    return {"train": iter_train, "valid": iter_valid}

def main():
    print("Building model and compile theano...")
    output_layer = build_model(input_dim = 4000, output_dim = 12)
    if len(sys.argv) == 3:
        print("loading model {}".format(sys.argv[2]))
        fin = open("model/{}/{}".format(sys.argv[1],sys.argv[2]),'rb')
        import pickle
        lasagne.layers.set_all_param_values(output_layer, pickle.load(fin))
    iter_funcs = create_iter_functions(output_layer)
    print("Training")
    trainin = open("DATA/{}.in".format(sys.argv[1]))
    validin = open("DATA/{}_valid.in".format(sys.argv[1]))
    epoch = 0
    while True:
        accu = 0
        cnt = 0
        now = time.time()
        while True:
            X, Y, OK, trainin = load_data(trainin, "DATA/"+sys.argv[1]+".in")
            if not OK:
                break
            accu_tmp = iter_funcs['train'](X, Y)
            accu += accu_tmp
            cnt += 1
        accu = accu / cnt
        print("Epoch {} took {:.3f}s\t{:.2f}".format(epoch+1, time.time() - now, accu * 100))

        accu = 0
        cnt = 0
        now = time.time()
        while True:
            X, Y, OK, validin = load_data(validin,"DATA/"+sys.argv[1]+"_valid.in")
            if not OK:
                break
            accu += iter_funcs['valid'](X, Y)
            cnt += 1
        accu = accu / cnt
        # print(cnt)
        print("Valid {} took {:.3f}s\t{:.2f}".format(epoch+1, time.time() - now, accu * 100))
        print()

        epoch += 1
        import pickle
        fout = open("model/{}/tmp/{}x{:.2f}".format(sys.argv[1],epoch, accu *100), "wb")
        pickle.dump(lasagne.layers.get_all_param_values(output_layer), fout)
        fout.close()

if __name__ == '__main__':
  main()


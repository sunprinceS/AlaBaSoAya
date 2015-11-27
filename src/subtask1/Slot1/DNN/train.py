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
from StringIO import StringIO
from settings import *

def load_transition(to=39):
    file = open("../../conf/state_48_39.map");
    trans = np.zeros((1943, to), dtype=np.float32)
    transmap = np.zeros((1943,), dtype=np.int32)
    lines = file.readlines()
    phones = dict()
    for i, line in enumerate(lines):
        lines[i] = line.split("\t")
        if to == 48:
            phones[lines[i][1]]=0
        else:
            phones[lines[i][2]]=0
    phones = collections.OrderedDict(sorted(phones.items()))
    for i, k in enumerate(phones):
        phones[k] = i
    for line in lines:
        if to == 48:
            trans[int(line[0]), phones[line[1]]] = 1
            transmap[int(line[0])] = phones[line[1]]
        else:
            trans[int(line[0]), phones[line[2]]] = 1
            transmap[int(line[0])] = phones[line[2]]
    return trans, transmap

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

def create_iter_functions(data, output_layer):
    X_batch = T.matrix('x')
    Y_batch = T.ivector('y')
    trans = T.matrix('trans')
    transmap = T.ivector('transmap')

    objective = lasagne.objectives.Objective(output_layer, loss_function=lasagne.objectives.categorical_crossentropy)

    all_params = lasagne.layers.get_all_params(output_layer)

    loss_train = objective.get_loss(X_batch, target=Y_batch)

    pred48 = T.argmax(T.dot(lasagne.layers.get_output(output_layer, X_batch, deterministic=True), trans), axis=1)
    # pred1943 = T.argmax(lasagne.layers.get_output(output_layer, X_batch, deterministic=True), axis = 1)
    accuracy48 = T.mean(T.eq(pred48, transmap[Y_batch]), dtype=theano.config.floatX)
    # accuracy1943 = T.mean(T.eq(pred1943, Y_batch), dtype=theano.config.floatX)


    updates = lasagne.updates.rmsprop(loss_train, all_params, LEARNING_RATE)

    iter_train = theano.function(
        [X_batch, Y_batch], accuracy48, updates=updates,
    )

    iter_valid = theano.function(
        [X_batch, Y_batch], accuracy48,
        givens={
            trans: data['trans'],
            transmap: data['transmap']
        }
    )

    return {"train": iter_train, "valid": iter_valid}

def main():
    data = {}
    trans, transmap = load_transition();
    data['trans'] = theano.shared(trans, borrow=True)
    data['transmap'] = theano.shared(transmap, borrow=True)
    print("Building model and compile theano...")
    output_layer = build_model(input_dim = 350, output_dim = 1943)
    if len(sys.argv) == 2:
        print("loading model {}".format(sys.argv[1]))
        fin = open("../model/{}".format(sys.argv[1]))
        import pickle
        lasagne.layers.set_all_param_values(output_layer, pickle.load(fin))
    iter_funcs = create_iter_functions(data, output_layer)
    print("Training")
    trainin = open("train.dat")
    validin = open("valid.dat")
    epoch = 0
    while True:
        accu = 0
        cnt = 0
        now = time.time()
        while True:
            X, Y, OK, trainin = load_data(trainin, "train.dat")
            if not OK:
                break
            accu += iter_funcs['train'](X, Y)
            cnt += 1
        accu = accu / cnt
        print("Epoch {} took {:.3f}s\t{:.2f}%(1943)".format(epoch+1, time.time() - now, accu * 100))

        accu = 0
        cnt = 0
        now = time.time()
        while True:
            X, Y, OK, validin = load_data(validin, "valid.dat")
            if not OK:
                break
            accu += iter_funcs['valid'](X, Y)
            cnt += 1
        accu = accu / cnt
        print("Valid {} took {:.3f}s\t{:.2f}%(39)".format(epoch+1, time.time() - now, accu * 100))
        print()

        epoch += 1
        import pickle
        fout = open("../model/tmp/{}x{:.2f}".format(epoch, accu * 100), "w")
        pickle.dump(lasagne.layers.get_all_param_values(output_layer), fout)
        fout.close()

if __name__ == '__main__':
  main()


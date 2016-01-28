"""
File: settings.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/sunprinceS
Description: setting of DNN
"""

import lasagne
import numpy as np
import collections
BATCH_SIZE = 250
MOMENTUM = 0.9
NUM_HIDDEN_UNITS = [256,64,32] #laptop(81)
# NUM_HIDDEN_UNITS = [2048, 2048,256,128] #laptop(81)
# NUM_HIDDEN_UNITS = [2000,1000,256,64] #restaurant(category : 12)
LEARNING_RATE = 0.0001
INPUT_DIM = 4000
OUTPUT_DIM = 3
NUM_EPOCHS=100

def build_model(inputVar,input_dim=INPUT_DIM, output_dim=OUTPUT_DIM,
        batch_size=BATCH_SIZE, num_hidden_units=NUM_HIDDEN_UNITS):
    l_in = lasagne.layers.InputLayer(shape=(BATCH_SIZE,input_dim),input_var=inputVar)
    l_dp0 = lasagne.layers.DropoutLayer(l_in, p=0.9)
    l_hidden1 = lasagne.layers.DenseLayer(
            l_dp0,
            num_units=num_hidden_units[0],
            nonlinearity=lasagne.nonlinearities.rectify,
            )
    l_dp1 = lasagne.layers.DropoutLayer(l_hidden1, p=0.9)
    l_hidden2 = lasagne.layers.DenseLayer(
            l_dp1,
            num_units=num_hidden_units[1],
            nonlinearity=lasagne.nonlinearities.rectify
            )
    l_dp2 = lasagne.layers.DropoutLayer(l_hidden2, p=0.9)
    l_hidden3 = lasagne.layers.DenseLayer(
            l_dp2,
            num_units=num_hidden_units[2],
            nonlinearity=lasagne.nonlinearities.rectify
            )
    l_dp3 = lasagne.layers.DropoutLayer(l_hidden3, p=0.9)
    l_out = lasagne.layers.DenseLayer(
            l_dp3,
            num_units=output_dim,
            nonlinearity=lasagne.nonlinearities.softmax
            )
    return l_out



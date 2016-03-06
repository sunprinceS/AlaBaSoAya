##################################################
#   FileName:    [ BinaryTreeLSTM.py ]           #
#   PackageName: [ TreeLSTM-MemNN ]              #
#   Synopsis:    [ A Binary Tree-LSTM with       #
#                  input at the leaf nodes. ]    #
#   Authors:     [ Juei-Yang Hsu, Wei Fang ]     #
##################################################


import TreeLSTM
import lasagne
import theano
import theano.tensor as T
from lasagne.nonlinearities import softmax
from lasagne.layers import InputLayer, DenseLayer, get_output
from lasagne.updates import adagrad
import numpy as np

class BinaryTreeLSTM(TreeLSTM):

    def __init__(self, config):
        TreeLSTM.__init__(self, config)
        self.gate_output = config['gate_output']
        if self.gate_output is None:
            self.gate_output = True

        self.criterion = config['criterion']

    def new_leaf_module(self, input):
        l_in = InputLayer((None, self.in_dim))
        linear = DenseLayer(l_in, num_units=self.mem_dim, nonlinearity=None)
        c = get_output(linear, input)
















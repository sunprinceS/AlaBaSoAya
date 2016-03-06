#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: TreeLSTM.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/yourname
Description: Parent class for dep. and cons. tree
"""
from __future__ import print_function

class TreeLSTM(object):
    def __init__(self,conf):
        self.in_dim = conf['in_dim']
        self.mem_dim = conf['mem_dim']
        self.mem_zeros = conf['mem_zeros']
        self.leaf_modules = None
        self.composer_modules = None
        self.output_modules = None
        self.train = False

    def training():
        self.train = True

    def evaluating():
        self.train = False
    
    def allocate_module(tree,module):
        if module == 'leaf_module':
            if self.leaf_modules is None:
                tree.leaf_module = self.new_leaf_module(self)
            else:
                n_free = len(self.leaf_modules)
                tree.leaf_module = self.leaf_modules[n_free]
                self.leaf_modules[n_free] = None

            if self.train :
                tree.leaf_module.training()
            else:
                tree.leaf_module.evaluating()

        elif module == 'composer':
            if self.composer_modules is None:
                tree.composer = self.new_composer(self)
            else:
                n_free = len(self.composer_modules)
                tree.composer = self.composer_modules[n_free]
                self.composer_modules[n_free] = None

            if self.train:
                tree.composer.training()
            else:
                tree.composer.evaluating()
        else:
                print("Unknowned moduleName {}".format(module),file=sys.stderr)
        ##output module :p
    # Virtual Function
    def new_composer():
        pass
    def new_leaf_module():
        pass
    def new_output_module():
        pass


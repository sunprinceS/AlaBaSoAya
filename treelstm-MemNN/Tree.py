#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: tree.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/yourname
Description: Basic structure for TreeLSTM
"""
def dfs_preorder(tree,node_list):
    node_list.extend(tree)
    for child in children_list:
        dfs_preorder(child,node_list)
        

class Tree(object):
    def __init__(self):
        self.parent = None
        self.n_children = 0
        self.size = None
        self.children_list = []

    def add_child(child):
        #chlid must be a tree
        child.parent = self
        self.n_children += 1
        self.children_list.extend(child)

    def get_size():
        if self.size is None:
            #haven't count the size yet
            self.size=1 #itself
            for child in children_list:
                self.size += child.get_size()
        else:
            return self.size

    def get_depth(): #it means "height"(leaf's depth is 0)
        depth = 0
        if self.n_children > 0:
            depth = max([children.get_depth() for child in children_list]) + 1

        return depth

    def dfs_preorder():
        nodes_list = []
        dfs_preorder(self,nodes_list)
        return nodes_list

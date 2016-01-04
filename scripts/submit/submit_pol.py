#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from xml.dom.minidom import parse
import xml.dom.minidom
import sys
# Open XML document using minidom parser
#Needs the file of predicted polarity
#<format>
#<sentence_id:polarity>

ans_dic={}
idList_file=open('misc_data/{}_te.id'.format(sys.argv[1]))
idList = idList_file.read().splitlines()
idList_file.close()
golden_asp_file=open('misc_data/{}')
def load_ans():
    # with open('output/{}_pol.out'.format(sys.argv[1],'r')) as ans_file:
        # tmp_ans = ans_file.read().splitlines()
        # for idx,ans in enumerate(tmp_ans):
            name='{}_{}'.format(idList[idx],)
            ans_dic[idList[idx]]= ans
    pass

submit_data="output/{}_pol.xml".format(sys.argv[1])
data_skeleton="Files/teGldAspTrg.xml"
DOMTree = xml.dom.minidom.parse(data_skeleton)
submit_DOMTree = DOMTree
reviews_root = submit_DOMTree.documentElement

# Get all the reviews
reviews = reviews_root.getElementsByTagName("Review")

for review in reviews:

   sentences_root = review.getElementsByTagName('sentences')[0]

   sentences = sentences_root.getElementsByTagName('sentence')
   for sentence in sentences:
       sentence_id = sentence.getAttribute("id")
       try:
           opinions_root = sentence.getElementsByTagName("Opinions")[0]
           opinions = opinions_root.getElementsByTagName("Opinion")
           for opinion in opinions:
               opinion.setAttribute("polarity",ans[sentence_id])
       except IndexError:
           continue

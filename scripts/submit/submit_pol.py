#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from xml.dom.minidom import parse
import xml.dom.minidom
import sys
# Open XML document using minidom parser

def load_ans():
    with open('misc_data/{}.pol.pred.{}'.format(sys.argv[1],sys.argv[2])) as ans_file:
        return ans_file.read().splitlines()

ans = load_ans()
submit_data="output/{}_pol.xml.{}".format(sys.argv[1],sys.argv[2])
data_skeleton="tmp_data/teGldAspTrg.xml.{}".format(sys.argv[2])
DOMTree = xml.dom.minidom.parse(data_skeleton)
submit_DOMTree = DOMTree
reviews_root = submit_DOMTree.documentElement

# Get all the reviews
reviews = reviews_root.getElementsByTagName("Review")
ans_idx = 0
for review in reviews:

   sentences_root = review.getElementsByTagName('sentences')[0]

   sentences = sentences_root.getElementsByTagName('sentence')
   for sentence in sentences:
       sentence_id = sentence.getAttribute("id")
       opinions_root = sentence.getElementsByTagName("Opinions")[0]
       opinions = opinions_root.getElementsByTagName("Opinion")
       for opinion in opinions:
           opinion.setAttribute("polarity",ans[ans_idx])
           ans_idx+=1
with open(submit_data,'w') as final_ans:
    final_ans.write(submit_DOMTree.toxml())

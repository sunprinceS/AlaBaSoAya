#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xml.dom.minidom import parse
import xml.dom.minidom
import sys
# Open XML document using minidom parser
#Needs the file of predicted polarity
#<format>
#<sentence_id:polarity>

ans={}
def load_ans():
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
       opinions_root = sentence.getElementsByTagName("Opinions")[0]
       opinions = opinions_root.getElementsByTagName("Opinion")
       for i,opinion in enumerate(opinions):
           opinion.setAttribute("polarity",ans[sentence_id])

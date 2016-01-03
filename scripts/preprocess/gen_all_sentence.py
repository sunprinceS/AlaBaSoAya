#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import xml.dom.minidom
import sys
from xml.dom.minidom import parse
# Open XML document using minidom parser

##Info Needed##
data_file="origin_data/subtask1/{}_train.xml".format(sys.argv[1])
data = open(data_file,'r')
all_sent=open("tmp_data/{}.all".format(sys.argv[1]),'w')

DOMTree = xml.dom.minidom.parse(data)
reviews_root = DOMTree.documentElement
reviews = reviews_root.getElementsByTagName("Review")

for review in reviews:

   sentences_root = review.getElementsByTagName('sentences')[0]
   sentences = sentences_root.getElementsByTagName('sentence')

   for sentence in sentences:
        text = str(sentence.getElementsByTagName("text")[0].childNodes[0].data)
        all_sent.write('{}\n'.format(text))

data.close()

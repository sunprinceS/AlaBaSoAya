#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import xml.dom.minidom
import sys
from xml.dom.minidom import parse
# Open XML document using minidom parser

##Info Needed##
data="origin_data/subtask1/{}.xml".format(sys.argv[1])

train_data_name="temp/{}.xml".format(sys.argv[1])
train_data = open(train_data_name,'w')


DOMTree = xml.dom.minidom.parse(data)
reviews_root = DOMTree.documentElement
reviews = reviews_root.getElementsByTagName("Review")

for review in reviews:

   sentences_root = review.getElementsByTagName('sentences')[0]
   sentences = sentences_root.getElementsByTagName('sentence')

   for sentence in sentences:
        sentence_id = sentence.getAttribute("id")
        text = str(sentence.getElementsByTagName("text")[0].childNodes[0].data)
        try:
            opinions_root = sentence.getElementsByTagName("Opinions")[0]
            opinions = opinions_root.getElementsByTagName("Opinion")
            for i,opinion in enumerate(opinions):
                category = opinion.getAttribute("category")
                polarity = opinion.getAttribute("polarity")
                train_data.write('{} {},{},{}\n'.format(sentence_id,text,category,polarity))
        except IndexError:
            print("{}\n".format(sentence_id))

train_data.close()
label.close()


#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xml.dom.minidom import parse
import xml.dom.minidom

# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse("../../data/trial.xml")
reviews_root = DOMTree.documentElement

# Get all the reviews
reviews = reviews_root.getElementsByTagName("Review")
train_data = open('data/train.dat','w')
label = open('data/train.lab','w')

for review in reviews:

   sentences_root = review.getElementsByTagName('sentences')[0]

   sentences = sentences_root.getElementsByTagName('sentence')
   for sentence in sentences:
        sentence_id = sentence.getAttribute("id")
        text = str(sentence.getElementsByTagName("text")[0].childNodes[0].data)

        opinions_root = sentence.getElementsByTagName("Opinions")[0]
        opinions = opinions_root.getElementsByTagName("Opinion")
        for i,opinion in enumerate(opinions):
            train_data.write('{},{}\n'.format(sentence_id,text))
            category = opinion.getAttribute("category")
            label.write('{},{}\n'.format(sentence_id,category))

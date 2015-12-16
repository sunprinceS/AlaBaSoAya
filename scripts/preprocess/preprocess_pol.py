#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xml.dom.minidom import parse
import xml.dom.minidom
import sys
# Open XML document using minidom parser

# data = "Files/" + sys.argv[1] + ".xml"
if(sys.argv[1] == 'tr'):
    data="Files/{}.xml".format(sys.argv[1])
    DOMTree = xml.dom.minidom.parse(data)
    reviews_root = DOMTree.documentElement

    # Get all the reviews
    reviews = reviews_root.getElementsByTagName("Review")
    train_data_name="Files/{}.pol.dat".format(sys.argv[1])
    golden_category_name="Files/{}.pol.goldenAsp".format(sys.argv[1])
    label_name="Files/{}.pol.label".format(sys.argv[1])
    train_data = open(train_data_name,'w')
    golden_category = open(golden_category_name,'w')
    label = open(label_name,'w')

    for review in reviews:

       sentences_root = review.getElementsByTagName('sentences')[0]

       sentences = sentences_root.getElementsByTagName('sentence')
       for sentence in sentences:
            sentence_id = sentence.getAttribute("id")
            text = str(sentence.getElementsByTagName("text")[0].childNodes[0].data)
            opinions_root = sentence.getElementsByTagName("Opinions")[0]
            opinions = opinions_root.getElementsByTagName("Opinion")
            for i,opinion in enumerate(opinions):
                train_data.write('{}\n'.format(text))
                category=opinion.getAttribute("category")
                golden_category.write('{}\n'.format(category))
                polarity= opinion.getAttribute("polarity")
                label.write('{},{}\n'.format(sentence_id,polarity))
else: #te
    data="Files/teCln.xml"
    DOMTree = xml.dom.minidom.parse(data)
    reviews_root = DOMTree.documentElement

    # Get all the reviews
    reviews = reviews_root.getElementsByTagName("Review")
    test_data_name="Files/{}.pol.dat".format(sys.argv[1])
    golden_category_name="Files/{}.pol.goldenAsp".format(sys.argv[1])
    train_data = open(train_data_name,'w')
    golden_category = open(golden_category_name,'w')

    for review in reviews:

       sentences_root = review.getElementsByTagName('sentences')[0]

       sentences = sentences_root.getElementsByTagName('sentence')
       for sentence in sentences:
            sentence_id = sentence.getAttribute("id")
            text = str(sentence.getElementsByTagName("text")[0].childNodes[0].data)
            opinions_root = sentence.getElementsByTagName("Opinions")[0]
            opinions = opinions_root.getElementsByTagName("Opinion")
            for i,opinion in enumerate(opinions):
                test_data.write('{}\n'.format(text))
                category=opinion.getAttribute("category")
                golden_category.write('{}\n'.format(category))

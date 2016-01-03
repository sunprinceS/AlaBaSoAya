#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import xml.dom.minidom
import sys
from xml.dom.minidom import parse
# Open XML document using minidom parser

if(sys.argv[2] == 'train'):

    ##Info Needed##
    data="tmp_data/tr.xml.{}".format(sys.argv[3])
    train_data_name="misc_data/{}_{}.asp.dat.{}".format(sys.argv[1],sys.argv[2],sys.argv[3])
    label_name="misc_data/{}_{}.asp.label.{}".format(sys.argv[1],sys.argv[2],sys.argv[3])
    train_data = open(train_data_name,'w')
    label = open(label_name,'w')


    DOMTree = xml.dom.minidom.parse(data)
    reviews_root = DOMTree.documentElement
    reviews = reviews_root.getElementsByTagName("Review")

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
                category = opinion.getAttribute("category")
                label.write('{},{}\n'.format(sentence_id,category))

    train_data.close()
    label.close()

elif sys.argv[2]=='te':

    ##Info needed##
    data="tmp_data/teCln.xml.{}".format(sys.argv[3])
    test_data_name="misc_data/{}_{}.asp.dat.{}".format(sys.argv[1],sys.argv[2],sys.argv[3])
    test_data = open(test_data_name,'w')
    idxMapping=open("misc_data/{}_te.id.{}".format(sys.argv[1],sys.argv[3]),'w')

    DOMTree = xml.dom.minidom.parse(data)
    reviews_root = DOMTree.documentElement
    reviews = reviews_root.getElementsByTagName("Review")

    for review in reviews:

       sentences_root = review.getElementsByTagName('sentences')[0]
       sentences = sentences_root.getElementsByTagName('sentence')

       for sentence in sentences:
            sentence_id = sentence.getAttribute("id")
            idxMapping.write('{}\n'.format(sentence_id))
            text = str(sentence.getElementsByTagName("text")[0].childNodes[0].data)
            test_data.write('{}\n'.format(text))

    test_data.close()
    idxMapping.close()
else:
    print('Unexpected input!',file=sys.stderr)
    sys.exit()

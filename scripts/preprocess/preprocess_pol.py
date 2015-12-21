#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from xml.dom.minidom import parse
import xml.dom.minidom
import sys
# Open XML document using minidom parser

if(sys.argv[2] == 'train'):
    data="tmp_data/tr.xml"

    DOMTree = xml.dom.minidom.parse(data)
    reviews_root = DOMTree.documentElement

    # Get all the reviews
    reviews = reviews_root.getElementsByTagName("Review")


    train_data_name="misc_data/{}_{}.pol.dat".format(sys.argv[1],sys.argv[2])
    golden_asp_name="misc_data/{}_{}.pol.goldenAsp".format(sys.argv[1],sys.argv[2])
    label_name="misc_data/{}_{}.pol.label".format(sys.argv[1],sys.argv[2])

    train_data = open(train_data_name,'w')
    golden_asp = open(golden_asp_name,'w')
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
                golden_asp.write('{}\n'.format(category))
                polarity= opinion.getAttribute("polarity")
                label.write('{},{}\n'.format(sentence_id,polarity))
    train_data.close()
    golden_asp.close()
    label.close()
elif sys.argv[2]=='te':

    data="tmp_data/teGldAspTrg.xml"
    idxMapping=open("misc_data/{}_te.id".format(sys.argv[1]),'w')
    DOMTree = xml.dom.minidom.parse(data)
    reviews_root = DOMTree.documentElement

    # Get all the reviews
    reviews = reviews_root.getElementsByTagName("Review")
    test_data_name="misc_data/{}_{}.pol.dat".format(sys.argv[1],sys.argv[2])
    golden_asp_name="misc_data/{}_{}.pol.goldenAsp".format(sys.argv[1],sys.argv[2])

    test_data = open(test_data_name,'w')
    golden_asp = open(golden_asp_name,'w')

    for review in reviews:

       sentences_root = review.getElementsByTagName('sentences')[0]

       sentences = sentences_root.getElementsByTagName('sentence')
       for sentence in sentences:
            sentence_id = sentence.getAttribute("id")
            text = str(sentence.getElementsByTagName("text")[0].childNodes[0].data)
            opinions_root = sentence.getElementsByTagName("Opinions")[0]
            opinions = opinions_root.getElementsByTagName("Opinion")

            for opinion in opinions:
                idxMapping.write('{}\n'.format(sentence_id))
                test_data.write('{}\n'.format(text))
                category=opinion.getAttribute("category")
                golden_asp.write('{}\n'.format(category))
    test_data.close()
    golden_asp.close()
else:
    print("Unexpected input",file=sys.stderr)
    sys.exit()

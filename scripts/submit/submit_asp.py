#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from xml.dom.minidom import parse
import xml.dom.minidom
import sys

#Needs the file of predicted categories
#<format>
#<sentence_id:category1 category2 ...>


##Info Needed##
ans_dic={}
idList_file=open('misc_data/{}_te.id.{}'.format(sys.argv[1],sys.argv[2]))
idList = idList_file.read().splitlines()
idList_file.close()
submit_data="output/{}_asp.xml.{}".format(sys.argv[1],sys.argv[2])
data_skeleton="tmp_data/teCln.xml.{}".format(sys.argv[2])

def load_ans():
    with open('output/{}_asp.out.{}'.format(sys.argv[1],sys.argv[2]),'r') as ans_file:
        tmp_ans = ans_file.read().splitlines()
        for idx,a in enumerate(tmp_ans):
            one_ans = a.split(' ')
            ans_dic[idList[idx]] = one_ans

##Root##
DOMTree = xml.dom.minidom.parse(data_skeleton)
submit_DOMTree = DOMTree
reviews_root = submit_DOMTree.documentElement
reviews = reviews_root.getElementsByTagName("Review")

load_ans()

for review in reviews:

    sentences_root = review.getElementsByTagName('sentences')[0]
    sentences = sentences_root.getElementsByTagName('sentence')

    for sentence in sentences:
        sentence_id = sentence.getAttribute("id")
        opinions_root = submit_DOMTree.createElement("Opinions")

        ##Set xml attribute
        for category in ans_dic[sentence_id]:
            opinion = submit_DOMTree.createElement("Opinion")
            opinion.setAttribute("target","NULL")
            opinion.setAttribute("category",category)
            opinion.setAttribute("polarity","positive")
            opinion.setAttribute("from","0")
            opinion.setAttribute("to","0")
            opinions_root.appendChild(opinion)

        sentence.appendChild(opinions_root)

##Write Ans##
with open(submit_data,'w') as final_ans:
    final_ans.write(submit_DOMTree.toxml())

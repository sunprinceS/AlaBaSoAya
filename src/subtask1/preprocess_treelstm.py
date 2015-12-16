#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xml.dom.minidom import parse
import xml.dom.minidom
import sys
# Open XML document using minidom parser

data = "../../../data/subtask1/" + sys.argv[1] + ".xml"
# data = "../../../data/subtask1/laptop_trial.xml"
DOMTree = xml.dom.minidom.parse(data)
reviews_root = DOMTree.documentElement

# Get all the reviews
reviews = reviews_root.getElementsByTagName("Review")
# train_data_name="data/"+sys.argv[1]+".dat"
label_name="dataForTreeLSTM/"+sys.argv[1]+".sent_lab"
sent_name='dataForTreeLSTM/'+sys.argv[1]+'.sent_dat'
# train_data = open(train_data_name,'w')
label = open(label_name,'w')
sent_data = open(sent_name,'w')

for review in reviews:

   sentences_root = review.getElementsByTagName('sentences')[0]

   sentences = sentences_root.getElementsByTagName('sentence')
   for sentence in sentences:
        sentence_id = sentence.getAttribute("id")
        try:
            text = str(sentence.getElementsByTagName("text")[0].childNodes[0].data)
        except UnicodeEncodeError:
            continue
        try:
            opinions_root = sentence.getElementsByTagName("Opinions")[0]
            opinions = opinions_root.getElementsByTagName("Opinion")
            same = True
            neg_cnt=0
            pos_cnt=0
            neu_cnt=0
            for i,opinion in enumerate(opinions):
                polarity= opinion.getAttribute("polarity")
                if(polarity == "negative"):
                    neg_cnt += 1
                elif(polarity == "positive"):
                    pos_cnt += 1
                else: #neutral
                    neu_cnt += 1

            if(pos_cnt > 1 ): #possible postive
                if(neg_cnt == 0):
                    sent_data.write('{}\n'.format(text))
                    label.write('{}\n'.format(5))
                else :
                    break
            elif(pos_cnt == 1):
                if(neg_cnt == 0):
                    sent_data.write('{}\n'.format(text))
                    label.write('{}\n'.format(4))
                else :
                    break

            if(neg_cnt > 1): #possible negative
                if(pos_cnt == 0):
                    sent_data.write('{}\n'.format(text))
                    label.write('{}\n'.format(1))
                else:
                    break
            elif(neg_cnt == 1):
                if(pos_cnt == 0):
                    sent_data.write('{}\n'.format(text))
                    label.write('{}\n'.format(2))
                else:
                    break

            if(neu_cnt > 0 and pos_cnt == 0 and neg_cnt == 0):
                sent_data.write('{}\n'.format(text))
                label.write('{}\n'.format(3))
        except IndexError:
            continue

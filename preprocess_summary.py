#!/usr/bin/env python
# -*- coding: utf-8 -*-

from xml.dom.minidom import parse
import xml.dom.minidom

# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse("../data/trial.xml")
reviews_root = DOMTree.documentElement

# Get all the reviews
reviews = reviews_root.getElementsByTagName("Review")

# Print detail of each movie.
for review in reviews:

   # if review.hasAttribute("rid"):
      # print ("***Sentence ID***: {}\n".format(movie.getAttribute("rid")))
   print("*******\n")
   sentences_root = review.getElementsByTagName('sentences')[0]

   sentences = sentences_root.getElementsByTagName('sentence')
   for sentence in sentences:
        print("Sentence ID:{}\n".format(sentence.getAttribute("id")))
        text = str(sentence.getElementsByTagName("text")[0].childNodes[0].data)
        print("{}\n".format(text))
        opinions_root = sentence.getElementsByTagName("Opinions")[0]
        opinions = opinions_root.getElementsByTagName("Opinion")
        for i,opinion in enumerate(opinions):
            target = opinion.getAttribute("target")
            category = opinion.getAttribute("category")
            polarity = opinion.getAttribute("polarity")
            OTE_from = int(opinion.getAttribute("from"))
            OTE_to = int(opinion.getAttribute("to"))
            OTE = text[OTE_from:OTE_to]
            if(OTE==""):
                OTE="NULL"
            print("Opinion {} Summary\n".format(i))
            print("===============\n")
            print("\t target : {}\n".format(target))
            print("\t category: {}\n".format(category))
            print("\t polarity : {}\n".format(polarity))
            print("\t OTE : {}\n".format(OTE))
            print("===============\n")
   print("*******\n")

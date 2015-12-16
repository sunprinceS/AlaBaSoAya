#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: collectData.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/yourname
Description: collect data for lasagne
"""
import sys

category_map={}
category_map['positive'] = 0
category_map['negative'] = 1
category_map['neutral'] = 2
lab_fn = "../data/"+sys.argv[1]+".sent"
embed_fn = "../../../../embed_concat/"+sys.argv[1]+".embed4000"
output_fn = "DATA/"+sys.argv[1]+".in"



if __name__ == '__main__':
    # loadMap("../data/"+sys.argv[1]+".category")
    output = open(output_fn,'w')
    output_lab = open("DATA/{}.lab_sent".format(sys.argv[1]),'w')
    labs = open(lab_fn,'r').read().splitlines()
    embeds = open(embed_fn,'r').read().splitlines()
    assert len(labs) == len(embeds)

    for lab_tmp,embed in zip(labs,embeds):
        lab=lab_tmp.split(',')[1]
        output.write("{}\n".format(embed))
        output_lab.write("{}\n".format(category_map[lab]))

    output.close()
    output_lab.close()
    # labs.close()
    # embeds.close()

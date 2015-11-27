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
lab_fn = "../data/"+sys.argv[1]+".lab"
embed_fn = "../../../../embed_concat/"+sys.argv[1]+".embed4000"
output_fn = "DATA/"+sys.argv[1]+".in"

def loadMap(categoryFile):
    with open(categoryFile,'r') as category:
        lines = category.read().splitlines()
        for idx,line in enumerate(lines):
            category_map[line] = idx


if __name__ == '__main__':
    loadMap("../data/"+sys.argv[1]+".category")
    output = open(output_fn,'w')
    labs = open(lab_fn,'r').read().splitlines()
    embeds = open(embed_fn,'r').read().splitlines()
    assert len(labs) == len(embeds)

    for lab_tmp,embed in zip(labs,embeds):
        lab=lab_tmp.split(',')[1]
        output.write("{} {}\n".format(category_map[lab],embed))

    output.close()
    # labs.close()
    # embeds.close()

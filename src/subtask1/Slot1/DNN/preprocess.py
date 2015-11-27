#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
File: preprocess.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/yourname
Description: concat 4 1000-dimvectors to 4000-dim vector
"""

import sys

i_fn = "../../../../embed/"+sys.argv[1]+".embed"
o_fn = "../../../../embed_concat/"+sys.argv[1]+".embed4000"

o_data = open(o_fn,'w')

with open(i_fn) as i_data:
    lines = i_data.read().splitlines()
    for line in lines:
        if(line!=""):
            o_data.write("{} ".format(line))
        else:
            o_data.write("\n")
o_data.close()

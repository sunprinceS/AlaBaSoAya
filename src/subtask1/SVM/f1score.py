#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from sklearn.metrics import f1_score

pred = open('prediction_{}'.format(sys.argv[1])).read().splitlines()
# print(len(pred))
golden = open('golden_{}'.format(sys.argv[1])).read().splitlines()
# print(len(golden))

print(f1_score(pred,golden,average='micro'))

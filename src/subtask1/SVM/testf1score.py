#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.metrics import f1_score
y_true=['a','b','c','a','b','c']
y_pred=['a','c','b','a','a','b']

print(f1_score(y_true,y_pred,average='micro'))

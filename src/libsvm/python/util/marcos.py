"""
File: marcos.py
Author: sunprinceS (TonyHsu)
Email: sunprince12014@gmail.com
Github: https://github.com/sunprinceS
Description: MARCOS
"""

from sklearn.pipeline import Pipeline , FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer

CAT_DIR='misc_data/categoryMap'
MISC_DIR='misc_data'
TRANSFORM_MODEL_DIR='src/libsvm/python/transformModel'
SVM_MODEL_DIR='src/libsvm/python/SVMmodel'


PREPROCESS_PIPE = Pipeline([('bow',CountVectorizer()),
                             ('tfidf',TfidfTransformer())])

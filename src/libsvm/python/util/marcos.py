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
NN_DIR='src/DNN'
TRANSFORM_MODEL_DIR='src/libsvm/python/transformModel'
MAX_ENT_MODEL_DIR='src/MaxEnt/MaxEntModel'
SVM_MODEL_DIR='src/libsvm/python/SVMmodel'
SVM_SENT_MODEL_DIR='src/libsvm/python/SVMSentModel'
MODEL='2layers_200neurons_0.5dropout_reluactivation'


PREPROCESS_PIPE = Pipeline([('bow',CountVectorizer()),
                             ('tfidf',TfidfTransformer())])

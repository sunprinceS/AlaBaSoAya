from __future__ import print_function
import numpy as np
import scipy as sp
import joblib as jl
import re
import sys

from sklearn.feature_extraction.text import CountVectorizer , TfidfTransformer
from sklearn.decomposition import PCA
from sklearn.decomposition import LatentDirichletAllocation as LDA
# from sklearn.externals import joblib

__all__ = ['io','transform','marcos']

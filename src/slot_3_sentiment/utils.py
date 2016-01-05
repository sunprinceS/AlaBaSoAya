#########################################################
#   FileName:       [ utils.py ]                        #
#   PackageName:    [ AlaBaSoAya ]                      #
#   Synopsis:       [ Define utility functions ]        #
#   Authors:        [ Wei Fang, SunprinceS ]            #
#########################################################

import numpy as np
import scipy.io as sio
import joblib
import sys
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils
from itertools import zip_longest

#########################
#     I/O functions     #
#########################
def LoadAspectMap(domain):
    # Description: Load mapping of aspect string to integer
    # Output: dictionary mapping aspect strings to its corresponding index
    asp_map = {}
    with open('misc_data/categoryMap/'+domain+'.category', 'r') as map_file:
        idx = 0
        for asp in map_file:
            asp = asp.strip()
            asp_map[asp] = idx
            idx += 1
    return asp_map

def LoadSentenceFeatures(domain, dataset, cross_val_idx):
    # Description: Load sentence features (features are from TreeLSTM)
    # Output: numpy ndarray with dimension (nb_sentences, sent_vec_dim)
    #   Note: if dataset=='train', returns 2 arrays (train, dev), or else returns only 1
    assert (dataset == 'train' or dataset == 'te' or dataset == 'test')
    feats = []
    with open('misc_data/'+domain+'_'+dataset+'.pol.feat.'+str(cross_val_idx), 'r') as feats_file:
        for feat in feats_file:
            feat = feat.strip().split()
            feat = [ float(x) for x in feat ]
            feats.append(feat)
    if dataset == 'te' or dataset == 'test':
        return np.asarray(feats, 'float32')

    train_feats = np.asarray(feats[:int((4/5)*len(feats))], 'float32')
    dev_feats = np.asarray(feats[int((4/5)*len(feats)):], 'float32')
    return train_feats, dev_feats

def LoadAspects(domain, dataset, cross_val_idx, asp_map):
    # Description: Load golden aspects
    # Output: list of aspect strings
    #   ex. [ 'FOOD#QUALITY','DRINKS#PRICES', ... ]
    #   Note: if dataset=='train', returns 2 arrays (train, dev), or else returns only 1
    assert (dataset == 'train' or dataset == 'te' or dataset == 'test')
    aspects = []
    with open('misc_data/'+domain+'_'+dataset+'.pol.goldenAsp.'+str(cross_val_idx), 'r') as asp_file:
        for asp in asp_file:
            asp = asp.strip()
            asp_idx = asp_map[asp]
            aspects.append(asp_idx)
    if dataset == 'te' or dataset == 'test':
        return aspects

    train_asp = aspects[:int((4/5)*len(aspects))]
    dev_asp = aspects[int((4/5)*len(aspects)):]
    return train_asp, dev_asp

def LoadLabels(domain, dataset, cross_val_idx):
    # Description: loads sentiment labels
    #   Note: if dataset=='train', returns 2 arrays (train, dev), or else returns only 1
    assert (dataset == 'train' or dataset == 'te' or dataset == 'test')
    labels = []
    with open('misc_data/'+domain+'_'+dataset+'.pol.label.'+str(cross_val_idx), 'r') as lab_file:
        for lab in lab_file:
            lab = lab.strip().split(',')[1]
            labels.append(lab)
    if dataset == 'te' or dataset == 'test':
        return labels

    train_labs = labels[:int((4/5)*len(labels))]
    dev_labs = labels[int((4/5)*len(labels)):]
    return train_labs, dev_labs

def SavePredictions(filepath, predictions, length):
    # Save predictions to filepath
    with open(filepath, 'w') as f:
        for i in range(length):
            pred = predictions[i]
            f.write('%s\n' % pred)

############################
#       Get Features       #
############################

def GetAspectFeatures(aspects, asp_encoder):
    # Description: converts string objects to 1-of-N vectors
    # Input:
    #   aspects: a list of aspect strings
    #   asp_encoder: a scikit-learn LabelEncoder object
    # Output:
    #   numpy ndarray of shape (batch_size, nb_classes)
    return asp_encoder.transform(np.asarray(aspects).reshape(-1,1)).toarray()

def GetLabelEncoding(labels, lab_encoder):
    # Description: converts string objects to class labels
    # Input:
    #   labels: a list of label strings
    #   lab_encoder: a scikit-learn LabelEncoder object
    # Output:
    #   numpy ndarray of shape (batch_size, nb_polarities)
    y = lab_encoder.transform(labels)
    nb_polarities = lab_encoder.classes_.shape[0]
    Y = np_utils.to_categorical(y, nb_polarities)
    return Y

############################
#  Other Helper Functions  #
############################

def MakeBatches(iterable, n, fillvalue=None):
    # Description: Split data into batches
    # Output: list of tuples, where a tuple contains the data for a batch
    #    ex. [ ('positive','negative'),('neutral','negative'),...) ] for labels with batch_size=2
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

def GetLabelEncoder(domain, cross_val_idx):
    # Description: Create a LabelEncoder object for encoding labels to 1-of-N vectors
    # Output: a scikit-learn LabelEncoder object
    with open('misc_data/'+domain+'_train.pol.label.'+str(cross_val_idx), 'r') as lab_file:
        lab_file = lab_file.read().splitlines()
        for i in range(len(lab_file)):
            lab_file[i] = lab_file[i].split(',')[1]
        labelencoder = LabelEncoder()
        labelencoder.fit(lab_file)
        return labelencoder

def GetAspectEncoder(asp_map):
    # Description: Create a OneHotEncoder object for encoding aspect strings to 1-of-N vectors
    # Output: a scikit-learn OneHotEncoder object
    #   Note: I don't know why I didn't use LabelEncoder ORZ
    asp_enc = np.asarray(list(range(len(asp_map)))).reshape(-1,1)
    enc = OneHotEncoder()
    enc.fit(asp_enc)
    return enc

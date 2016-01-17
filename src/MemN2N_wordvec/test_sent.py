######################################################################################
#   FileName:       [ test_sent.py ]                                                 #
#   PackageName:    [ AlaBasoAya ]                                                   #
#   Synopsis:       [ Test MemNN_wordvec for ABSA sentiment classification ]         #
#   Authors:        [ Wei Fang, SunprinceS ]                                         #
######################################################################################

import numpy as np
import sys
import argparse
import joblib
import time
import signal
import random

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.utils import generic_utils
from keras.optimizers import Adagrad

from utils import LoadAspectMap, LoadSentenceFeatures, LoadAspects, LoadLabels, SavePredictions, GetAspectFeatures, GetLabelEncoding, GetLabelEncoder, GetAspectEncoder, MakeBatches, GetAspectFeatures, LoadSentences, LoadGloVe, GetSentenceTensor
from settings import CreateGraph, transpose

def main():
    start_time = time.time()

    # argument parser
    parser = argparse.ArgumentParser(prog='test_sent.py',
            description='Test MemNN-wordvec model for ABSA sentiment classification')
    parser.add_argument('--mlp-hidden-units', type=int, default=256, metavar='<mlp-hidden-units>')
    parser.add_argument('--mlp-hidden-layers', type=int, default=2, metavar='<mlp-hidden-layers>')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='<dropout-rate>')
    parser.add_argument('--mlp-activation', type=str, default='relu', metavar='<activation-function>')
    parser.add_argument('--batch-size', type=int, default=32, metavar='<batch-size>')
    parser.add_argument('--learning-rate', type=float, default=0.001, metavar='<learning-rate>')
    parser.add_argument('--aspects', type=int, required=True, metavar='<number of aspects>')
    parser.add_argument('--domain', type=str, required=True, choices=['rest','lapt'], metavar='<domain>')
    parser.add_argument('--cross-val-index', type=int, required=True, choices=range(0,10), metavar='<cross-validation-index>')
    parser.add_argument('--weights', type=str, required=True, metavar='<weights-path>')
    parser.add_argument('--output', type=str, required=True, metavar='<prediction-path>')
    args = parser.parse_args()
    args = parser.parse_args()

    word_vec_dim = 300
    aspect_dim = args.aspects
    polarity_num = 3
    emb_dim = 75
    emb_size = 100
    img_dim = word_vec_dim
    hops = 2

    ######################
    # Model Descriptions #
    ######################
    print('Generating and compiling model...')
    model = CreateGraph(emb_dim, hops, 'relu', args.mlp_hidden_units, args.mlp_hidden_layers, word_vec_dim, aspect_dim, img_dim, emb_size, polarity_num)

    # loss and optimizer
    adagrad = Adagrad(lr=args.learning_rate)
    model.compile(loss={'output':'categorical_crossentropy'}, optimizer=adagrad)
    model.load_weights(args.weights)
    print('Compilation finished.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #      Load Data     #
    ######################
    print('Loading data...')

    # aspect mapping
    asp_map = LoadAspectMap(args.domain)
    # sentences
    te_sents = LoadSentences(args.domain, 'te', args.cross_val_index)
    # aspects
    te_asps = LoadAspects(args.domain, 'te', args.cross_val_index, asp_map)
    print('Finished loading data.')
    print('Time: %f s' % (time.time()-start_time))

    #####################
    #       GloVe       #
    #####################
    print('Loading GloVe vectors...')

    word_embedding, word_map = LoadGloVe()
    print('GloVe vectors loaded')
    print('Time: %f s' % (time.time()-start_time))


    #####################
    #      Encoders     #
    #####################
    asp_encoder = GetAspectEncoder(asp_map)
    lab_encoder = joblib.load('models/'+args.domain+'_labelencoder_'+str(args.cross_val_index)+'.pkl')

    ######################
    #    Make Batches    #
    ######################
    print('Making batches...')

    # validation batches
    te_sent_batches = [ b for b in MakeBatches(te_sents, args.batch_size, fillvalue=te_sents[-1]) ]
    te_asp_batches = [ b for b in MakeBatches(te_asps, args.batch_size, fillvalue=te_asps[-1]) ]

    print('Finished making batches.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #      Testing       #
    ######################

    # start testing
    print('Testing started...')
    pbar = generic_utils.Progbar(len(te_sent_batches)*args.batch_size)

    predictions = []
    # testing feedforward
    for i in range(len(te_sent_batches)):
        X_sent_batch = GetSentenceTensor(te_sent_batches[i], word_embedding, word_map)
        X_asp_batch = GetAspectFeatures(te_asp_batches[i], asp_encoder)
        pred = model.predict_on_batch({'sentence': X_sent_batch, 'aspect': X_asp_batch})
        pred = pred[0]
        pred = np.argmax(pred, axis=1)
        pol = lab_encoder.inverse_transform(pred).tolist()
        predictions.extend(pol)
        pbar.add(args.batch_size)
    SavePredictions(args.output, predictions, len(te_sents))

    print('Testing finished.')
    print('Time: %f s' % (time.time()-start_time))

if __name__ == "__main__":
    main()

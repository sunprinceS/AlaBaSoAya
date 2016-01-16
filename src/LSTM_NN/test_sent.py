######################################################################################
#   FileName:       [ train_sent.py ]                                                #
#   PackageName:    [ AlaBasoAya ]                                                   #
#   Synopsis:       [ Train MLP for ABSA sentiment classification ]                  #
#   Authors:        [ Wei Fang, SunprinceS ]                                         #
######################################################################################

import numpy as np
import sys
import argparse
import joblib
import time
import signal
import random

from keras.models import Sequential, model_from_json
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.utils import generic_utils
from keras.optimizers import Adagrad

from utils import LoadAspectMap, LoadSentenceFeatures, LoadAspects, LoadLabels, SavePredictions, GetAspectFeatures, GetLabelEncoding, GetLabelEncoder, GetAspectEncoder, MakeBatches

def main():
    start_time = time.time()

    # argument parser
    parser = argparse.ArgumentParser(prog='test_sent.py',
            description='Test MLP model for ABSA sentiment classification')
    parser.add_argument('--aspects', type=int, required=True, metavar='<number of aspects>')
    parser.add_argument('--domain', type=str, required=True, choices=['rest','lapt'], metavar='<domain>')
    parser.add_argument('--cross-val-index', type=int, required=True, choices=range(0,10), metavar='<cross-validation-index>')
    parser.add_argument('--model', type=str, required=True, metavar='<model-path>')
    parser.add_argument('--weights', type=str, required=True, metavar='<weights-path>')
    parser.add_argument('--output', type=str, required=True, metavar='<prediction-path>')
    args = parser.parse_args()

    sent_vec_dim = 300
    batch_size = 128
    aspect_dim = args.aspects
    polarity_num = 3

    #######################
    #      Load Model     #
    #######################
    print('Loading model and weights...')
    model = model_from_json(open(args.model,'r').read())
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    model.load_weights(args.weights)
    print('Model and weights loaded.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #      Load Data     #
    ######################
    print('Loading data...')

    # aspect mapping
    asp_map = LoadAspectMap(args.domain)
    # features
    te_feats = LoadSentenceFeatures(args.domain, 'te', args.cross_val_index)
    # aspects
    te_asps = LoadAspects(args.domain, 'te', args.cross_val_index, asp_map)
    print('Finished loading data.')
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

    # te batches
    te_feat_batches = [ b for b in MakeBatches(te_feats, batch_size, fillvalue=te_feats[-1]) ]
    te_asp_batches = [ b for b in MakeBatches(te_asps, batch_size, fillvalue=te_asps[-1]) ]

    print('Finished making batches.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #      Training      #
    ######################
    dev_accs = []
    max_acc = -1
    max_acc_epoch = -1

    # define interrupt handler
    def PrintDevAcc():
        print('Max validation accuracy epoch: %i' % max_acc_epoch)
        print(dev_accs)

    def InterruptHandler(sig, frame):
        print(str(sig))
        PrintDevAcc()
        sys.exit(-1)

    signal.signal(signal.SIGINT, InterruptHandler)
    signal.signal(signal.SIGTERM, InterruptHandler)

    # start testing
    print('Testing started...')
    pbar = generic_utils.Progbar(len(te_feat_batches)*batch_size)

    predictions = []
    # testing feedforward
    for i in range(len(te_feat_batches)):
        X_feature_batch = np.asarray(te_feat_batches[i], 'float32')
        X_asp_batch = GetAspectFeatures(te_asp_batches[i], asp_encoder)
        pred = model.predict_classes([X_feature_batch, X_asp_batch], batch_size, verbose=0)
        pol = lab_encoder.inverse_transform(pred).tolist()
        predictions.extend(pol)
    SavePredictions(args.output, predictions, len(te_feats))

    print('Testing finished.')
    print('Time: %f s' % (time.time()-start_time))

if __name__ == "__main__":
    main()

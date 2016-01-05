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

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Merge, Dropout, Reshape
from keras.utils import generic_utils
from keras.optimizers import Adagrad

from utils import LoadAspectMap, LoadSentenceFeatures, LoadAspects, LoadLabels, SavePredictions, GetAspectFeatures, GetLabelEncoding, GetLabelEncoder, GetAspectEncoder, MakeBatches

def main():
    start_time = time.time()

    # argument parser
    parser = argparse.ArgumentParser(prog='train_sent.py',
            description='Train MLP model for ABSA sentiment classification')
    parser.add_argument('--mlp-hidden-units', type=int, default=256, metavar='<mlp-hidden-units>')
    parser.add_argument('--mlp-hidden-layers', type=int, default=3, metavar='<mlp-hidden-layers>')
    parser.add_argument('--dropout', type=float, default=0.3, metavar='<dropout-rate>')
    parser.add_argument('--mlp-activation', type=str, default='relu', metavar='<activation-function>')
    parser.add_argument('--num-epochs', type=int, default=100, metavar='<num-epochs>')
    parser.add_argument('--batch-size', type=int, default=128, metavar='<batch-size>')
    parser.add_argument('--learning-rate', type=float, default=0.0001, metavar='<learning-rate>')
    parser.add_argument('--aspects', type=int, required=True, metavar='<number of aspects>')
    parser.add_argument('--domain', type=str, required=True, choices=['rest','lapt'], metavar='<domain>')
    parser.add_argument('--cross-val-index', type=int, required=True, choices=range(0,10), metavar='<cross-validation-index>')
    args = parser.parse_args()

    sent_vec_dim = 300
    aspect_dim = args.aspects
    polarity_num = 3

    ######################
    # Model Descriptions #
    ######################
    print('Generating and compiling model...')
    # sentence input
    sent_model = Sequential()
    sent_model.add(Reshape(
        input_shape=(sent_vec_dim,), dims=(sent_vec_dim,)
        ))
    # aspect input
    aspect_model = Sequential()
    aspect_model.add(Reshape(
        input_shape=(aspect_dim,), dims=(aspect_dim,)
        ))
    # feedforward model (MLP)
    model = Sequential()
    model.add(Merge(
        [sent_model, aspect_model], mode='concat', concat_axis=1
        ))
    for i in range(args.mlp_hidden_layers):
        model.add(Dense(
            args.mlp_hidden_units, init='uniform'
            ))
        model.add(Activation(args.mlp_activation))
        model.add(Dropout(args.dropout))
    model.add(Dense(polarity_num))
    model.add(Activation('softmax'))

    # save model configuration
    json_string = model.to_json()
    model_filename = 'models/%s_mlp_units_%i_layers_%i_%s_lr%.1e_dropout%.1f.%i' % (args.domain, args.mlp_hidden_units, args.mlp_hidden_layers, args.mlp_activation, args.learning_rate, args.dropout, args.cross_val_index)
    open(model_filename + '.json', 'w').write(json_string)

    # loss and optimizer
    adagrad = Adagrad(lr=args.learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad)
    print('Compilation finished.')
    print('Time: %f s' % (time.time()-start_time))

    ######################
    #      Load Data     #
    ######################
    print('Loading data...')

    # aspect mapping
    asp_map = LoadAspectMap(args.domain)
    # features
    train_feats, dev_feats = LoadSentenceFeatures(args.domain, 'train', args.cross_val_index)
    # aspects
    train_asps, dev_asps = LoadAspects(args.domain, 'train', args.cross_val_index, asp_map)
    # labels
    train_labs, dev_labs = LoadLabels(args.domain, 'train', args.cross_val_index)
    print('Finished loading data.')
    print('Time: %f s' % (time.time()-start_time))

    #####################
    #      Encoders     #
    #####################
    asp_encoder = GetAspectEncoder(asp_map)
    lab_encoder = GetLabelEncoder(args.domain, args.cross_val_index)
    joblib.dump(lab_encoder,'models/'+args.domain+'_labelencoder_'+str(args.cross_val_index)+'.pkl')

    ######################
    #    Make Batches    #
    ######################
    print('Making batches...')

    # training batches
    train_feat_batches = [ b for b in MakeBatches(train_feats, args.batch_size, fillvalue=train_feats[-1]) ]
    train_asp_batches = [ b for b in MakeBatches(train_asps, args.batch_size, fillvalue=train_asps[-1]) ]
    train_lab_batches = [ b for b in MakeBatches(train_labs, args.batch_size, fillvalue=train_labs[-1]) ]
    train_indices = list(range(len(train_feat_batches)))

    # validation batches
    dev_feat_batches = [ b for b in MakeBatches(dev_feats, args.batch_size, fillvalue=dev_feats[-1]) ]
    dev_asp_batches = [ b for b in MakeBatches(dev_asps, args.batch_size, fillvalue=dev_asps[-1]) ]
    dev_lab_batches = [ b for b in MakeBatches(dev_labs, args.batch_size, fillvalue=dev_labs[-1]) ]

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

    # print training information
    print('-'*80)
    print('Training Information')
    print('# of MLP hidden units: %i' % args.mlp_hidden_units)
    print('# of MLP hidden layers: %i' % args.mlp_hidden_layers)
    print('Dropout: %f' % args.dropout)
    print('MLP activation function: %s' % args.mlp_activation)
    print('# of training epochs: %i' % args.num_epochs)
    print('Batch size: %i' % args.batch_size)
    print('Learning rate: %f' % args.learning_rate)
    print('-'*80)

    # start training
    print('Training started...')
    for k in range(args.num_epochs):
        print('-'*80)
        print('Epoch %i' % (k+1))
        progbar = generic_utils.Progbar(len(train_indices)*args.batch_size)
        # shuffle batch indices
        random.shuffle(train_indices)
        for i in train_indices:
            X_feature_batch = np.asarray(train_feat_batches[i], 'float32')
            X_asp_batch = GetAspectFeatures(train_asp_batches[i], asp_encoder)
            Y_lab_batch = GetLabelEncoding(train_lab_batches[i], lab_encoder)
            loss = model.train_on_batch([X_feature_batch, X_asp_batch], Y_lab_batch)
            loss = loss[0].tolist()
            progbar.add(args.batch_size, values=[('train loss', loss)])
        print('Time: %f s' % (time.time()-start_time))

        # evaluate on dev set
        pbar = generic_utils.Progbar(len(dev_feat_batches)*args.batch_size)

        # validation feedforward
        dev_correct = 0
        for i in range(len(dev_feat_batches)):
            X_feature_batch = np.asarray(dev_feat_batches[i], 'float32')
            X_asp_batch = GetAspectFeatures(dev_asp_batches[i], asp_encoder)
            Y_lab_batch = GetLabelEncoding(dev_lab_batches[i], lab_encoder)
            pred = model.predict_classes([X_feature_batch, X_asp_batch], args.batch_size, verbose=0)

            if i != (len(dev_feat_batches)-1):
                dev_correct += np.count_nonzero(np.argmax(Y_lab_batch, axis=1)==pred)
            else:
                num_padding = args.batch_size * len(dev_feat_batches) - len(dev_feats)
                last_idx = args.batch_size - num_padding
                dev_correct += np.count_nonzero(np.argmax(Y_lab_batch[:last_idx], axis=1)==pred[:last_idx])
            pbar.add(args.batch_size)

        # calculate validation accuracy
        dev_acc = float(dev_correct)/len(dev_feats)
        dev_accs.append(dev_acc)
        print('Validation Accuracy: %f' % dev_acc)
        print('Time: %f s' % (time.time()-start_time))

        # save best weights
        if dev_acc > max_acc:
            max_acc = dev_acc
            max_acc_epoch = k
            model.save_weights(model_filename + '_best.hdf5', overwrite=True)

    print(dev_accs)
    print('Best validation accuracy: %f; epoch#%i' % (max_acc, (max_acc_epoch+1)))
    print('Training finished.')
    print('Time: %f s' % (time.time()-start_time))

if __name__ == "__main__":
    main()

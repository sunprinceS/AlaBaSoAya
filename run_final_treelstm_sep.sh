#!/bin/bash
base_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

data_dir=$base_dir/misc_data
sent_dir=$base_dir/src/TreeLSTM_NN_sep
model_dir=$base_dir/models
pred_dir=$base_dir/sent_predictions_treelstm_nn_sep
log=$base_dir/log

i=10
echo 'Sentiment Classification: Restaurant'
#python $sent_dir/train_sent.py --aspects 12 --domain rest --cross-val-index $i

echo 'Sentiment Classification: Laptop'
python $sent_dir/train_sent.py --aspects 81 --domain lapt --cross-val-index $i

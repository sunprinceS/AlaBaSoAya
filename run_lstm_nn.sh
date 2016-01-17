#!/bin/bash
base_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

data_dir=$base_dir/misc_data
sent_dir=$base_dir/src/LSTM_NN
model_dir=$base_dir/models
pred_dir=$base_dir/sent_predictions_lstm_nn

echo 'Sentiment Classification: Restaurant'
i=0
while [ $i -lt 10 ]; do
    python $sent_dir/train_sent.py --aspects 12 --domain rest --cross-val-index $i
    python $sent_dir/test_sent.py --aspects 12 --domain rest --cross-val-index $i \
        --model $model_dir/rest.lstmNN.lstm_units_512_layers_2.mlp_units_512_layers_2_relu.lr1.0e-04.dropout0.3.$i.json \
        --weights $model_dir/rest.lstmNN.lstm_units_512_layers_2.mlp_units_512_layers_2_relu.lr1.0e-04.dropout0.3.${i}_best.hdf5 \
        --output $pred_dir/rest.pol.pred.$i
    let i=i+1
done

echo 'Sentiment Classification: Laptop'
i=0
while [ $i -lt 10 ]; do
    python $sent_dir/train_sent.py --aspects 81 --domain lapt --cross-val-index $i
    python $sent_dir/test_sent.py --aspects 81 --domain lapt --cross-val-index $i \
        --model $model_dir/lapt.lstmNN.lstm_units_512_layers_2.mlp_units_512_layers_2_relu.lr1.0e-04.dropout0.3.$i.json \
        --weights $model_dir/lapt.lstmNN.lstm_units_512_layers_2.mlp_units_512_layers_2_relu.lr1.0e-04.dropout0.3.${i}_best.hdf5 \
        --output $pred_dir/lapt.pol.pred.$i
    let i=i+1
done

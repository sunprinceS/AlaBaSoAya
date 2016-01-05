#!/bin/bash
base_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

data_dir=$base_dir/misc_data
pred_dir=$base_dir/sent_predictions
sent_dir=$base_dir/src/slot_3_sentiment
model_dir=$base_dir/models
pred_dir=$base_dir/sent_predictions

echo 'Sentiment Classification: Restaurant'
i=0
while [ $i -lt 10 ]; do
    python $sent_dir/train_sent.py --aspects 12 --domain rest --cross-val-index $i
    python $sent_dir/test_sent.py --aspects 12 --domain rest --cross-val-index $i \
        --model $model_dir/rest_mlp_units_256_layers_3_relu_lr1.0e-04_dropout0.3.$i.json \
        --weights $model_dir/rest_mlp_units_256_layers_3_relu_lr1.0e-04_dropout0.3.${i}_best.hdf5 \
        --output $pred_dir/rest.pol.pred.$i
    let i=i+1
done

echo 'Sentiment Classification: Laptop'
i=0
while [ $i -lt 10 ]; do
    python $sent_dir/train_sent.py --aspects 81 --domain lapt --cross-val-index $i
    python $sent_dir/test_sent.py --aspects 81 --domain lapt --cross-val-index $i \
        --model $model_dir/lapt_mlp_units_256_layers_3_relu_lr1.0e-04_dropout0.3.$i.json \
        --weights $model_dir/lapt_mlp_units_256_layers_3_relu_lr1.0e-04_dropout0.3.${i}_best.hdf5 \
        --output $pred_dir/lapt.pol.pred.$i
    let i=i+1
done

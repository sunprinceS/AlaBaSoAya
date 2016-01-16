#!/bin/bash

all=all_data
cross=cross_val_data

echo 'Getting restaurant train features'
i=0
while [ $i -lt 10 ]; do
    python $cross/get_cross_validation_features.py \
        $all/restaurant.all $all/restaurant.feat \
        $cross/rest_train.pol.dat.$i $cross/rest_train.pol.feat.$i
    let i=i+1
done
echo 'Getting restaurant test features'
i=0
while [ $i -lt 10 ]; do
    python $cross/get_cross_validation_features.py \
        $all/restaurant.all $all/restaurant.feat \
        $cross/rest_te.pol.dat.$i $cross/rest_te.pol.feat.$i
    let i=i+1
done
echo 'Getting laptop train features'
i=0
while [ $i -lt 10 ]; do
    python $cross/get_cross_validation_features.py \
        $all/laptop.all $all/laptop.feat \
        $cross/lapt_train.pol.dat.$i $cross/lapt_train.pol.feat.$i
    let i=i+1
done
echo 'Getting laptop test features'
i=0
while [ $i -lt 10 ]; do
    python $cross/get_cross_validation_features.py \
        $all/laptop.all $all/laptop.feat \
        $cross/lapt_te.pol.dat.$i $cross/lapt_te.pol.feat.$i
    let i=i+1
done

#!/usr/bin/env sh

auto_encoder_dir="../../../util/auto_encoder"
input_data="../../src/subtask1/Slot1/data/$1.dat"

cd $auto_encoder_dir
if [ -f $input_data ]
then
	output_encode="$1.encode"
	encode_dir="../../encode_data"
	python2 encoder.py $input_data $encode_dir/$output_encode dictionary
	echo "$input_data encoding finish"
else
	echo "$input_data not found"
fi

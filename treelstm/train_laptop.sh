#!/bin/bash

layer_vec="2 3 4";
dim_vec="300 500 750";
epochs=50

for layers in $layer_vec; do
	for dimension in $dim_vec; do
		echo '---------------------------------------------------------------------------'
		printf 'Training LSTM----layers: %s, dimension: %s\n' "$layers" "$dimension"
		log_path=$(printf 'log/sent-dependency.laptop.%s.%s.log' "$layers" "$dimension")
		th sentiment/main.lua -m dependency -l $layers -d $dimension -e $epochs -t laptop > $log_path
	done
done

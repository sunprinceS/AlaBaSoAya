#!/bin/bash

layer_vec="1 2 3 4 5";
dim_vec="150 300 500";
epochs=150

for layers in $layer_vec; do
	for dimension in $dim_vec; do
		echo '---------------------------------------------------------------------------'
		printf 'Training LSTM----layers: %s, dimension: %s\n' "$layers" "$dimension"
		log_path=$(printf 'log/sent-dependency.restaurant.%s.%s.lrn5e-3.log' "$layers" "$dimension")
		th sentiment/main.lua -m dependency -l $layers -d $dimension -e $epochs -t restaurant > $log_path
	done
done

#for layers in $layer_vec; do
	#for dimension in $dim_vec; do
		#echo '---------------------------------------------------------------------------'
		#printf 'Training LSTM----layers: %s, dimension: %s\n' "$layers" "$dimension"
		#log_path=$(printf 'log/sent-dependency.laptop.%s.%s.lrn5-e3.log' "$layers" "$dimension")
		#th sentiment/main.lua -m dependency -l $layers -d $dimension -e $epochs -t laptop > $log_path
	#done
#done


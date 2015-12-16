#!/usr/bin/env sh

auto_encoder_dir="../../../util/auto_encoder"
cd $auto_encoder_dir
	output_embed="$1.embed"

	encode_dir="../../encode_data"
	embed_dir="../../embed"
	size_dir="../../size"

	#python2 encoder.py $input_data $encode_dir/$1.encode dictionary
	th enc.lua -batch_size 1 -model 11.enc -input $encode_dir -embs $embed_dir -size $size_dir
	echo "\n\nEmbedding finish!"

#!/bin/bash

data_dir=raw_data
log=preprocess.log

echo 'Splitting data'
cd $data_dir
#python ../lib/stanford-parser/split.py rest_train.pol.dat
#python ../lib/stanford-parser/split.py rest_te.pol.dat
python ../lib/stanford-parser/split.py lapt_train.asp.dat
python ../lib/stanford-parser/split.py lapt_te.asp.dat
cd ..

echo 'Parsing sentences with Stanford Parser'
cd lib/stanford-parser
#./parse_sentences.sh temp_data/rest_train.pol.dat 2280 temp_data/rest_train_parsed.txt >> ../$log
#./parse_sentences.sh temp_data/rest_te.pol.dat 227 temp_data/rest_te_parsed.txt >> ../$log
./parse_sentences.sh temp_data/lapt_train.asp.dat 2280 temp_data/lapt_train_parsed.txt >> ../$log
./parse_sentences.sh temp_data/lapt_te.asp.dat 227 temp_data/lapt_te_parsed.txt >> ../$log

cd ../..

echo 'Getting parsed sentences'
#python lib/stanford-parser/get_parsed_sentence.py lib/stanford-parser/temp_data/rest_train_parsed.txt parsed_data/restaurant_emb/train/sents.txt
#python lib/stanford-parser/get_parsed_sentence.py lib/stanford-parser/temp_data/rest_te_parsed.txt parsed_data/restaurant_emb/test/sents.txt
python lib/stanford-parser/get_parsed_sentence.py lib/stanford-parser/temp_data/lapt_train_parsed.txt parsed_data/laptop_emb/train/sents.txt
python lib/stanford-parser/get_parsed_sentence.py lib/stanford-parser/temp_data/lapt_te_parsed.txt parsed_data/laptop_emb/test/sents.txt

echo 'Dependency parsing'
#./dependency_parse.sh parsed_data/restaurant_emb/train
#./dependency_parse.sh parsed_data/restaurant_emb/test
./dependency_parse.sh parsed_data/laptop_emb/train
./dependency_parse.sh parsed_data/laptop_emb/test

echo 'Generating vocab'
#python gen_vocab.py parsed_data/restaurant_emb
python gen_vocab.py parsed_data/laptop_emb

echo 'Finished'

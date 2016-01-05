#!/bin/bash

data_dir=all_data
log=all_preprocess.log

echo 'Splitting data'
cd $data_dir
#python ../lib/stanford-parser/all_split.py restaurant.all
#python ../lib/stanford-parser/all_split.py laptop.all
cd ..

echo 'Parsing sentences with Stanford Parser'
cd lib/stanford-parser
#./parse_sentences.sh all_data/restaurant.all 2000 all_data/rest_all_parsed.txt >> ../$log
#./parse_sentences.sh all_data/laptop.all 2500 all_data/lapt_all_parsed.txt >> ../$log

cd ../..

echo 'Getting parsed sentences'
#python lib/stanford-parser/get_parsed_sentence.py lib/stanford-parser/all_data/rest_all_parsed.txt all_data/parsed/restaurant/sents.txt
#python lib/stanford-parser/get_parsed_sentence.py lib/stanford-parser/all_data/lapt_all_parsed.txt all_data/parsed/laptop/sents.txt

echo 'Dependency parsing'
./dependency_parse.sh all_data/parsed/restaurant
./dependency_parse.sh all_data/parsed/laptop

echo 'Generating vocab'
#python gen_all_vocab.py all_data/parsed/restaurant
#python gen_all_vocab.py all_data/parsed/laptop

echo 'Finished'

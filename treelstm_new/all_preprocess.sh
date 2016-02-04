#!/bin/bash

data_dir=all_data
log=all_preprocess.log

echo 'Splitting data'
cd $data_dir
#python ../lib/stanford-parser/all_split.py restaurant.pol.train.all
#python ../lib/stanford-parser/all_split.py laptop.pol.train.all
cd ..

echo 'Parsing sentences with Stanford Parser'
cd lib/stanford-parser
#./parse_sentences.sh all_data/restaurant.pol.train.all 2507 all_data/rest_all_parsed_new.txt >> ../../$data_dir/$log
#./parse_sentences.sh all_data/laptop.pol.train.all 2909 all_data/lapt_all_parsed_new.txt >> ../../$data_dir/$log

cd ../..

echo 'Getting parsed sentences'
#python lib/stanford-parser/get_parsed_sentence.py lib/stanford-parser/all_data/rest_all_parsed_new.txt all_data/restaurant/sents.txt
#python lib/stanford-parser/get_parsed_sentence.py lib/stanford-parser/all_data/lapt_all_parsed_new.txt all_data/laptop/sents.txt

echo 'Constituency parsing'
./dependency_parse.sh all_data/restaurant
./dependency_parse.sh all_data/laptop

echo 'Generating vocab'
#python gen_all_vocab.py all_data/parsed/restaurant
#python gen_all_vocab.py all_data/parsed/laptop

echo 'Finished'

#!/bin/bash

dir=$1
#dir=/project/peskotiveswf/treelstm/lib/stanford-parser/preprocess/laptop/train
java -cp /home/peskotiveswf/AlaBaSoAya/treelstm/lib:/home/peskotiveswf/AlaBaSoAya/treelstm/lib/stanford-parser/stanford-parser.jar:/home/peskotiveswf/AlaBaSoAya/treelstm/lib/stanford-parser/stanford-parser-3.5.1-models.jar ConstituencyParse -tokpath $dir/sents.toks -parentpath $dir/cparents.txt -relpath $dir/rels.txt  < $dir/sents.txt

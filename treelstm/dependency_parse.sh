#!/bin/bash

dir=/project/peskotiveswf/treelstm/lib/stanford-parser/preprocess/laptop/train
java -cp /project/peskotiveswf/treelstm/lib:/project/peskotiveswf/treelstm/lib/stanford-parser/stanford-parser.jar:/project/peskotiveswf/treelstm/lib/stanford-parser/stanford-parser-3.5.1-models.jar DependencyParse -tokpath $dir/sents.toks -parentpath $dir/dparents.txt -relpath $dir/rels.txt  < $dir/sents.txt

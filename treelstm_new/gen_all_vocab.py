"""
Preprocessing script for Stanford Sentiment Treebank data.

"""

import os
import glob
import sys

def build_vocab(filepaths, dst_path, lowercase=True):
    vocab = set()
    for filepath in filepaths:
        with open(filepath) as f:
            for line in f:
                if lowercase:
                    line = line.lower()
                vocab |= set(line.split())
    with open(dst_path, 'w') as f:
        for w in sorted(vocab):
            f.write(w + '\n')

if __name__ == '__main__':

    #base_dir = '/project/peskotiveswf/treelstm/lib/stanford-parser/preprocess/laptop'
    base_dir = sys.argv[1]
    #build_vocab([os.path.join(base_dir, 'train/sents.txt'), os.path.join(base_dir, 'dev/sents.txt')], os.path.join(base_dir, 'vocab.txt'))
    #build_vocab([os.path.join(base_dir, 'train/sents.txt'), os.path.join(base_dir, 'dev/sents.txt')], os.path.join(base_dir, 'vocab-cased.txt'), lowercase=False)
    build_vocab([os.path.join(base_dir, 'sents.txt')], os.path.join(base_dir, 'vocab.txt'))
    build_vocab([os.path.join(base_dir, 'sents.txt')], os.path.join(base_dir, 'vocab-cased.txt'), lowercase=False)

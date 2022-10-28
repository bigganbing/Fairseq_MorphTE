#!/bin/bash

src=en
tgt=it
bpe_num=40000
TEXT=examples/translation/opus-100/$src-$tgt-bpe$bpe_num

python preprocess.py --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/opus100_$src-${tgt}_bpe${bpe_num}_joined_dict \
  --joined-dictionary --workers 20
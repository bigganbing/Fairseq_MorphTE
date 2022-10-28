#!/bin/bash

# Download and prepare the data
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..


# Preprocess/binarize the data
src=de
tgt=en
TEXT=examples/translation/iwslt14.bpe-10000.de-en

python preprocess.py --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.bpe-10000.de-en \
  --workers 20
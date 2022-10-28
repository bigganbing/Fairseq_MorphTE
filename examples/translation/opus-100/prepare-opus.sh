#!/usr/bin/env bash
#
# Adapted from https://github.com/facebookresearch/MIXER/blob/master/prepareData.sh

echo 'Cloning Moses github repository (for tokenization scripts)...'
git clone https://github.com/moses-smt/mosesdecoder.git

echo 'Cloning Subword NMT repository (for BPE pre-processing)...'
git clone https://github.com/rsennrich/subword-nmt.git

SCRIPTS=../mosesdecoder/scripts
TOKENIZER=$SCRIPTS/tokenizer/tokenizer.perl
LC=$SCRIPTS/tokenizer/lowercase.perl
CLEAN=$SCRIPTS/training/clean-corpus-n.perl
BPEROOT=../subword-nmt/subword_nmt
BPE_TOKENS=40000


if [ ! -d "$SCRIPTS" ]; then
    echo "Please set SCRIPTS variable correctly to point to Moses scripts."
    exit
fi

src=en
tgt=it
lang=$src-$tgt
orig=original/$lang
prep=$lang-bpe$BPE_TOKENS
tmp=$prep/tmp

mkdir -p  $prep $tmp

for l in $src $tgt; do
    for d in train dev test ; do
    f=$orig/opus.${lang}-$d.$l
    o=$tmp/$d.$l
    echo $f ====== $o
    cat $f | \
    perl $TOKENIZER -threads 8 -l $l | \
    perl $LC > $o
    echo ""
    done
done

for l in $src $tgt; do
    mv $tmp/dev.$l $tmp/valid.$l
done


TRAIN=$tmp/train.$src-$tgt
BPE_CODE=$prep/code
rm -f $TRAIN
for l in $src $tgt; do
    cat $tmp/train.$l >> $TRAIN
done

echo "learn_bpe.py on ${TRAIN}..."
python $BPEROOT/learn_bpe.py -s $BPE_TOKENS < $TRAIN > $BPE_CODE

for L in $src $tgt; do
    for f in train.$L valid.$L test.$L; do
        echo "apply_bpe.py to ${f}..."
        python $BPEROOT/apply_bpe.py -c $BPE_CODE < $tmp/$f > $prep/$f
    done
done

#rm -r $tmp

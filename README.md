Codes  for the NeurIPS 2022 paper `MorphTE: Injecting Morphology in Tensorized Embeddings`.

### Requirements and Installation

- torch >= 1.9.0
- polyglot
- enchant

```shell
pip install --editable ./
```

### Take IWSLT14-De-En as an example

#### 1. Preprocessing

```shell
# Download and prepare the data (result in "examples/translation/iwslt14.bpe-10000.de-en")
cd examples/translation/
bash prepare-iwslt14.sh
cd ../..

# Preprocess/binarize the data (result in "data-bin/iwslt14.bpe-10000.de-en")
src=de
tgt=en
TEXT=examples/translation/iwslt14.bpe-10000.de-en
python preprocess.py --source-lang $src --target-lang $tgt \
  --trainpref $TEXT/train --validpref $TEXT/valid --testpref $TEXT/test \
  --destdir data-bin/iwslt14.bpe-10000.de-en \
  --workers 20

# Morpheme segmentation (result in "mors/iwslt14deen_bpe10000_morSeg_results.json")
python mor_for_bpe_corpus_sep.py
```

The processed IWSLT14-De-En dataset is available [here](https://drive.google.com/file/d/1-5v6W8rLklz4K_Q6x1Nsni8dleSxIoGf/view?usp=sharing).

#### 2. Training && Evaluating

##### (1) MorphTE Embedding

```shell
# MorphTE with rank=7 and order=3

CUDA_ID=0
SEED=42
EMB_RANK=7
EMB_MODE=MorphTE
SAVE_DIR=iwslt14deen_bpe10000_${EMB_MODE}_rank${EMB_RANK}
DATD_DIR=data-bin/iwslt14.bpe-10000.de-en
MOR_PATH=mors/iwslt14deen_bpe10000_morSeg_results.json
mkdir -p checkpoints/${SAVE_DIR}

# Training
CUDA_VISIBLE_DEVICES=$CUDA_ID python train.py $DATD_DIR \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 20000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-epoch 70 --keep-last-epochs 1 --keep-best-checkpoints 1 \
    --max-tokens 4096 --seed $SEED \
    --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
    --emb_rank $EMB_RANK --emb_mode $EMB_MODE --mor_path $MOR_PATH --amp \
    --save-dir  checkpoints/${SAVE_DIR} | tee -a  checkpoints/${SAVE_DIR}/run.log

# Evaluating
CUDA_VISIBLE_DEVICES=$CUDA_ID python generate.py $DATD_DIR --path checkpoints/${SAVE_DIR}/checkpoint_best.pt --emb_mode $EMB_MODE --emb_rank $EMB_RANK --mor_path $MOR_PATH --batch-size 128 --beam 5 --remove-bpe --quiet | tee -a  checkpoints/${SAVE_DIR}/run.log
```

A trained Transformer model with MorphTE on IWSLT14-De-En is available [here](https://drive.google.com/file/d/1--V_qZZyCLV2KuQYX-dfQT4I71orBPuX/view?usp=sharing).

##### (2) Original Embedding

```shell
# Original Embedding without compression

CUDA_ID=0
SEED=42
EMB_MODE=Original
SAVE_DIR=iwslt14deen_bpe10000_original
DATD_DIR=data-bin/iwslt14.bpe-10000.de-en
mkdir -p checkpoints/${SAVE_DIR}

# Training
CUDA_VISIBLE_DEVICES=$CUDA_ID python train.py $DATD_DIR \
    --arch transformer_iwslt_de_en --share-decoder-input-output-embed \
    --optimizer adam --adam-betas '(0.9, 0.98)' --clip-norm 0.0 \
    --lr 5e-4 --lr-scheduler inverse_sqrt --warmup-updates 4000 \
    --dropout 0.3 --weight-decay 0.0001 \
    --criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
    --max-epoch 70 --keep-last-epochs 1 --keep-best-checkpoints 1 \
    --max-tokens 4096 --seed $SEED \
    --encoder-ffn-embed-dim 1024 --decoder-ffn-embed-dim 1024 \
    --emb_mode $EMB_MODE --amp \
    --save-dir  checkpoints/${SAVE_DIR} | tee -a  checkpoints/${SAVE_DIR}/run.log

# Evaluating
CUDA_VISIBLE_DEVICES=$CUDA_ID python generate.py $DATD_DIR --path checkpoints/${SAVE_DIR}/checkpoint_best.pt --emb_mode $EMB_MODE --batch-size 128 --beam 5 --remove-bpe --quiet | tee -a  checkpoints/${SAVE_DIR}/run.log
```


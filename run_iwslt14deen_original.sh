CUDA_ID=0
SEED=42
EMB_MODE=Original
SAVE_DIR=iwslt14deen_bpe10000_original
DATD_DIR=data-bin/iwslt14.bpe-10000.de-en
mkdir -p checkpoints/${SAVE_DIR}

# training
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
CUDA_VISIBLE_DEVICES=$CUDA_ID python generate.py $DATD_DIR --path checkpoints/${SAVE_DIR}/checkpoint_best.pt \
    --emb_mode $EMB_MODE --batch-size 128 --beam 5 --remove-bpe --quiet | tee -a  checkpoints/${SAVE_DIR}/run.log
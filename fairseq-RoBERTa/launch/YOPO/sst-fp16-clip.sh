#!/bin/bash

GPU=0
TOTAL_NUM_UPDATES=20935
WARMUP_UPDATES=1256
LR=1e-5
NUM_CLASSES=2
MAX_SENTENCES=8
FREQ=4
DATA=SST-2
ADV_LR=1e-1
ADV_STEPS=2
INIT_MAG=1
SEED=9017
MNORM=6e-1
yopos=3

ROBERTA_PATH=pretrained/roberta.large/model.pt
exp_name=yopo-syncdp-${DATA}-iters${TOTAL_NUM_UPDATES}-warmup${WARMUP_UPDATES}-lr${LR}-bsize${MAX_SENTENCES}-freq${FREQ}-advlr${ADV_LR}-advstep${ADV_STEPS}-initmag${INIT_MAG}-fp32-seed${SEED}-beta0.999-morm${MNORM}-yopos${yopos}
log_path=logs/${exp_name}.log
echo "running with GPU"${GPU}
echo ${log_path}


CUDA_VISIBLE_DEVICES=${GPU} \
    python train.py ${DATA}-bin/ \
    --restore-file $ROBERTA_PATH \
    --max-positions 512 \
    --max-sentences $MAX_SENTENCES \
    --max-tokens 4400 \
    --task sentence_prediction \
    --reset-optimizer --reset-dataloader --reset-meters \
    --required-batch-size-multiple 1 \
    --init-token 0 --separator-token 2 \
    --arch roberta_large \
    --save-interval 1 --keep-last-epochs 0 \
    --criterion sentence_prediction \
    --num-classes $NUM_CLASSES \
    --update-freq ${FREQ} \
    --dropout 0.1 --attention-dropout 0.1 \
    --weight-decay 0.1 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-06 \
    --clip-norm 0.0 \
    --save-dir roberta-chks/${exp_name} \
    --lr-scheduler polynomial_decay --lr $LR --total-num-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
    --max-epoch 10 \
    --seed ${SEED} \
    --find-unused-parameters \
    --best-checkpoint-metric accuracy --maximize-best-checkpoint-metric \
    --rand-init-mag ${INIT_MAG} --adv-lr ${ADV_LR} --adv-steps ${ADV_STEPS} \
    --max-norm ${MNORM} \
    --yopo-steps ${yopos} \
    > ${log_path} 2>&1 
    #--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \




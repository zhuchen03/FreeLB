
function run_exp {
    GPU=${1}
    TOTAL_NUM_UPDATES=${2}
    WARMUP_UPDATES=${3}
    LR=${4}
    NUM_CLASSES=${5}
    MAX_SENTENCES=${6}
    FREQ=${7}
    DATA=${8}
    ADV_LR=${9}
    ADV_STEPS=${10}
    INIT_MAG=${11}
    SEED=${12}
    MNORM=${13}

ROBERTA_PATH=pretrained/roberta.large/model.pt
exp_name=FreeLB-syncdp-${DATA}-iters${TOTAL_NUM_UPDATES}-warmup${WARMUP_UPDATES}-lr${LR}-bsize${MAX_SENTENCES}-freq${FREQ}-advlr${ADV_LR}-advstep${ADV_STEPS}-initmag${INIT_MAG}-fp32-seed${SEED}-beta0.999-morm${MNORM}
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
    > ${log_path} 2>&1 &
    #--fp16 --fp16-init-scale 4 --threshold-loss-scale 1 --fp16-scale-window 128 \
}


# run_exp   GPU    TOTAL_NUM_UPDATES    WARMUP_UPDATES  LR      NUM_CLASSES MAX_SENTENCES   FREQ    DATA    ADV_LR  ADV_STEP  INIT_MAG  SEED    MNORM
run_exp      0        1562                 122         1e-5       2           2            8     RTE           0      0       0         4207     0 
run_exp      1        1562                 122         1e-5       2           2            8     RTE           0      0       0         9017     0 
#run_exp      2        2036                 122         1e-5       2           2            8     RTE           6e-2      3      1.6e-1     1234  1.4e-1
#run_exp      3        2036                 122         1e-5       2           2            8     RTE           9e-2      3      1.6e-1     1234  1.4e-1
#run_exp      0        2036                 122         1e-5       2           2            8     RTE           6e-2      3      1.6e-1     9017  1.4e-1

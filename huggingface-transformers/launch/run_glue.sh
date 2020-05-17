#!/usr/bin/env bash

function runexp {

export GLUE_DIR=glue_data
export TASK_NAME=${1}

gpu=${2}      # The GPU you want to use
mname=${3}    # Model name
alr=${4}      # Step size of gradient ascent
amag=${5}     # Magnitude of initial (adversarial?) perturbation
anorm=${6}    # Maximum norm of adversarial perturbation
asteps=${7}   # Number of gradient ascent steps for the adversary
lr=${8}       # Learning rate for model parameters
bsize=${9}    # Batch size
gas=${10}     # Gradient accumulation. bsize * gas = effective batch size
seqlen=512    # Maximum sequence length
hdp=${11}     # Hidden layer dropouts for ALBERT
adp=${12}     # Attention dropouts for ALBERT
ts=${13}      # Number of training steps (counted as parameter updates)
ws=${14}      # Learning rate warm-up steps
seed=${15}    # Seed for randomness
wd=${16}      # Weight decay

expname=FreeLB-${mname}-${TASK_NAME}-alr${alr}-amag${amag}-anm${anorm}-as${asteps}-sl${seqlen}-lr${lr}-bs${bsize}-gas${gas}-hdp${hdp}-adp${adp}-ts${ts}-ws${ws}-wd${wd}-seed${seed}

python examples/run_glue_freelb.py \
  --model_type albert \
  --model_name_or_path ${mname} \
  --task_name $TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir $GLUE_DIR/$TASK_NAME \
  --max_seq_length ${seqlen} \
  --per_gpu_train_batch_size ${bsize} --gradient_accumulation_steps ${gas} \
  --learning_rate ${lr} --weight_decay ${wd} \
  --gpu ${gpu} \
  --output_dir checkpoints/${expname}/ \
  --hidden_dropout_prob ${hdp} --attention_probs_dropout_prob ${adp} \
  --adv-lr ${alr} --adv-init-mag ${amag} --adv-max-norm ${anorm} --adv-steps ${asteps} \
  --expname ${expname} --evaluate_during_training \
  --max_steps ${ts} --warmup_steps ${ws} --seed ${seed} \
  --logging_steps 100 --save_steps 100 \
  --fp16 \
  --comet \
  > logs/${expname}.log 2>&1 &
}


# runexp TASK_NAME  gpu      model_name      adv_lr  adv_mag  anorm  asteps  lr     bsize  grad_accu  hdp  adp      ts     ws     seed      wd
runexp  SST-2       0       albert-xxlarge-v2   1e-1    6e-1      0    2    1e-5     32       1        0.1   0    20935   1256     42     1e-2

runexp  MRPC        0       albert-xxlarge-v2   3e-2       0      0    4    2e-5     16       1        0.1   0      800    200     42     1e-2

runexp  CoLA        0       albert-xxlarge-v2   4e-2    5e-2      0    3    1e-5     16       1        0.1   0     5336    320     42     1e-2

runexp  STS-B       0       albert-xxlarge-v2   1e-1    2e-1      0    3    2e-5     16       1        0.1   0     3598    214     42     1e-2

runexp  MNLI        0       albert-xxlarge-v2   4e-2    8e-2      0    3    3e-5    128       1        0.1   0    10000   1000     42     1e-2

runexp  QQP         0       albert-xxlarge-v2   2e-1  3.2e-1   7e-1    3    5e-5    128       1        0.1   0.1  14000   1000     42     1e-2

runexp  RTE         0       albert-xxlarge-v2   1e-1      0       0    3    3e-5     32       1        0.1   0.1    800    200     42     1e-2

runexp  QNLI        0       albert-xxlarge-v2   5e-2    1e-1      0    3    1e-5      8       4        0.1   0    33112   1986     42     1e-2




#!/usr/bin/env bash

function runexp {

export GLUE_DIR=glue_data
export TASK_NAME=${1}

gpu=${2}
mname=${3}
alr=${4}
amag=${5}
anorm=${6}
asteps=${7}
lr=${8}
bsize=${9}
gas=${10}
seqlen=512
hdp=${11}
adp=${12}
ts=${13}
ws=${14}
seed=${15}
wd=${16}
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
  --logging_steps 1000 --save_steps 1000 \
  --fp16 \
  --comet \
  > logs/${expname}.log 2>&1 &
}


# runexp TASK_NAME  gpu      model_name      adv_lr  adv_mag  anorm  asteps  lr     bsize  grad_accu  hdp  adp    ts     ws       seed      wd
runexp  SST-2     0,1       albert-xxlarge-v2   1e-1    1.0     6e-1    3    1e-5   16       1        0.1   0  20935   1256      42         1e-2

runexp  MNLI       0        albert-xxlarge-v2   4e-2     8e-2      0       3    3e-5   16        8        0.1   0  10000   1000      8023         1e-2

#
runexp  QQP        0        albert-xxlarge-v2   2e-1   3.2e-1   7e-1    3    5e-5   128      1        0.1   0.1  14000   1000      42       1e-2


runexp  QNLI       1        albert-xxlarge-v2   5e-2      1e-1     0       3    1e-5   8       4        0.1   0    33112   1986      42    1e-2




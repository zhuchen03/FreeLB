#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

OUTDIR=data/Old_QNLI

mkdir -p $OUTDIR

wget -O $OUTDIR/QNLI.zip https://firebasestorage.googleapis.com/v0/b/mtl-sentence-representations.appspot.com/o/data%2FQNLI.zip?alt=media&token=c24cad61-f2df-4f04-9ab6-aa576fa829d0
wget -O $OUTDIR/dict.txt https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt
cd $OUTDIR
unzip QNLI.zip

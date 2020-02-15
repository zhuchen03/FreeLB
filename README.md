# Introduction
This repository contains the implementation for FreeLB on GLUE tasks based on both [fairseq](https://github.com/pytorch/fairseq) and [HuggingFace's transformers](https://github.com/huggingface/transformers) libraries, under `./fairseq-RoBERTa/` and `./huggingface-transformers/` respectively.
We also integrated our implementations of vanilla PGD, [FreeAT](https://arxiv.org/abs/1904.12843) and [YOPO](https://arxiv.org/abs/1905.00877) in our fairseq version.
FreeLB is an adversarial training approach for improving transformer-based language models on Natural Language Understanding tasks.
It accumulates the gradient in the ascent steps and updates the parameters with the accumulated gradients, which is approximately equivalent to enlarging the batch size with diversified adversarial examples within different radiuses around the clean input.
FreeLB improves the performance of BERT and RoBERTa on various Natural Language Understanding tasks including Question Answering, Natural Language Inference, and Sentiment Analysis.

For technical details and additional experimental results, please refer to our paper:

Chen Zhu, Yu Cheng, Zhe Gan, Siqi Sun, Tom Goldstein, and Jingjing Liu. [FreeLB: Enhanced Adversarial Training for Language Understanding](https://arxiv.org/abs/1909.11764). In ICLR, 2020.

# What's New
* Feb 15, 2020: Initial release of FreeLB based on fairseq and HuggingFace's transformers. The first one contains our implementations of FreeLB, FreeAT, YOPO for [RoBERTa](https://arxiv.org/abs/1907.11692), while the latter one is FreeLB for [ALBERT](https://arxiv.org/abs/1909.11942).

# Prerequisites
The code is compatible with PyTorch 1.4.0. 
In addition, you need to execute the followings in order, to install the prerequisites for fairseq and HuggingFace's transformers:
```
# Install apex
git clone https://github.com/NVIDIA/apex
cd apex
pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Configure fairseq
cd ../fairseq-RoBERTa
pip install --editable .

# Download and pre-process GLUE data
wget https://gist.githubusercontent.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e/raw/17b8dd0d724281ed7c3b2aeeda662b92809aadd5/download_glue_data.py
python download_glue_data.py --data_dir glue_data --tasks all
source ./examples/roberta/preprocess_GLUE_tasks.sh glue_data ALL

cd ../huggingface-transformers
pip install --editable .
mkdir logs
```

# Launch
The launch scripts are under `./fairseq-RoBERTa/launch/` and `./huggingface-transformers/launch/`, where we have included most of the running scripts for RoBERTa and ALBERT on GLUE dev sets. 
We will release more details in the future.




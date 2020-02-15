import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import qnli_ranking  # load the Pairwise Ranking QNLI task
import argparse
import glob
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--file-path", default="CQA-res/scores-191002", type=str)
parser.add_argument("--out-path", default="CQA-res", type=str)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

glob_files = glob.glob(os.path.join(args.file_path, "CQA_*"))
score_list_ensemble = []
for gfile in glob_files:
    single_dict = torch.load(gfile)
    for n, score_dict in enumerate(single_dict['scores_list']):
        if isinstance(score_dict['scores'], list):
            score_dict['scores'] = torch.cat(score_dict['scores'])

        if len(score_list_ensemble) <= n:
            score_list_ensemble.append(score_dict)
        else:
            # assert score_list_ensemble[n]['idxes'] == score_dict['idxes']
            score_list_ensemble[n]['scores'] += score_dict['scores']

    print(single_dict['model'])

res_dict = {}
ans_map = ["A", "B", "C", "D", "E"]
fs = open(os.path.join(args.out_path, "CQA-ensemble-191002.tsv"), "w+")
for score_dict in score_list_ensemble:
    pred = score_dict['scores'].argmax()
    fs.write("%s,%s\n"%(score_dict['idxes'], ans_map[pred]))


fs.close()


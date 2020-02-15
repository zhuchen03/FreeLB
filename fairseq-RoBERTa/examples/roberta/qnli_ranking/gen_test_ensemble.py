import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import qnli_ranking  # load the Pairwise Ranking QNLI task
import argparse
import glob
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--file-path", default="glue-test-single/scores-0902", type=str)
parser.add_argument("--out-path", default="glue-test-ensemble-0902", type=str)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

glob_files = glob.glob(os.path.join(args.file_path, "QNLI_*"))
score_list_ensemble = []
for gfile in glob_files:
    single_dict = torch.load(gfile)
    for n, score_dict in enumerate(single_dict['scores_list']):
        if len(score_list_ensemble) <= n:
            score_list_ensemble.append(score_dict)
        else:
            assert score_list_ensemble[n]['idxes'] == score_dict['idxes']
            score_list_ensemble[n]['scores'][0] += score_dict['scores'][0]
            score_list_ensemble[n]['scores'][1] += score_dict['scores'][1]
    print(single_dict['model'])

res_dict = {}
for score_dict in score_list_ensemble:
    if score_dict['scores'][0] > score_dict['scores'][1]:
        reses = ['entailment', 'not_entailment']
    else:
        reses = ['not_entailment', 'entailment']

    for idx, res in zip(score_dict['idxes'], reses):
        res_dict[idx] = res

fs = open(os.path.join(args.out_path, "QNLI.tsv"), "w+")
fs.write("id\tlabel\n")
int_keys = [int(k) for k in res_dict.keys()]
sorted_keys = sorted(int_keys)
for key in sorted_keys:
    if key == -1:
        # these are auxiliary keys
        continue
    fs.write("%s\t%s\n"%(str(key), res_dict[str(key)]))

fs.close()


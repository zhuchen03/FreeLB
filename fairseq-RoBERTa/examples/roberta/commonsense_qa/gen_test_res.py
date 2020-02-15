import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import commonsense_qa  # load the Pairwise Ranking QNLI task
import argparse
from collections import OrderedDict
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", default="/scratch0/roberta-chks/freeadv-cqa-iters4000-warmup150-lr1e-05-bsize2-freq8-advlr5e-2-advstep2-initmag5e-3-fp32-seed9017-beta0.98-newinit", type=str)
parser.add_argument("--chk-name", default="checkpoint_best.pt", type=str)
parser.add_argument("--out-path", default="CQA-res", type=str)
parser.add_argument("--test-fname", default="test_rand_split_no_answers.jsonl", type=str)
# parser.add_argument("--tsv-name", default="test.tsv", type=str)
parser.add_argument("--num", default=0, type=int)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

roberta = RobertaModel.from_pretrained(args.model_path, args.chk_name, 'data/CommonsenseQA')
roberta.eval()  # disable dropout
roberta.cuda()  # use the GPU (optional)
nsamples, ncorrect = 0, 0
res_dict = OrderedDict()
scores_list = []

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
out_fs = open(os.path.join(args.out_path, "test_%d.tsv"%args.num), "w+")
ans_map = ["A", "B", "C", "D", "E"]
with open(os.path.join('CQA', args.test_fname)) as h:
    with torch.no_grad():
        for line in h:
            example = json.loads(line)

            scores = []
            for choice in example['question']['choices']:
                input = roberta.encode(
                    'Q: ' + example['question']['stem'],
                    'A: ' + choice['text'],
                    no_separator=True
                )
                score = roberta.predict('sentence_classification_head', input, return_logits=True)
                scores.append(score)

            cat_scores = torch.cat(scores)
            pred = cat_scores.argmax()
            scores_list.append({'idxes': example['id'], 'scores': cat_scores})
            out_fs.write("%s,%s\n"%(example['id'], ans_map[pred]))

# out_fs.write("\n")
out_fs.close()

torch.save({"model": args.model_path, "scores_list": scores_list}, os.path.join(args.out_path, "scores", "CQA_%d.pt"%args.num))

print("Done.")
import json
import torch
from fairseq.models.roberta import RobertaModel
from examples.roberta import qnli_ranking  # load the Pairwise Ranking QNLI task
import argparse
from collections import OrderedDict
import os
import pdb

parser = argparse.ArgumentParser()
parser.add_argument("--model-path", default="/scratch0/roberta-chks/freeadv-qnliranking-iters33112-warmup1986-lr1e-05-bsize1-freq16-advlr5e-2-advstep2-initmag1e-2-fp32-seed9017-beta0.999", type=str)
parser.add_argument("--chk-name", default="checkpoint_best.pt", type=str)
parser.add_argument("--out-path", default="QNLI-ranking-res", type=str)
parser.add_argument("--test-fname", default="test.jsonl", type=str)
parser.add_argument("--num", default=0, type=int)
parser.add_argument("--gpu", default="0", type=str)
args = parser.parse_args()
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu

roberta = RobertaModel.from_pretrained(args.model_path, args.chk_name, 'QNLI-pair-bin')
roberta.eval()  # disable dropout
roberta.cuda()  # use the GPU (optional)
nsamples, ncorrect = 0, 0
res_dict = OrderedDict()
scores_list = []
with open(os.path.join('QNLI-pair-bin', args.test_fname)) as h:
    with torch.no_grad():
        for line in h:
            example = json.loads(line)
            if example['sent1'] == None:
                res_dict[example['idxes'][0]] = 'entailment' # write entailment for lonely senteneces
                ncorrect += 1
                pdb.set_trace()
            else:
                scores = []
                choices = [example['sent0'], example['sent1']]
                for choice in choices:
                    input = roberta.encode(
                        'Q: ' + example['ques'],
                        'A: ' + choice,
                        no_separator=True
                    )
                    score = roberta.predict('sentence_classification_head', input, return_logits=True)
                    scores.append(score)

                scores_list.append({'idxes': example['idxes'], 'scores': scores})

                if scores[0] > scores[1]:
                    reses = ['entailment', 'not_entailment']
                    if 'label' in example:
                        ncorrect += example['label'] == 0
                else:
                    reses = ['not_entailment', 'entailment']
                    if 'label' in example:
                        ncorrect += example['label'] == 1
                for idx, res in zip(example['idxes'], reses):
                    res_dict[idx] = res
                nsamples += 2

if not os.path.exists(args.out_path):
    os.makedirs(args.out_path)
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

torch.save({"model": args.model_path, "scores_list": scores_list}, os.path.join(args.out_path, "scores", "QNLI_%d.pt"%args.num))

print("Accuracy: {}".format(float(ncorrect*2)/nsamples))
print("Done.")
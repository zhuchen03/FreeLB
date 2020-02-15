import json
import random
import pdb
from collections import defaultdict

qnli_test_path = "glue_data/QNLI/test.tsv"


def write_paired_jsonl(in_path, out_path, is_train=True):
    # load the training and dev lines..
    train_lines = open("glue_data/QNLI/dev.tsv").readlines()[1:]
    train_line_dict = defaultdict(list)
    for nl, line in enumerate(train_lines):
        ques = line.split('\t')[1]
        train_line_dict[ques].append(line.strip())

    fs = open(in_path)

    ques_dict = {}
    # repeated_num = 0
    for nl, line in enumerate(fs):
        if nl == 0:
            continue
        idx, ques, sent = line.strip().split('\t')
        if ques in ques_dict:
            ques_dict[ques].append(line.strip())
        else:
            ques_dict[ques] = [line.strip()]

    print("Got {} distinct pairwise ranking formulations".format(len(ques_dict)))
    fs.close()

    out_fs = open(out_path, "w+")
    pair_number = 0
    for key, val in ques_dict.items():
        if len(val) == 1:
            idx0, ques0, sent0 = val[0].split("\t")
            pair_number += 2
            if ques0 not in train_line_dict or len(train_line_dict[ques0])>1:
                # after checking, there's only one question having 3 pairs..
                train_line = train_line_dict[ques0][-1]
            else:
                train_line = train_line_dict[ques0][0]
            idx1, ques1, sent1, label1 = train_line.split("\t")
            idx1 = -1
        elif len(val) == 2:
            idx0, ques0, sent0 = val[0].split("\t")
            idx1, ques1, sent1 = val[1].split("\t")
            pair_number += 2
        else:
            print("Got a question\n{} \n with {} distinct lines".format(key, len(val)))
            exit()
        line_dict = {'idxes': [idx0, idx1], 'ques': ques0, "sent0": sent0, "sent1": sent1}
        out_fs.write(json.dumps(line_dict))
        out_fs.write("\n")

    out_fs.close()
    print("wrote {} question/answer pairs".format(pair_number))

# write_paired_jsonl(qnli_train_path, qnli_train_path.split(".")[0]+".jsonl", is_train=True)
# write_paired_jsonl(qnli_valid_path, qnli_valid_path.split(".")[0]+".jsonl", is_train=False)

write_paired_jsonl(qnli_test_path, qnli_test_path.split(".")[0]+".jsonl", is_train=False)

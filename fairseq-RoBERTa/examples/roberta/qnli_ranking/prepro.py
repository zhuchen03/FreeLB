import json
import random
import pdb

# qnli_train_path = "data/Old_QNLI/train.tsv"
# qnli_valid_path = "data/Old_QNLI/valid.tsv"
# qnli_train_path = "glue_data/QNLI/train.tsv"
# qnli_valid_path = "glue_data/QNLI/dev.tsv"
qnli_test_path = "glue_data/QNLI/test.tsv"

def write_paired_jsonl(in_path, out_path, is_train=True):
    fs = open(in_path)

    ques_dict = {}
    repeated_num = 0
    for nl, line in enumerate(fs):
        if nl == 0:
            continue
        idx, ques, sent, label = line.strip().split('\t')
        if ques in ques_dict:
            ques_dict[ques][label].append(line.strip())
        else:
            ques_dict[ques] = {"entailment": [], "not_entailment": []}
            ques_dict[ques][label].append(line.strip())

        # ques_dict[ques] = {'idxes': [idx0, idx1], "ques": ques0, "sent0": sent0, "sent1": sent1,
        #                "label": label1 == "entailment"}

    print("Got {} distinct pairwise ranking formulations".format(len(ques_dict)))
    fs.close()

    out_fs = open(out_path, "w+")
    pair_number = 0
    for key, val in ques_dict.items():
        if is_train and len(val['entailment']) != len(val['not_entailment']):
            print(key)
        for entail, nonentail in zip(val['entailment'], val['not_entailment']):

            line_list = [entail, nonentail]
            if is_train:
                random.shuffle(line_list)
            idx0, ques0, sent0, label0 = line_list[0].split("\t")
            idx1, ques1, sent1, label1 = line_list[1].split("\t")
            line_dict = {'idxes': [idx0, idx1], 'ques': ques0, "sent0": sent0, "sent1": sent1, "label": int(label1 == "entailment")}
            out_fs.write(json.dumps(line_dict))
            out_fs.write("\n")
            pair_number += 1
    out_fs.close()
    print("wrote {} pairs".format(pair_number))

# write_paired_jsonl(qnli_train_path, qnli_train_path.split(".")[0]+".jsonl", is_train=True)
# write_paired_jsonl(qnli_valid_path, qnli_valid_path.split(".")[0]+".jsonl", is_train=False)

write_paired_jsonl(qnli_test_path, qnli_test_path.split(".")[0]+".jsonl", is_train=False)

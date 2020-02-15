fs_ref = open("glue_data/QNLI/dev.tsv")
fs_gen = open("QNLI-ranking-res/QNLI-val.tsv")

ref_dict = {}
ref_lines = fs_ref.readlines()[1:]
for line in ref_lines:
    idx, ques, sent, label = line.strip().split('\t')
    ref_dict[idx] = label

n_corr = 0
val_lines = fs_gen.readlines()[1:]
for line in val_lines:
    idx, label = line.strip().split('\t')
    if label == ref_dict[idx]:
        n_corr += 1
print("Valid acc: {}".format(float(n_corr)/len(val_lines)))
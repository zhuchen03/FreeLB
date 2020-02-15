# import pdb
# # fs = open("glue_data/QNLI/test.tsv")
# # fs = open("glue_data/QNLI/train.tsv")
# fs = open("glue_data/QNLI/dev.tsv")
#
# freq_dict = {}
# for nl, line in enumerate(fs):
#     if nl == 0:
#         continue
#
#     res = line.strip().split("\t")[1:]
#     if len(res) == 2:
#         res.append(None)
#     ques, sent, label = res
#     if ques not in freq_dict:
#         freq_dict[ques] = [1, sent, [label]]
#     else:
#         freq_dict[ques][0] += 1
#         freq_dict[ques][2].append(label)
#
# n_once = 0
# for key, val in freq_dict.items():
#     # print(val[2])
#     if val[0] < 2:
#         n_once += 1
#         # print(key, val[1])
#         # print(val[2])
#         # pdb.set_trace()
#
# fs.close()
# print("{} distinct queries, {} appeared only once".format(len(freq_dict), n_once))
# #
# train_fs = open("glue_data/QNLI/test.tsv")
# for nl, line in enumerate(train_fs):
#     if nl == 0:
#         continue
#
#     ques = line.split("\t")[1]
#     if ques in freq_dict:
#         freq_dict[ques][0] += 1
#         if freq_dict[ques][2][0] == "entailment":
#             pdb.set_trace()
#
#
# n_once = 0
# for key, val in freq_dict.items():
#     if val < 2:
#         n_once += 1
# print("{} distinct queries, {} appeared only once".format(len(freq_dict), n_once))

# test the overlap of questions between old and new QNLIs
# fs = open("glue_data/QNLI/train.tsv")
fs = open("glue_data/QNLI/test.tsv")
new_ques_list = []
for nl, line in enumerate(fs):
    if nl == 0:
        continue
    ques = line.strip().split("\t")[1]
    new_ques_list.append(ques)

new_ques_set = set(new_ques_list)

old_fs = open("data/Old_QNLI/train.tsv")
old_ques_list = []
for nl, line in enumerate(old_fs):
    if nl == 0:
        continue
    ques = line.strip().split("\t")[1]
    old_ques_list.append(ques)
old_ques_set = set(old_ques_list)

print("new: {}, old: {}, overlap: {}".format(len(new_ques_set), len(old_ques_set), len(new_ques_set & old_ques_set)))

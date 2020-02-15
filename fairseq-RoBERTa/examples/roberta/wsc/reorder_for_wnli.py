import json
from collections import defaultdict
from string import punctuation
import pdb

first_n = 40
# build a list for wnli, and a dict for WSC
# fs = open('glue_data/WNLI/test.tsv')
fs = open('WSC/wnli_test_edited.tsv')
lines = fs.readlines()[1:]
wnli_cand_dict = defaultdict(list)
wnli_query_dict = defaultdict(list)
wnli_key_list = [None]
for line in lines:
    line = line.strip()
    sent1 = line.split("\t")[1]

    if "Chameleon" in sent1:
        key = "Chameleon"
    elif "Muriel " in sent1:
        key = sent1
    else:
        key = sent1[:first_n]
    # appending the candidate sentences, which should contain the query
    wnli_cand_dict[key].append(line.strip().split('\t')[2])
    wnli_query_dict[key].append(line.strip().split('\t')[1])

    if key != wnli_key_list[-1]:
        wnli_key_list.append(key)
wnli_key_list.remove(None)
print("First entry of WNLI key list: {}".format(wnli_key_list[0]))
fs.close()

wsc_fs = open('WSC/test.jsonl')
wsc_dict_all = defaultdict(list)
for line in wsc_fs:
    data_dict = json.loads(line.strip())

    if "Chameleon" in data_dict['text']:
        key = "Chameleon"
    elif "Muriel " in data_dict['text']:
        key = data_dict['text']
    else:
        key = data_dict['text'][:first_n]
    wsc_dict_all[key].append(line)

wsc_fs.close()

# re-order WSC data according to the order of the list
# out_fs = open("WSC/test_reorder.jsonl", "w+")
out_fs = open("WSC/test_reorder_strip.jsonl", "w+")

for key in wnli_key_list:
    if key not in wsc_dict_all:
        # check for bugs
        print("%s not included in WSC"%key)

    verifier = []
    ordered_wsc_line_list = []

    wsc_tuple_list = [(json.loads(line), line) for line in wsc_dict_all[key]]
    wsc_idx_ordered = [wsc_t[0]['idx'] for wsc_t in wsc_tuple_list]
    query_list = [wsc_t[0]['target']['span1_text'] for wsc_t in wsc_tuple_list]
    for nw, wnli_cand in enumerate(wnli_cand_dict[key]):
        # find the matched wnli data in the list of WSC data
        for wsc_data in wsc_tuple_list:
            wsc_dict, wsc_line = wsc_data
            query = wsc_dict['target']['span1_text']
            if query.lower() in wnli_cand.lower():
                wsc_dict['target']['span1_text'] = wsc_dict['target']['span1_text'].strip('.')
                wsc_dict['target']['span2_text'] = wsc_dict['target']['span2_text'].strip('.')

                if wsc_dict['text'] != wnli_query_dict[key][nw]:
                    print(wsc_dict['text'])
                    print(wnli_query_dict[key][nw])
                    print("\n")
                    wsc_dict['text'] = wnli_query_dict[key][nw]
                    # pdb.set_trace()
                out_fs.write("%s\n"%json.dumps(wsc_dict))
                verifier.append(wsc_dict['idx'])
                wsc_tuple_list.remove(wsc_data)
                break

    if wsc_idx_ordered != verifier:
        print("Switch happened for key \n{}\n".format(key))
        print("WSC query list: {}".format(query_list))
        print("Original idxes: {}".format(wsc_idx_ordered))
        print("New idxes     : {}".format(verifier))
        pdb.set_trace()

out_fs.close()
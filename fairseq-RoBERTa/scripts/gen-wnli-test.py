import argparse
from fairseq.models.roberta import RobertaModel
from examples.roberta.wsc import wsc_utils  # also loads WSC task and criterion
import torch
import os
import glob
import pdb

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chk-dir", default="checkpoints", type=str)
    parser.add_argument("--chk-fname", default="checkpoint_best.pt", type=str)
    parser.add_argument("--file-path", default="glue-test-single/scores")
    parser.add_argument("--test-path", default="glue_data")
    parser.add_argument("--out-path", default="glue-test-ensemble-0901")
    parser.add_argument("--ensemble", default=False, action="store_true")
    parser.add_argument("--num", default=0, type=int)
    parser.add_argument("--voting", default=False, action="store_true")
    args = parser.parse_args()
    print(args)

    if args.ensemble:
        glob_files = glob.glob(os.path.join(args.file_path, "WSC_*"))
        query_scores_list, cand_scores_list = [], []
        for n, gfile in enumerate(glob_files):
            score_dict = torch.load(gfile, map_location='cpu')
            query_scores_list.append(score_dict['query_scores'])
            cand_scores_list.append(score_dict['cand_scores'])

            print("Loaded %s" % score_dict['model_name'])

        # query_ensemble = sum(query_score_list) / len(query_score_list)
        # cand_ensemble = sum(cand_score_list) / len(cand_score_list)

        out_fs = open(os.path.join(args.out_path, "WNLI.tsv"), "w+")
        out_fs.write("id\tlabel\n")
        for i in range(len(cand_scores_list[0])):
            if query_scores_list[0][i] is None:
                out_fs.write("%d\t0\n" % (i))
            else:
                cand_score, query_score = 0, 0
                for j in range(len(glob_files)):
                    if args.voting:
                        pred = int((query_scores_list[j][i] >= cand_scores_list[j][i]).all().item() == 1)
                        query_score += pred
                        cand_score += 1 - pred
                    else:
                        cand_score = cand_score + cand_scores_list[j][i]
                        query_score = query_score + query_scores_list[j][i]

                #     print(cand_scores_list[j][i])
                #     print(query_scores_list[j][i])
                # print("Next")
                if args.voting:
                    pred = int(query_score >= cand_score)
                else:
                    pred = int((query_score >= cand_score).all().item() == 1)
                out_fs.write("%d\t%s\n" % (i, pred))
        out_fs.close()

    else:
        roberta = RobertaModel.from_pretrained(args.chk_dir, args.chk_fname, 'WSC/')
        roberta.cuda()
        roberta.eval()

        query_scores_list, cand_scores_list = [], []
        out_fs = open(os.path.join("glue-test-single", "WNLI.tsv"), 'w+')
        out_fs.write("id\tlabel\n")

        with torch.no_grad():
            for idx, (sentence, label) in enumerate(wsc_utils.jsonl_iterator('WSC/test_reorder_strip.jsonl', eval=True)):
                if sentence is None:
                    query_scores_list.append(None)
                    cand_scores_list.append(None)
                    print("Idx {} does not have valid span.".format(idx))
                    out_fs.write("%d\t0\n"%idx)
                    continue

                query_scores, cand_scores = roberta.disambiguate_pronoun_scores(sentence)
                query_scores_list.append(query_scores)
                cand_scores_list.append(cand_scores)

                pred = roberta.disambiguate_pronoun(sentence)
                out_fs.write("%d\t%d\n"%(idx, pred))

        torch.save({"model_name": args.chk_dir, "query_scores": query_scores_list, "cand_scores": cand_scores_list},
                   os.path.join(args.file_path, "WSC_scores_%d.pt" % (args.num)))

        out_fs.close()
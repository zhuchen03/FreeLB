import argparse
import os
from fairseq.models.roberta import RobertaModel
import torch
from fairseq import tasks, utils
import pdb

def get_tokens(line, roberta, task):
    tokens = line.strip().split('\t')
    if task == "RTE":
        sent1, sent2 = tokens[1], tokens[2]
        tokens = roberta.encode(sent1, sent2)
    elif task == "MRPC":
        sent1, sent2 = tokens[3], tokens[4]
        tokens = roberta.encode(sent1, sent2)
    elif task == "CoLA":
        sent = tokens[1]
        tokens = roberta.encode(sent)
    elif task == "SST-2":
        sent = tokens[1]
        tokens = roberta.encode(sent)
    elif task == "STS-B":
        sent1, sent2 = tokens[7], tokens[8]
        tokens = roberta.encode(sent1, sent2)
    elif task == "QQP":
        sent1, sent2 = tokens[1], tokens[2]
        tokens = roberta.encode(sent1, sent2)
    elif task == "QNLI":
        sent1, sent2 = tokens[1], tokens[2]
        tokens = roberta.encode(sent1, sent2)
    elif task == "WNLI":
        sent1, sent2 = tokens[1], tokens[2]
        tokens = roberta.encode(sent1, sent2)
    elif task == "MNLI-m":
        sent1, sent2 = tokens[-2], tokens[-1]
        tokens = roberta.encode(sent1, sent2)
    elif task == "MNLI-mm":
        sent1, sent2 = tokens[-2], tokens[-1]
        tokens = roberta.encode(sent1, sent2)
    elif task == "AX":
        sent1, sent2 = tokens[-2], tokens[-1]
        tokens = roberta.encode(sent1, sent2)
    else:
        print("Task {} undefined".format(task))
        exit()
    return tokens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chk-dir", default="checkpoints", type=str)
    parser.add_argument("--chk-fname", default="checkpoint_last.pt", type=str)
    parser.add_argument("--task", default="sentence_prediction", type=str)
    parser.add_argument("--dset", default="CoLA", type=str)
    parser.add_argument("--test-path", default="glue_data")
    parser.add_argument("--out-path", default="glue-test")
    parser.add_argument("--num", default=0, type=int)
    parser.add_argument("--batch-size", default=32, type=int)
    parser.add_argument("--num-classes", default=2, type=int)
    parser.add_argument("--max-positions", default=512, type=int)
    parser.add_argument("--regression-target", default=False, type=bool)
    parser.add_argument("--dataset-impl", default=None)
    parser.add_argument("--init-token", default=0, type=int)
    parser.add_argument("--no-shuffle", default=True, type=int)
    parser.add_argument("--seed", default=1, help="Shall not be used. Placeholder")
    parser.add_argument("--truncate-sequence", default=False)
    args = parser.parse_args()
    print(parser)

    args.data = args.dset+"-bin"

    bin_task = "MNLI" if "MNLI" in args.dset or args.dset=="AX" else args.dset

    roberta = RobertaModel.from_pretrained(
        args.chk_dir,
        checkpoint_file=args.chk_fname,
        data_name_or_path=bin_task+"-bin"
    )

    # task = tasks.setup_task(args)
    # task.load_dataset("test")
    # itr = task.get_batch_iterator(
    #     dataset=task.dataset("test"),
    #     max_tokens=4400,
    #     max_sentences=args.batch_size,
    #     max_positions=512,
    #     ignore_invalid_inputs=False,
    #     required_batch_size_multiple=1,
    #     seed=1,
    # ).next_epoch_itr(shuffle=False)
    # for batch in itr:
    #     pdb.set_trace()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.target_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.cuda()
    roberta.eval()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not os.path.exists(os.path.join(args.out_path, "scores")):
        os.makedirs(os.path.join(args.out_path, "scores"))

    if args.dset == "MNLI-m":
        test_path = os.path.join(args.test_path, "MNLI", "test_matched.tsv")
    elif args.dset == "MNLI-mm":
        test_path = os.path.join(args.test_path, "MNLI", "test_mismatched.tsv")
    elif args.dset == "AX":
        test_path = os.path.join(args.test_path, "diagnostic", "diagnostic.tsv")
    else:
        test_path = os.path.join(args.test_path, args.dset, "test.tsv")

    with open(test_path) as fin:
        fin.readline()
        out_fs = open(os.path.join(args.out_path, args.dset+".tsv"), "w+")
        out_fs.write("id\tlabel\n")

        scores_list = []
        with torch.no_grad():
            for index, line in enumerate(fin):
                # tokens = line.strip().split('\t')
                # sent1, sent2, target = tokens[1], tokens[2], tokens[3]
                # tokens = roberta.encode(sent1, sent2)
                tokens = get_tokens(line, roberta, args.dset)
                if args.dset == "STS-B":
                    prediction_label = roberta.predict('sentence_classification_head', tokens, return_logits=True)
                    out_fs.write("%d\t%f\n"%(index, prediction_label.item()))
                    scores_list.append(prediction_label)
                else:
                    logit = roberta.predict('sentence_classification_head', tokens, return_logits=True)
                    prediction = logit.argmax().item()
                    prediction_label = label_fn(prediction)
                    out_fs.write("%d\t%s\n" % (index, prediction_label))
                    scores_list.append(logit)
            scores_mat = torch.cat(scores_list, 0)

        torch.save({"model_name": args.chk_dir, "scores": scores_mat}, os.path.join(args.out_path, "scores", "%s_scores_%d.pt"%(args.dset, args.num)))

    out_fs.close()
    print("done.")


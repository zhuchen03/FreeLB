import argparse
import os
from fairseq.models.roberta import RobertaModel
import torch
import pdb
import glob

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
    parser.add_argument("--file-path", default="glue-test-single-nomnli/scores")
    parser.add_argument("--task", default="CoLA", type=str)
    parser.add_argument("--test-path", default="glue_data")
    parser.add_argument("--out-path", default="glue-test-ensemble")
    args = parser.parse_args()
    print(args)

    bin_task = "MNLI" if "MNLI" in args.task or args.task=="AX" else args.task

    roberta = RobertaModel.from_pretrained(
        args.chk_dir,
        checkpoint_file=args.chk_fname,
        data_name_or_path=bin_task+"-bin"
    )

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.target_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    # roberta.cuda()
    # roberta.eval()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    glob_files = glob.glob(os.path.join(args.file_path, "%s_*"%args.task))
    score_list = []
    for n, gfile in enumerate(glob_files):
        score_dict = torch.load(gfile, map_location='cpu')
        score_list.append(score_dict['scores'])
        print("Loaded %s"%score_dict['model_name'])

    ensemble = sum(score_list) / len(score_list)

    if args.task == "STS-B":
        prediction = ensemble.view(-1)
    else:
        prediction = torch.argmax(ensemble, dim=1)

    out_fs = open(os.path.join(args.out_path, args.task + ".tsv"), "w+")
    out_fs.write("id\tlabel\n")
    for index in range(ensemble.size(0)):
        if args.task == "STS-B":
            out_fs.write("%d\t%f\n" % (index, prediction[index].item()))
        else:
            prediction_label = label_fn(prediction[index].item())
            out_fs.write("%d\t%s\n" % (index, prediction_label))
    out_fs.close()


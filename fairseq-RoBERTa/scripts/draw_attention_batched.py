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

def get_entropy(attn_mat, tokens):
    # attn_mat: batch_size x shape
    # attn_mat = attn_mat.view(attn_mat.size(0), -1)
    # batch_size x num_heads x tok_len x tok_len
    bsize, nheads, toklen, _ = attn_mat.size()
    # attn_mat = attn_mat.transpose(1,2).contiguous().view(bsize, toklen, -1) / nheads
    # weighted_likelihood = - attn_mat * torch.log(torch.clamp(attn_mat, min=1e-10))
    # mask = (tokens != 1).float().unsqueeze(2).cuda()
    # total_entropy = torch.sum(mask * weighted_likelihood)
    cls_head_attn = attn_mat[:,:,0,:].contiguous()#.view(bsize, -1) / nheads
    total_entropy = torch.sum(cls_head_attn * torch.clamp(cls_head_attn, min=1e-10)).item()
    return total_entropy

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chk-dir", default="checkpoints", type=str)
    parser.add_argument("--chk-fname", default="checkpoint_best.pt", type=str)
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
    parser.add_argument("--separator-token", default=2, type=int)
    parser.add_argument("--no-shuffle", default=True, type=int)
    parser.add_argument("--seed", default=1, help="Shall not be used. Placeholder")
    parser.add_argument("--truncate-sequence", default=False)
    parser.add_argument("--attn-layer-idx", default=0, type=int)
    parser.add_argument("--gpu", default="0", type=str)
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    bin_task = "MNLI" if "MNLI" in args.dset or args.dset == "AX" else args.dset
    args.data = bin_task+"-bin"

    roberta = RobertaModel.from_pretrained(
        args.chk_dir,
        checkpoint_file=args.chk_fname,
        data_name_or_path=bin_task+"-bin"
    )

    # debug
    # untrained = torch.load('pretrained/roberta.large/model.pt')
    # roberta.model.load_state_dict(untrained['model'], strict=False)
    # pdb.set_trace()

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.target_dictionary.nspecial]
    )
    ncorrect, nsamples = 0, 0
    roberta.to('cuda')
    roberta.eval()
    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not os.path.exists(os.path.join(args.out_path, "scores")):
        os.makedirs(os.path.join(args.out_path, "scores"))

    task = tasks.setup_task(args)

    split = "valid"

    task.load_dataset(split)
    itr = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=4400,
        max_sentences=args.batch_size,
        max_positions=512,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
    ).next_epoch_itr(shuffle=False)
    total_entropy, total_attns = 0, 0
    total_corr, total_samples = 0, 0
    with torch.no_grad():

        scores_list = []
        for batch in itr:

            tokens = batch['net_input']['src_tokens']
            labels = batch['target']
            ids = batch['id']
            if args.dset == "STS-B":
                prediction_label, attn_list = roberta.predict_attn('sentence_classification_head', tokens, return_logits=True)
                scores_list.append(prediction_label)
                n_corr = torch.sum(torch.abs(prediction_label-labels))
            else:
                logit, attn_list = roberta.predict_attn('sentence_classification_head', tokens, return_logits=True)
                prediction = logit.argmax(dim=1)
                n_corr = torch.sum(prediction.view(-1)==labels.view(-1).cuda()).item()
                scores_list.append(logit)
            total_corr += n_corr
            # batch_size x num_heads x tok_len x tok_len
            # for attn in attn_list:
            #     batch_sum_entropy = get_entropy(attn, tokens)
            #     total_entropy += batch_sum_entropy
            #     # number_of_heads x token_lens
            #     total_attns += tokens.size(0)
            batch_sum_entropy = get_entropy(attn_list[-1], tokens)
            total_entropy += batch_sum_entropy
            total_attns += tokens.size(0) * attn_list[-1].size(1)
            total_samples += tokens.size(0)
        scores_mat = torch.cat(scores_list, 0)

        # torch.save({"model_name": args.chk_dir, "scores": scores_mat}, os.path.join(args.out_path, "scores", "%s_scores_%d.pt"%(args.dset, args.num)))
        print("Metric: {}".format(total_corr / float(total_samples)))
        print("Mean Entropy on dev set: {}".format(total_entropy/total_attns))


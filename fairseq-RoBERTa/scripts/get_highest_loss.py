import argparse
import os
from fairseq.models.roberta import RobertaModel
import torch
from fairseq import tasks
import torch.nn.functional as F
from fairseq.data import Dictionary
from my_utils import visualize_input_grad
import pdb
import math
import numpy as np
import sys

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


def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    The reversible bpe codes work on unicode strings.
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    This is a signficant percentage of your normal, say, 32K bpe vocab.
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    """
    bs = list(range(ord("!"), ord("~")+1))+list(range(ord("¡"), ord("¬")+1))+list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8+n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def decode(tokens, decoder, byte_decoder):
    text = ''.join([decoder[token.item()] for token in tokens])
    text = bytearray([byte_decoder[c] for c in text]).decode('utf-8', errors='replace')
    return text

def load_dictionary(filename):
    """Load the dictionary from the filename

    Args:
        filename (str): the filename
    """
    dictionary = Dictionary.load(filename)
    dictionary.add_symbol('<mask>')
    return dictionary


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

def get_loss(args, embeds, batch, roberta_hub):
    tokens = batch['net_input']['src_tokens'].cuda()
    labels = batch['target'].view(-1).cuda()
    ids = batch['id']
    if args.dset == "STS-B":
        logits = roberta_hub.predict_from_embed('sentence_classification_head', tokens, embeds,
                                            return_logits=True)
        loss = F.mse_loss(
            logits,
            labels,
            reduction="sum",
        )
        prediction = logits
    else:
        logit = roberta_hub.predict_from_embed('sentence_classification_head', tokens, embeds,
                                           return_logits=True)
        loss = F.nll_loss(
            F.log_softmax(logit, dim=-1, dtype=torch.float32),
            labels,
            reduction="sum",
        )

        prediction = logit.argmax(dim=1).detach()

    return loss, prediction


def run_one_trial(args, roberta_hub, t, batch, max_norm, early_stop=False):
    max_loss, this_loss = -1, 0
    satisfy_count = 0
    n_iters = 0
    tol = args.tol
    hist_len = 50
    print("===== Trial {}".format(t))
    adv_lr = args.adv_lr
    # while satisfy_count < 30 and n_iters < 10000:
    running_history = [-1]
    tokens = batch['net_input']['src_tokens'].cuda()
    embeds_init = roberta_hub.model.decoder.sentence_encoder.embed_tokens(tokens.cuda()).detach().clone()
    input_mask = (batch['net_input']['src_tokens'] != 1).to(embeds_init)
    delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
    mag = max_norm / torch.sqrt(
        batch['net_input']['src_lengths'].to(delta) * embeds_init.size(-1))
    delta = (delta * mag.view(-1, 1, 1)).detach()
    delta.requires_grad_()
    # while satisfy_count < hist_len and n_iters < 5000:
    while satisfy_count < hist_len and n_iters < 2000:
        n_iters += 1
        loss, prediction = get_loss(args, embeds_init + delta, batch, roberta_hub)

        if n_iters == 1:
            print("Prediction at initialization: {}, label: {}".format(prediction.item(), batch['target'].item()))

        delta_grad = torch.autograd.grad([loss], delta)[0]

        embed_grad_norm = max(torch.norm(delta_grad, p=2).item(), 1e-10)
        delta = (delta + adv_lr * delta_grad / embed_grad_norm).detach()
        if n_iters in [2000]:
            adv_lr *= 0.1
            print("* Adjusted LR to {}".format(adv_lr))

        delta_norm = torch.norm(delta).item()
        if delta_norm > max_norm:
            delta.data = delta.data / delta_norm * max_norm
        delta.requires_grad_()

        this_loss = loss.item()

        delta_change = abs((this_loss - max_loss) / this_loss)

        # delta_history = np.std(running_history) / embed_grad_norm
        delta_history = np.std(running_history) / this_loss
        if len(running_history) < hist_len//2:
            running_history.append(this_loss)
        else:
            running_history = running_history[1:]
            running_history.append(this_loss)

        if (delta_change < tol or delta_history < tol) and delta_norm >= max_norm*0.999:
            satisfy_count += 1
        else:
            satisfy_count = 0

        if n_iters==1 or n_iters % 100 == 0:
            print("Step {}, satisfy_count: {}, loss {:.2e}, max loss {:.2e}, delta change {:.2e}, delta history {:.2e} delta norm: {:.2e}, delta grad norm: {:.2e}".format(n_iters,
                                                                                                    satisfy_count,
                                                                                                    this_loss, max_loss,
                                                                                                    delta_change,
                                                                                                    delta_history,
                                                                                                    torch.norm(
                                                                                                        delta).item(),
                                                                                                    torch.norm(
                                                                                                        delta_grad)))
            sys.stdout.flush()
        # opt.zero_grad()
        if this_loss > max_loss:
            max_loss = this_loss

        torch.cuda.empty_cache()
        roberta_hub.model.zero_grad()
        if early_stop and max_loss > -math.log(0.5):
            break

    print("* Trial {}, max loss: {}".format(t, max_loss))
    return max_loss

def get_correct_idx(batch_iter, model_path, roberta_hub):
    correct_idx_list = []


    n_corr, n_total = 0, 0
    with torch.no_grad():
        for nb, batch in enumerate(batch_iter):
            tokens = batch['net_input']['src_tokens'].cuda()
            labels = batch['target'].view(-1).cuda()
            ids = batch['id']

            logit = roberta_hub.predict('sentence_classification_head', tokens, return_logits=True)
            prediction = logit.argmax(dim=1)
            correct_flag = prediction.view(-1) == labels
            correct_idx_list += [id.item() for id in ids[correct_flag]]
            n_corr += torch.sum(correct_flag)
            n_total += correct_flag.size(0)
    print("*** Loaded model from {}, with acc {}".format(model_path, float(n_corr)/float(n_total)))
    return correct_idx_list

def check_one_radius(radius, roberta_hub, batch, early_stop=False):
    tokens = batch['net_input']['src_tokens'].cuda()
    labels = batch['target'].view(-1).cuda()
    ids = batch['id']

    embeds_init = roberta_hub.model.decoder.sentence_encoder.embed_tokens(tokens.cuda()).detach().clone()
    print("Max norm is: {}".format(radius))
    init_loss, init_pred = get_loss(args, embeds_init, batch, roberta_hub)
    print("**** Clean Loss: {}".format(init_loss))
    # max_norm = args.max_norm
    max_loss = run_one_trial(args, roberta_hub, 0, batch, radius, early_stop=early_stop)
    return max_loss, init_loss.item()


def get_radiuses(single_iter, roberta_hub, resume_dict=None):

    epsilons = np.linspace(args.max_norm, 1e-2, args.norm_steps)
    epsilons = np.concatenate([epsilons, np.linspace(9e-3, 1e-4, args.norm_steps)])
    if resume_dict is not None:
        eps_dict = resume_dict['eps_dict']
        max_loss_dict = resume_dict['max_loss_dict']
        init_loss_dict = resume_dict['init_loss_dict']
    else:
        eps_dict = {}
        max_loss_dict, init_loss_dict = {}, {}
    for nb, batch in enumerate(single_iter):
        if batch['id'].item() in eps_dict:
            continue
        # tokens = batch['net_input']['src_tokens'].cuda()
        # embeds_init = roberta_hub.model.decoder.sentence_encoder.embed_tokens(tokens).data.detach()
        # init_loss, init_pred = get_loss(args, embeds_init, batch, roberta_hub)
        # if init_loss > -math.log(0.5):
        #     pdb.set_trace()
        for epsilon in epsilons:
            if epsilon == 0:
                print("Sample {} does not have valid epsilon. Must be a bug! Or filter samples into clean samples first".format(batch['id']))
                exit()
            max_loss, init_loss = check_one_radius(epsilon, roberta_hub, batch, early_stop=True)
            if max_loss <= -math.log(0.5):
                break
            else:
                print("Sample {}, max loss {} @ radius {}, exceeds thershold.".format(batch['id'], max_loss, epsilon))
        eps_dict[batch['id'].item()] = epsilon
        max_loss_dict[batch['id'].item()] = max_loss
        init_loss_dict[batch['id'].item()] = init_loss
        print("**** Got eps for sample {}: {}, with loss {}".format(batch['id'].item(), epsilon, max_loss))
        torch.save({'eps_dict': eps_dict, 'max_loss_dict': max_loss_dict, "init_loss_dict": init_loss_dict},
                   'losses/{}-maxeps{}-res-dict-anchor-{}.pt'.format(args.dset, args.max_norm, args.suffix))
    return eps_dict, max_loss_dict, init_loss_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chk-dirs", default=["analysis-models/RTE-adv-2e-2_3_5e-3-89.21-5",
            "analysis-models/freeadv-syncdp-RTE-iters2036-warmup122-lr1e-5-bsize2-freq8-advlr2e-2-advstep3-initmag5e-3-fp32-seed4207-abi0-beta0.999-stdadv",
            "analysis-models/RTE-baseline-86.69-4"], type=str, nargs="+", help="Will use the first one as anchor")
    parser.add_argument("--chk-fname", default="checkpoint_best.pt", type=str)
    parser.add_argument("--task", default="sentence_prediction", type=str)
    parser.add_argument("--dset", default="RTE", type=str)
    parser.add_argument("--test-path", default="glue_data")
    parser.add_argument("--out-path", default="glue-test")
    parser.add_argument("--num", default=0, type=int)
    # parser.add_argument("--batch-size", default=2, type=int)
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
    parser.add_argument("--tol", default=1e-5, type=float)
    parser.add_argument("--n-trials", default=100, type=int)
    # parser.add_argument("--sample-idx", default=[12,13], type=int, nargs="+")
    # parser.add_argument("--max-norm-ratio", default=0.22, type=float)
    parser.add_argument("--max-norm", default=0.2, type=float)
    parser.add_argument("--norm-steps", default=11, type=int)
    parser.add_argument("--adv-lr", default=5e-3, type=float)
    # parser.add_argument("--init-mag-ratio", default=0.01, type=float)
    parser.add_argument("--out-fname", default='init.pt', type=str)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--suffix", default="", type=str)
    args = parser.parse_args()
    print(args)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    bin_task = "MNLI" if "MNLI" in args.dset or args.dset == "AX" else args.dset
    args.data = bin_task+"-bin"

    ncorrect, nsamples = 0, 0

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not os.path.exists(os.path.join(args.out_path, "scores")):
        os.makedirs(os.path.join(args.out_path, "scores"))

    task = tasks.setup_task(args)

    split = "valid"

    task.load_dataset(split)
    batch_itr = task.get_batch_iterator(
        dataset=task.dataset(split),
        max_tokens=4400,
        max_sentences=16,
        max_positions=512,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
    ).next_epoch_itr(shuffle=False)

    total_entropy, total_attns = 0, 0
    total_corr, total_samples = 0, 0
    scores_list = []

    total_norm_grad, total_tokens = 0, 0

    # 1. get the intersection where all models make the correct prediction
    roberta_hub_list = []
    for n_chk, chk_dir in enumerate(args.chk_dirs):
        roberta = RobertaModel.from_pretrained(
            chk_dir,
            checkpoint_file=args.chk_fname,
            data_name_or_path=bin_task + "-bin"
        )
        roberta.to('cuda')
        roberta.eval()
        roberta_hub_list.append(roberta)

        correct_idx_list = get_correct_idx(batch_itr, chk_dir, roberta)

        if n_chk == 0:
            valid_set = set(correct_idx_list)
        else:
            valid_set = valid_set & set(correct_idx_list)
    valid_idx_list = sorted(list(valid_set))

    # 2. get the radius for robust nets
    single_valid_iter = task.get_batch_iterator_from_idx(
        dataset=task.dataset(split),
        max_tokens=4400,
        max_sentences=1,
        max_positions=512,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        idx_list=valid_idx_list
    ).next_epoch_itr(shuffle=False)
    if args.resume:
        resume_dict = torch.load(args.resume)
    else:
        resume_dict = None
    eps_dict, max_loss_dict, init_loss_dict = get_radiuses(single_valid_iter, roberta_hub_list[0], resume_dict)
    torch.save({'eps_dict': eps_dict, 'max_loss_dict': max_loss_dict, "init_loss_dict":init_loss_dict}, 'losses/{}-maxeps{}-res-dict-anchor-{}.pt'.format(args.dset, args.max_norm, args.suffix))

    # 3. Get the max loss for other models in such radiuses
    init_loss_dict_list, max_loss_dict_list = [init_loss_dict], [max_loss_dict]
    for nr, roberta_hub in enumerate(roberta_hub_list[1:]):
        this_init_loss_dict, this_max_loss_dict = {}, {}
        for nb, batch in enumerate(single_valid_iter):
            id_int = batch['id'].item()
            eps = eps_dict[id_int]
            max_loss, init_loss = check_one_radius(eps, roberta_hub, batch)
            this_init_loss_dict[id_int] = init_loss
            this_max_loss_dict[id_int] = max_loss
            print("Model {}, sample {}, this max loss: {}, anchor max loss: {}".format(nr, id_int, max_loss, max_loss_dict_list[0][id_int]))

            torch.save({'eps_dict': eps_dict, 'max_loss_dict': this_max_loss_dict, "init_loss_dict": this_init_loss_dict},
                       'losses/{}-maxeps{}-res-dict-{}-{}.pt'.format(args.dset, args.max_norm, nr, args.suffix))
        init_loss_dict_list.append(this_init_loss_dict)
        max_loss_dict_list.append(this_max_loss_dict)
    torch.save({'eps_dict': eps_dict, 'max_loss_dict': max_loss_dict_list, "init_loss_dict": init_loss_dict_list},
               'losses/{}-maxeps{}-res-dict-all-{}.pt'.format(args.dset, args.max_norm, args.suffix))






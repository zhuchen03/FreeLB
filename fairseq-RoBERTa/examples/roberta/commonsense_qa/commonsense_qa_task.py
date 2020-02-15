# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os

import numpy as np
import torch

from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    IdDataset,
    ListDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
)
from fairseq.tasks import FairseqTask, register_task
import pdb

@register_task('commonsense_qa')
class CommonsenseQATask(FairseqTask):
    """Task to finetune RoBERTa for Commonsense QA."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR',
                            help='path to data directory; we load <split>.jsonl')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--num-classes', type=int, default=5)
        parser.add_argument('--adv-lr', type=float, default=0)
        parser.add_argument('--adv-steps', type=int, default=0)
        parser.add_argument('--rand-init-mag', type=float, default=0)
        parser.add_argument('--grad-square', default=False, action="store_true")
        parser.add_argument('--adv-begin-iter', default=-1, type=int)
        parser.add_argument('--no-sync-dp', default=False, action="store_true")
        parser.add_argument('--per-step-adv', default=False, action="store_true")
        parser.add_argument('--std-adv', default=False, action="store_true")
        parser.add_argument("--max-norm", default=-1, type=float)
        parser.add_argument("--yopo-steps", default=1, type=int)
    def __init__(self, args, vocab):
        super().__init__(args)
        self.vocab = vocab
        self.mask = vocab.add_symbol('<mask>')

        self.bpe = encoders.build_bpe(args)

    @classmethod
    def load_dictionary(cls, filename):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.criterion == 'sentence_ranking', 'Must set --criterion=sentence_ranking'

        # load data and label dictionaries
        vocab = cls.load_dictionary(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(vocab)))

        return cls(args, vocab)

    def freelb_train_step(self, sample, model, criterion, optimizer, ignore_grad=False, num_updates=0):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        # self.model.decoder.sentence_encoder.token_embed # the variable
        # self.model.decoder.sentence_encoder.embed_tokens # the embedding layer
        model.train()
        total_loss = 0
        if self.args.rand_init_mag > 0 and num_updates >= self.args.adv_begin_iter:
            token_embeds_list = []
            delta_list = []
            total_len = 0
            for idx in range(self.args.num_classes):
                input_name = 'net_input{idx}'.format(idx=idx + 1)
                total_len += torch.Tensor(sample[input_name]['src_lengths'])

            for idx in range(self.args.num_classes):
                input_name = 'net_input{idx}'.format(idx=idx+1)
                embeds_init = model.decoder.sentence_encoder.embed_tokens(sample[input_name]['src_tokens'])
                input_mask = (sample[input_name]['src_tokens'] != 1).to(embeds_init)
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                mag = self.args.rand_init_mag / torch.sqrt((total_len).to(delta) * embeds_init.size(-1))
                delta = (delta * mag.view(-1, 1, 1)).detach()
                delta.requires_grad_()
                token_embeds_list.append(delta + embeds_init)
                delta_list.append(delta)
            loss, sample_size, logging_output = criterion(model, sample=sample, token_embeds=token_embeds_list,
                                                          init_dp=True)
        else:
            delta_list = [0 for _ in range(self.args.num_classes)]
            loss, sample_size, logging_output = criterion(model, sample, init_dp=True)
        if ignore_grad:
            loss *= 0
            pdb.set_trace()  # check if this is ever called; if not, remove it directly!

        if num_updates >= self.args.adv_begin_iter:
            loss = loss / (1 + self.args.adv_steps)

        optimizer.backward(loss)
        total_loss += loss.detach()

        if self.args.rand_init_mag > 0 and num_updates >= self.args.adv_begin_iter:
            delta_grad_list = [delta.grad.clone().detach() for delta in delta_list]
        else:
            delta_grad_list = [embed.grad.clone().detach() for embed in model.decoder.sentence_encoder.token_embed_cache]

        for t_adv in range(self.args.adv_steps):
            if num_updates < self.args.adv_begin_iter:
                break

            denorm = 0
            for dgrad in delta_grad_list:
                if self.args.fp16:
                    dgrad = dgrad.float()
                denorm += torch.sum(dgrad.view(dgrad.size(0), -1).float()**2, dim=1).view(-1, 1, 1)

            denorm = torch.clamp(torch.sqrt(denorm).detach(), min=1e-10).to(embeds_init)

            token_embeds_list = []
            total_norm = 0
            for idx in range(self.args.num_classes):
                delta = (delta_list[idx] + self.args.adv_lr * delta_grad_list[idx] / denorm).detach()
                total_norm += torch.sum(delta.view(delta.size(0), -1).float() ** 2, 1).detach()
                delta.requires_grad_()
                delta_list[idx] = delta

                adv_embeds_init = model.decoder.sentence_encoder.embed_tokens(sample['net_input{idx}'.format(idx=idx+1)]['src_tokens'])
                token_embeds_list.append(delta+adv_embeds_init)

            total_norm = torch.clamp(torch.sqrt(total_norm), min=1e-10).to(embeds_init)
            if self.args.max_norm > 0:
                exceed_mask = (total_norm > self.args.max_norm).to(embeds_init)
                scaler = self.args.max_norm / total_norm * exceed_mask + (1 - exceed_mask)
                for idx in range(self.args.num_classes):
                    delta_list[idx].data = delta_list[idx].data * scaler.view(-1, 1, 1)
                # pdb.set_trace()

            loss, sample_size, logging_output = criterion(model, sample=sample, token_embeds=token_embeds_list,
                                                          init_dp=self.args.no_sync_dp)
            loss = loss / (1 + self.args.adv_steps)
            optimizer.backward(loss)

            delta_grad_list = [delta.grad.detach().clone() for delta in delta_list]
            total_loss += loss.detach()

        return total_loss, sample_size, logging_output

    def load_dataset(self, split, epoch=0, combine=False, data_path=None, return_only=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def binarize(s, append_bos=False):
            if self.bpe is not None:
                s = self.bpe.encode(s)
            tokens = self.vocab.encode_line(
                s, append_eos=True, add_if_not_exist=False,
            ).long()
            if append_bos and self.args.init_token is not None:
                tokens = torch.cat([tokens.new([self.args.init_token]), tokens])
            return tokens

        if data_path is None:
            data_path = os.path.join(self.args.data, split + '.jsonl')
        if not os.path.exists(data_path):
            raise FileNotFoundError('Cannot find data: {}'.format(data_path))

        src_tokens = [[] for i in range(self.args.num_classes)]
        src_lengths = [[] for i in range(self.args.num_classes)]
        labels = []

        with open(data_path) as h:
            for line in h:
                example = json.loads(line.strip())
                if 'answerKey' in example:
                    label = ord(example['answerKey']) - ord('A')
                    labels.append(label)
                question = example['question']['stem']
                assert len(example['question']['choices']) == self.args.num_classes
                # format: `<s> Q: Where would I not want a fox? </s> A: hen house </s>`
                question = 'Q: ' + question
                question_toks = binarize(question, append_bos=True)
                for i, choice in enumerate(example['question']['choices']):
                    src = 'A: ' + choice['text']
                    src_bin = torch.cat([question_toks, binarize(src)])
                    src_tokens[i].append(src_bin)
                    src_lengths[i].append(len(src_bin))
        assert all(len(src_tokens[0]) == len(src_tokens[i]) for i in range(self.args.num_classes))
        assert len(src_tokens[0]) == len(src_lengths[0])
        assert len(labels) == 0 or len(labels) == len(src_tokens[0])

        for i in range(self.args.num_classes):
            src_lengths[i] = np.array(src_lengths[i])
            src_tokens[i] = ListDataset(src_tokens[i], src_lengths[i])
            src_lengths[i] = ListDataset(src_lengths[i])

        dataset = {
            'id': IdDataset(),
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens[0], reduce=True),
        }

        for i in range(self.args.num_classes):
            dataset.update({
                'net_input{}'.format(i + 1): {
                    'src_tokens': RightPadDataset(
                        src_tokens[i],
                        pad_idx=self.source_dictionary.pad(),
                    ),
                    'src_lengths': src_lengths[i],
                }
            })

        if len(labels) > 0:
            dataset.update({'target': RawLabelDataset(labels)})

        dataset = NestedDictionaryDataset(
            dataset,
            sizes=[np.maximum.reduce([src_token.sizes for src_token in src_tokens])],
        )

        with data_utils.numpy_seed(self.args.seed):
            dataset = SortDataset(
                dataset,
                # shuffle
                sort_order=[np.random.permutation(len(dataset))],
            )

        print('| Loaded {} with {} samples'.format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models
        model = models.build_model(args, self)

        model.register_classification_head(
            'sentence_classification_head',
            num_classes=1,
        )

        return model

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        return self.vocab

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

import numpy as np
import torch
import pdb

from fairseq.data import (
    ConcatSentencesDataset,
    data_utils,
    Dictionary,
    IdDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    OffsetTokensDataset,
    PrependTokenDataset,
    RawLabelDataset,
    RightPadDataset,
    SortDataset,
    StripTokenDataset,
    TruncateDataset,
)

from . import FairseqTask, register_task


@register_task('sentence_prediction')
class SentencePredictionTask(FairseqTask):
    """
    Sentence (or sentence pair) prediction (classification or regression) task.

    Args:
        dictionary (Dictionary): the dictionary for the input of the task
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='FILE',
                            help='file prefix for data')
        parser.add_argument('--num-classes', type=int, default=-1,
                            help='number of classes')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--separator-token', type=int, default=None,
                            help='add separator token between inputs')
        parser.add_argument('--regression-target', action='store_true', default=False)
        parser.add_argument('--no-shuffle', action='store_true', default=False)
        parser.add_argument('--truncate-sequence', action='store_true', default=False,
                            help='Truncate sequence to max_sequence_length')
        parser.add_argument('--adv-lr', type=float, default=0)
        parser.add_argument('--adv-steps', type=int, default=0)
        parser.add_argument('--rand-init-mag', type=float, default=0)
        parser.add_argument('--grad-square', default=False, action="store_true")
        parser.add_argument('--adv-begin-iter', default=-1, type=int)
        parser.add_argument('--no-sync-dp', default=False, action="store_true")
        parser.add_argument('--per-step-adv', default=False, action="store_true")
        parser.add_argument('--std-adv', default=False, action="store_true")
        parser.add_argument('--max-norm', default=-1, type=float)
        parser.add_argument('--yopo-steps', default=1, type=int, help="Only works when > 1")
        parser.add_argument('--norm-method', default="l2", type=str)

    def __init__(self, args, data_dictionary, label_dictionary):
        super().__init__(args)
        self.dictionary = data_dictionary
        self.label_dictionary = label_dictionary

        self.args = args

        # self.args.adv_steps = args.adv_steps
        # self.rand_init_mag = args.rand_init_mag

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
            embeds_init = model.decoder.sentence_encoder.embed_tokens(sample['net_input']['src_tokens'])
            input_mask = (sample['net_input']['src_tokens'] != 1).to(embeds_init)

            if self.args.norm_method == "l2":
                delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
                dims = sample['net_input']['src_lengths'].to(delta) * embeds_init.size(-1)
                mag = self.args.rand_init_mag / torch.sqrt(dims)
                delta = (delta * mag.view(-1, 1, 1)).detach()
            elif self.args.norm_method == "linf":
                delta = torch.zeros_like(embeds_init).uniform_(-self.args.rand_init_mag, self.args.rand_init_mag) * input_mask.unsqueeze(2)

            delta.requires_grad_()
            # pdb.set_trace()
            loss, sample_size, logging_output = criterion(model, sample=sample, token_embed=delta + embeds_init, init_dp=True, clean_logits=None)
        else:
            delta = 0
            loss, sample_size, logging_output = criterion(model, sample, init_dp=True, clean_logits=None)

        if ignore_grad:
            loss *= 0
            pdb.set_trace() # check if this is ever called; if not, remove it directly!

        if num_updates >= self.args.adv_begin_iter:
            loss = loss / (1 + self.args.adv_steps)

        optimizer.backward(loss)
        total_loss += loss.detach()

        if self.args.rand_init_mag > 0 and num_updates >= self.args.adv_begin_iter:
            delta_grad = delta.grad.clone().detach()
        else:
            delta_grad = model.decoder.sentence_encoder.token_embed.grad.clone().detach()

        for t_adv in range(self.args.adv_steps):
            if num_updates < self.args.adv_begin_iter:
                break

            if self.args.norm_method == "l2":
                # doing l2 norm normalization and clipping
                denorm = torch.clamp(torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1), min=1e-10)
                if self.args.grad_square:
                    denorm = denorm ** 2
                delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
                if self.args.max_norm > 0:
                    delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).to(embeds_init).detach()
                    exceed_mask = (delta_norm > self.args.max_norm).to(embeds_init)
                    delta = delta * (self.args.max_norm / delta_norm * exceed_mask + (1-exceed_mask)).view(-1, 1, 1).detach()
            elif self.args.norm_method == "linf":
                denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1, p=float("inf")).view(-1, 1, 1)
                denorm = torch.clamp(denorm, min=1e-8)
                delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
                if self.args.max_norm > 0:
                    delta = torch.clamp(delta, -self.args.max_norm, self.args.max_norm).detach()
            else:
                print("Normalization method {} not defined.".format(self.args.norm_method))

            delta.requires_grad_()
            adv_embeds_init = model.decoder.sentence_encoder.embed_tokens(sample['net_input']['src_tokens'])

            # for the smoothness regularization
            if t_adv == self.args.adv_steps-1 and self.args.reg_weight > 0:
                clean_logits = model(
                    **sample['net_input'],
                    token_embed=None,  # use this to pass token embeddings
                    init_dp=False,
                    features_only=True,
                    classification_head_name='sentence_classification_head',
                    yopo=False
                )[0]

            else:
                clean_logits = None

            loss, sample_size, logging_output = criterion(model, sample=sample, token_embed=delta + adv_embeds_init,
                                                          init_dp=self.args.no_sync_dp, clean_logits=clean_logits)
            loss = loss / (1 + self.args.adv_steps)
            optimizer.backward(loss)
            delta_grad = delta.grad.clone().detach()
            total_loss += loss.detach()

        return total_loss, sample_size, logging_output

    def std_adv_train_step(self, sample, model, criterion, optimizer, ignore_grad=False, num_updates=0):
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
        if num_updates >= self.args.adv_begin_iter:
            embeds_init = model.decoder.sentence_encoder.embed_tokens(sample['net_input']['src_tokens'])
            input_mask = (sample['net_input']['src_tokens'] != 1).to(embeds_init)
            delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
            dims = sample['net_input']['src_lengths'].to(delta) * embeds_init.size(-1)
            mag = self.args.rand_init_mag / torch.sqrt(dims)
            delta = (delta * mag.view(-1, 1, 1)).detach()
            delta.requires_grad_()
            loss, sample_size, logging_output = criterion(model, sample=sample, token_embed=delta + embeds_init, init_dp=True)
        else:
            delta = 0
            loss, sample_size, logging_output = criterion(model, sample, init_dp=True)

        if ignore_grad:
            loss *= 0
            pdb.set_trace() # check if this is ever called; if not, remove it directly!

        for t_adv in range(self.args.adv_steps):
            if num_updates < self.args.adv_begin_iter:
                break

            delta_grad = torch.autograd.grad(loss, delta)[0].detach()
            denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)

            if self.args.grad_square:
                denorm = denorm ** 2
            delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()

            if self.args.max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1).float(), p=2, dim=1).to(embeds_init).detach()
                exceed_mask = (delta_norm > self.args.max_norm).to(embeds_init)
                delta = delta * (self.args.max_norm / delta_norm * exceed_mask + (1-exceed_mask)).view(-1, 1, 1).detach()

            delta.requires_grad_()
            adv_embeds_init = model.decoder.sentence_encoder.embed_tokens(sample['net_input']['src_tokens'])
            loss, sample_size, logging_output = criterion(model, sample=sample, token_embed=delta + adv_embeds_init,
                                                          init_dp=self.args.no_sync_dp)

        optimizer.backward(loss)
        return loss.detach(), sample_size, logging_output

    def yopo_inner_steps(self, p_var, delta, embeds_init, sample, model):
        for yopo_t in range(self.args.yopo_steps):
            hal = model.decoder.sentence_encoder.Hamiltonian_fwd(p_var=p_var, tokens=sample['net_input']['src_tokens'],
                                token_embed=embeds_init + delta, init_dp=False)
            delta_grad = torch.autograd.grad(hal, delta, only_inputs=True, retain_graph=False)[0]

            denorm = torch.clamp(torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1), min=1e-10)

            delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()
            if self.args.max_norm > 0:
                delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).detach()
                exceed_mask = (delta_norm > self.args.max_norm).to(embeds_init)
                delta = delta * (self.args.max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1,
                                                                                                         1).detach()
            delta.requires_grad_()
        return delta


    def yopo_train_step(self, sample, model, criterion, optimizer, ignore_grad=False, num_updates=0):
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
            embeds_init = model.decoder.sentence_encoder.embed_tokens(sample['net_input']['src_tokens'])
            input_mask = (sample['net_input']['src_tokens'] != 1).to(embeds_init)
            delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
            dims = sample['net_input']['src_lengths'].to(delta) * embeds_init.size(-1)
            mag = self.args.rand_init_mag / torch.sqrt(dims)
            delta = (delta * mag.view(-1, 1, 1)).detach()
            delta.requires_grad_()
            # pdb.set_trace()
            loss, sample_size, logging_output = criterion(model, sample=sample, token_embed=delta + embeds_init,
                                                        init_dp=True, yopo=True)
        else:
            delta = 0
            loss, sample_size, logging_output = criterion(model, sample, init_dp=True, yopo=True)
        if ignore_grad:
            loss *= 0
            pdb.set_trace() # check if this is ever called; if not, remove it directly!

        if num_updates >= self.args.adv_begin_iter:
            loss = loss / (1 + self.args.adv_steps)

        optimizer.backward(loss)

        total_loss += loss.detach()

        # if self.args.rand_init_mag > 0 and num_updates >= self.args.adv_begin_iter:
        #     delta_grad = delta.grad.clone().detach()
        # else:
        #     delta_grad = model.decoder.sentence_encoder.token_embed.grad.clone().detach()

        for t_adv in range(self.args.adv_steps):
            p_var = model.decoder.sentence_encoder.first_layer_out.grad.clone().detach()

            const_embeds_init = model.decoder.sentence_encoder.embed_tokens(sample['net_input']['src_tokens']).detach()
            delta = self.yopo_inner_steps(p_var, delta, const_embeds_init, sample, model)

            adv_embeds_init = model.decoder.sentence_encoder.embed_tokens(sample['net_input']['src_tokens'])
            loss, sample_size, logging_output = criterion(model, sample=sample, token_embed=delta + adv_embeds_init,
                                                          init_dp=self.args.no_sync_dp, yopo=True)
            loss = loss / (1 + self.args.adv_steps)
            optimizer.backward(loss)
            total_loss += loss.detach()

        return total_loss, sample_size, logging_output

    def freeat_adv_train_step(self, sample, model, criterion, optimizer, ignore_grad=False, delta=0, init_dp=True):
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
        embeds_init = model.decoder.sentence_encoder.embed_tokens(sample['net_input']['src_tokens'])
        if delta is None:
            input_mask = (sample['net_input']['src_tokens'] != 1).to(embeds_init)
            delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * input_mask.unsqueeze(2)
            dims = sample['net_input']['src_lengths'].to(delta) * embeds_init.size(-1)
            mag = self.args.rand_init_mag / torch.sqrt(dims)
            delta = (delta * mag.view(-1, 1, 1)).detach()

        delta.requires_grad_()

        loss, sample_size, logging_output = criterion(model, sample=sample, token_embed=delta + embeds_init,
                                                      init_dp=init_dp)
        if ignore_grad:
            loss *= 0
            pdb.set_trace() # check if this is ever called; if not, remove it directly!

        loss = loss / (1 + self.args.adv_steps)
        optimizer.backward(loss)

        delta_grad = delta.grad.data
        denorm = torch.norm(delta_grad.view(delta_grad.size(0), -1), dim=1).view(-1, 1, 1)
        if self.args.grad_square:
            denorm = denorm ** 2
        delta = (delta + self.args.adv_lr * delta_grad / denorm).detach()

        if self.args.max_norm > 0:
            # clip the norm of delta
            delta_norm = torch.norm(delta.view(delta.size(0), -1), p=2, dim=1).detach()
            exceed_mask = (delta_norm > self.args.max_norm).to(embeds_init)
            delta = delta * (self.args.max_norm / delta_norm * exceed_mask + (1 - exceed_mask)).view(-1, 1, 1).detach()

        return loss.detach(), sample_size, logging_output, delta

    @classmethod
    def load_dictionary(cls, args, filename, source=True):
        """Load the dictionary from the filename

        Args:
            filename (str): the filename
        """
        dictionary = Dictionary.load(filename)
        dictionary.add_symbol('<mask>')
        return dictionary

    @classmethod
    def setup_task(cls, args, **kwargs):
        assert args.num_classes > 0, 'Must set --num-classes'

        args.tokens_per_sample = args.max_positions

        # load data dictionary
        data_dict = cls.load_dictionary(
            args,
            os.path.join(args.data, 'input0', 'dict.txt'),
            source=True,
        )
        print('| [input] dictionary: {} types'.format(len(data_dict)))

        label_dict = None
        if not args.regression_target:
            # load label dictionary
            label_dict = cls.load_dictionary(
                args,
                os.path.join(args.data, 'label', 'dict.txt'),
                source=False,
            )
            print('| [label] dictionary: {} types'.format(len(label_dict)))
        else:
            label_dict = data_dict
        return SentencePredictionTask(args, data_dict, label_dict)

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split (e.g., train, valid, test)."""
        def get_path(type, split):
            return os.path.join(self.args.data, type, split)

        def make_dataset(type, dictionary):
            split_path = get_path(type, split)

            dataset = data_utils.load_indexed_dataset(
                split_path,
                self.source_dictionary,
                self.args.dataset_impl,
                combine=combine,
            )
            return dataset

        input0 = make_dataset('input0', self.source_dictionary)
        assert input0 is not None, 'could not find dataset: {}'.format(get_path(type, split))
        input1 = make_dataset('input1', self.source_dictionary)

        if self.args.init_token is not None:
            input0 = PrependTokenDataset(input0, self.args.init_token)

        if input1 is None:
            src_tokens = input0
        else:
            if self.args.separator_token is not None:
                input1 = PrependTokenDataset(input1, self.args.separator_token)

            src_tokens = ConcatSentencesDataset(input0, input1)

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(src_tokens))

        if self.args.truncate_sequence:
            src_tokens = TruncateDataset(src_tokens, self.args.max_positions)

        dataset = {
            'id': IdDataset(),
            'net_input': {
                'src_tokens': RightPadDataset(
                    src_tokens,
                    pad_idx=self.source_dictionary.pad(),
                ),
                'src_lengths': NumelDataset(src_tokens, reduce=False),
            },
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(src_tokens, reduce=True),
        }

        if not self.args.regression_target:
            label_dataset = make_dataset('label', self.target_dictionary)
            if label_dataset is not None:
                dataset.update(
                    target=OffsetTokensDataset(
                        StripTokenDataset(
                            label_dataset,
                            id_to_strip=self.target_dictionary.eos(),
                        ),
                        offset=-self.target_dictionary.nspecial,
                    )
                )
        else:
            label_path = "{0}.label".format(get_path('label', split))
            if os.path.exists(label_path):
                dataset.update(
                    target=RawLabelDataset([
                        float(x.strip()) for x in open(label_path).readlines()
                    ])
                )

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[src_tokens.sizes],
        )

        if self.args.no_shuffle:
            dataset = nested_dataset
        else:
            dataset = SortDataset(
                nested_dataset,
                # shuffle
                sort_order=[shuffle],
            )

        print("| Loaded {0} with #samples: {1}".format(split, len(dataset)))

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_model(self, args):
        from fairseq import models


        model = models.build_model(args, self)

        model.register_classification_head(
            'sentence_classification_head',
            num_classes=self.args.num_classes,
        )

        return model

    def max_positions(self):
        return self.args.max_positions

    @property
    def source_dictionary(self):
        return self.dictionary

    @property
    def target_dictionary(self):
        return self.label_dictionary

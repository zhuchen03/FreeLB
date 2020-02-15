# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import (
    data_utils,
    Dictionary,
    encoders,
    IdDataset,
    ListDataset,
    NestedDictionaryDataset,
    NumSamplesDataset,
    NumelDataset,
    SortDataset,
)
from fairseq.tasks import FairseqTask, register_task

from . import wsc_utils
import pdb

@register_task('wsc')
class WSCTask(FairseqTask):
    """Task to finetune RoBERTa for Winograd Schemas."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        parser.add_argument('data', metavar='DIR',
                            help='path to data directory; we load <split>.jsonl')
        parser.add_argument('--init-token', type=int, default=None,
                            help='add token at the beginning of each batch item')
        parser.add_argument('--adv-lr', type=float, default=0)
        parser.add_argument('--adv-steps', type=int, default=0)
        parser.add_argument('--rand-init-mag', type=float, default=0)
        parser.add_argument('--grad-square', default=False, action="store_true")
        parser.add_argument('--adv-begin-iter', default=-1, type=int)
        parser.add_argument('--no-sync-dp', default=False, action="store_true")
        parser.add_argument('--per-step-adv', default=False, action="store_true")
        parser.add_argument('--std-adv', default=False, action="store_true")

    def __init__(self, args, vocab):
        super().__init__(args)
        self.vocab = vocab
        self.mask = vocab.add_symbol('<mask>')

        self.bpe = encoders.build_bpe(args)
        self.tokenizer = encoders.build_tokenizer(args)
        self.args = args

        # hack to handle GPT-2 BPE, which includes leading spaces
        if args.bpe == 'gpt2':
            self.leading_space = True
            self.trailing_space = False
        else:
            self.leading_space = False
            self.trailing_space = True

    def get_init_delta(self, masks, embeds_init):
        # input_mask = (tokens != 1).to(embeds_init)
        delta = torch.zeros_like(embeds_init).uniform_(-1, 1) * masks.unsqueeze(2).to(embeds_init)
        mag = self.args.rand_init_mag / torch.sqrt(torch.sum(masks.to(delta).view(
                            embeds_init.size(0), -1), 1))
        delta = (delta * mag.view(-1, 1, 1)).detach()
        delta.requires_grad_()

        return delta

    def get_masked_input(self, tokens, mask):
        masked_tokens = tokens.clone()
        masked_tokens[mask.bool()] = self.mask
        return masked_tokens

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
        # actually only works for batch_size = 1
        model.train()
        total_loss = 0
        total_logging_output = {}
        total_sample_size = 0
        for n_, (query_tokens, query_masks, candidate_tokens, candidate_masks) in enumerate(zip(sample['query_tokens'],
                                sample['query_masks'], sample['candidate_tokens'], sample['candidate_masks'])):
            single_sample = {}
            single_sample['query_tokens'] = [query_tokens]
            single_sample['query_masks'] = [query_masks]
            single_sample['candidate_tokens'] = [candidate_tokens]
            single_sample['candidate_masks'] = [candidate_masks]
            single_sample['labels'] = [sample['labels'][n_]]
            single_sample['id'] = [sample['id'][n_]]
            single_sample['nsentences'] = sample['nsentences']
            single_sample['ntokens'] = sample['ntokens']

            if self.args.rand_init_mag > 0 and num_updates >= self.args.adv_begin_iter:
                # token_embed_dict = {'query': [], 'candidate': [], 'query_delta': [], 'candidate_delta': []}
                query_embeds_init = model.decoder.sentence_encoder.embed_tokens(
                                self.get_masked_input(query_tokens.unsqueeze(0), query_masks.unsqueeze(0))
                )
                candidate_embeds_init = model.decoder.sentence_encoder.embed_tokens(
                                self.get_masked_input(candidate_tokens, candidate_masks)
                )

                query_delta = self.get_init_delta(query_masks.unsqueeze(0), query_embeds_init)
                candidate_delta = self.get_init_delta(candidate_masks, candidate_embeds_init)

                token_embed_dict = {'query': [query_embeds_init + query_delta],
                                    'candidate': [candidate_embeds_init + candidate_delta]}
                loss, sample_size, logging_output = criterion(model, sample=single_sample,
                                                              token_embed_dict=token_embed_dict,
                                                              init_dp=True)
            else:
                token_embed_dict = {}
                query_delta = 0
                candidate_delta = 0
                loss, sample_size, logging_output = criterion(model, single_sample, init_dp=True)
            if ignore_grad:
                loss *= 0

            if loss.item() == 0:
                # this sample is probably skipped
                continue

            if num_updates >= self.args.adv_begin_iter:
                loss = loss / (1 + self.args.adv_steps)

            for key, val in logging_output.items():
                if key not in total_logging_output:
                    total_logging_output[key] = val / (1 + self.args.adv_steps)
                else:
                    total_logging_output[key] += val / (1 + self.args.adv_steps)

            optimizer.backward(loss)
            total_loss += loss.detach()

            if self.args.rand_init_mag > 0 and num_updates >= self.args.adv_begin_iter:
                query_delta_grad = query_delta.grad.clone().detach() if query_delta.grad is not None else 0
                candidate_delta_grad = candidate_delta.grad.clone().detach() if candidate_delta.grad is not None else 0
            else:
                query_delta_grad = model.decoder.sentence_encoder.token_embed_cache[0].grad
                candidate_delta_grad = model.decoder.sentence_encoder.token_embed_cache[1].grad
                if query_delta_grad is None:
                    # due to the possible by-pass on the negative samples
                    query_delta_grad = 0
                else:
                    query_delta_grad = query_delta_grad.clone().detach()
                if candidate_delta_grad is None:
                    candidate_delta_grad = 0
                else:
                    candidate_delta_grad = candidate_delta_grad.clone().detach()

            for t_adv in range(self.args.adv_steps):
                if num_updates < self.args.adv_begin_iter:
                    break

                # update the embeddings
                denorm = torch.sum(query_delta_grad**2) + torch.sum(candidate_delta_grad**2)
                denorm = max(torch.sqrt(denorm).view(-1, 1, 1).item(), 1e-10)

                query_delta = (query_delta + self.args.adv_lr * query_delta_grad / denorm).detach()
                candidate_delta = (candidate_delta + self.args.adv_lr * candidate_delta_grad / denorm).detach()
                query_delta.requires_grad_()
                candidate_delta.requires_grad_()

                query_embeds_init = model.decoder.sentence_encoder.embed_tokens(
                            self.get_masked_input(query_tokens.unsqueeze(0), query_masks.unsqueeze(0))
                )
                candidate_embeds_init = model.decoder.sentence_encoder.embed_tokens(
                            self.get_masked_input(candidate_tokens, candidate_masks)
                )

                # token_embed_dict = {"query": query_embeds_init + query_delta,
                #                     "candidate": candidate_embeds_init + candidate_delta}
                token_embed_dict['query'] = [query_embeds_init + query_delta]
                token_embed_dict['candidate'] = [candidate_embeds_init + candidate_delta]
                loss, sample_size, logging_output = criterion(model, sample=single_sample, token_embed_dict=token_embed_dict,
                                                              init_dp=self.args.no_sync_dp)
                loss = loss / (1 + self.args.adv_steps)
                optimizer.backward(loss)

                query_delta_grad = query_delta.grad.clone().detach() if query_delta.grad is not None else 0
                candidate_delta = candidate_delta.grad.clone().detach() if candidate_delta.grad is not None else 0

                total_loss += loss.detach()

                for key, val in logging_output.items():
                    total_logging_output[key] += val / (1 + self.args.adv_steps)

            total_sample_size += sample_size

        total_logging_output['ntokens'] = sample['ntokens']
        total_logging_output['nsentences'] = sample['nsentences']
        return total_loss, total_sample_size, total_logging_output

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
        assert args.criterion == 'wsc', 'Must set --criterion=wsc'

        # load data and label dictionaries
        vocab = cls.load_dictionary(os.path.join(args.data, 'dict.txt'))
        print('| dictionary: {} types'.format(len(vocab)))

        return cls(args, vocab)

    def load_dataset(self, split, epoch=0, combine=False, data_path=None, return_only=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """

        def binarize(s: str, append_eos: bool = False):
            if self.tokenizer is not None:
                s = self.tokenizer.encode(s)
            if self.bpe is not None:
                s = self.bpe.encode(s)
            tokens = self.vocab.encode_line(
                s, append_eos=append_eos, add_if_not_exist=False,
            ).long()
            if self.args.init_token is not None:
                tokens = torch.cat([tokens.new([self.args.init_token]), tokens])
            return tokens

        if data_path is None:
            data_path = os.path.join(self.args.data, split + '.jsonl')
        if not os.path.exists(data_path):
            raise FileNotFoundError('Cannot find data: {}'.format(data_path))

        query_tokens = []
        query_masks = []
        query_lengths = []
        candidate_tokens = []
        candidate_masks = []
        candidate_lengths = []
        labels = []

        for sentence, pronoun_span, query, label in wsc_utils.jsonl_iterator(data_path):
            prefix = sentence[:pronoun_span.start].text
            suffix = sentence[pronoun_span.end:].text_with_ws

            # spaCy spans include trailing spaces, but we need to know about
            # leading spaces for the GPT-2 BPE
            leading_space = ' ' if sentence[:pronoun_span.start].text_with_ws.endswith(' ') else ''
            trailing_space = ' ' if pronoun_span.text_with_ws.endswith(' ') else ''

            # get noun phrases, excluding pronouns and anything overlapping with the query
            cand_spans = wsc_utils.filter_noun_chunks(
                wsc_utils.extended_noun_chunks(sentence),
                exclude_pronouns=True,
                exclude_query=query,
                exact_match=False,
            )

            def binarize_with_mask(txt):
                toks = binarize(
                    prefix + leading_space + txt + trailing_space + suffix,
                    append_eos=True,
                )
                mask = torch.zeros_like(toks, dtype=torch.uint8)
                mask_start = len(binarize(prefix))
                mask_size = len(binarize(leading_space + txt))
                mask[mask_start:mask_start + mask_size] = 1
                return toks, mask

            if query is not None:
                query_toks, query_mask = binarize_with_mask(query)
                query_len = len(query_toks)
            else:
                query_toks, query_mask, query_len = None, None, 0

            query_tokens.append(query_toks)
            query_masks.append(query_mask)
            query_lengths.append(query_len)
            cand_toks, cand_masks = [], []
            for cand_span in cand_spans:
                toks, mask = binarize_with_mask(cand_span.text)
                cand_toks.append(toks)
                cand_masks.append(mask)

            # collate candidates
            cand_toks = data_utils.collate_tokens(cand_toks, pad_idx=self.vocab.pad())
            cand_masks = data_utils.collate_tokens(cand_masks, pad_idx=0)
            assert cand_toks.size() == cand_masks.size()

            candidate_tokens.append(cand_toks)
            candidate_masks.append(cand_masks)
            candidate_lengths.append(cand_toks.size(1))

            labels.append(label)

        query_lengths = np.array(query_lengths)
        query_tokens = ListDataset(query_tokens, query_lengths)
        query_masks = ListDataset(query_masks, query_lengths)

        candidate_lengths = np.array(candidate_lengths)
        candidate_tokens = ListDataset(candidate_tokens, candidate_lengths)
        candidate_masks = ListDataset(candidate_masks, candidate_lengths)

        labels = ListDataset(labels, [1]*len(labels))

        dataset = {
            'id': IdDataset(),
            'query_tokens': query_tokens,
            'query_masks': query_masks,
            'candidate_tokens': candidate_tokens,
            'candidate_masks': candidate_masks,
            'labels': labels,
            'nsentences': NumSamplesDataset(),
            'ntokens': NumelDataset(query_tokens, reduce=True),
        }

        nested_dataset = NestedDictionaryDataset(
            dataset,
            sizes=[query_lengths],
        )

        with data_utils.numpy_seed(self.args.seed):
            shuffle = np.random.permutation(len(query_tokens))
        dataset = SortDataset(
            nested_dataset,
            # shuffle
            sort_order=[shuffle],
        )

        if return_only:
            return dataset

        self.datasets[split] = dataset
        return self.datasets[split]

    def build_dataset_for_inference(self, sample_json):
        with tempfile.NamedTemporaryFile(buffering=0) as h:
            h.write((json.dumps(sample_json) + '\n').encode('utf-8'))
            dataset = self.load_dataset(
                'disambiguate_pronoun',
                data_path=h.name,
                return_only=True,
            )
        return dataset

    def disambiguate_pronoun(self, model, sentence, use_cuda=False):
        sample_json = wsc_utils.convert_sentence_to_json(sentence)
        dataset = self.build_dataset_for_inference(sample_json)
        sample = dataset.collater([dataset[0]])
        if use_cuda:
            sample = utils.move_to_cuda(sample)

        def get_masked_input(tokens, mask):
            masked_tokens = tokens.clone()
            masked_tokens[mask] = self.mask
            return masked_tokens

        def get_lprobs(tokens, mask):
            logits, _ = model(src_tokens=get_masked_input(tokens, mask))
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float)
            scores = lprobs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
            mask = mask.type_as(scores)
            scores = (scores * mask).sum(dim=-1) / mask.sum(dim=-1)
            return scores

        cand_lprobs = get_lprobs(
            sample['candidate_tokens'][0],
            sample['candidate_masks'][0].bool(),
        )
        if sample['query_tokens'][0] is not None:
            query_lprobs = get_lprobs(
                sample['query_tokens'][0].unsqueeze(0),
                sample['query_masks'][0].bool().unsqueeze(0),
            )
            return (query_lprobs >= cand_lprobs).all().item() == 1
        else:
            best_idx = cand_lprobs.argmax().item()
            full_cand = sample['candidate_tokens'][0][best_idx]
            mask = sample['candidate_masks'][0][best_idx]
            toks = full_cand[mask]
            return self.bpe.decode(self.source_dictionary.string(toks)).strip()

    def disambiguate_pronoun_scores(self, model, sentence, use_cuda=False):
        sample_json = wsc_utils.convert_sentence_to_json(sentence)
        dataset = self.build_dataset_for_inference(sample_json)
        sample = dataset.collater([dataset[0]])
        if use_cuda:
            sample = utils.move_to_cuda(sample)

        def get_masked_input(tokens, mask):
            masked_tokens = tokens.clone()
            masked_tokens[mask] = self.mask
            return masked_tokens

        def get_lprobs(tokens, mask):
            logits, _ = model(src_tokens=get_masked_input(tokens, mask))
            lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float)
            scores = lprobs.gather(2, tokens.unsqueeze(-1)).squeeze(-1)
            mask = mask.type_as(scores)
            scores = (scores * mask).sum(dim=-1) / mask.sum(dim=-1)
            return scores

        cand_lprobs = get_lprobs(
            sample['candidate_tokens'][0],
            sample['candidate_masks'][0].bool(),
        )
        if sample['query_tokens'][0] is not None:
            query_lprobs = get_lprobs(
                sample['query_tokens'][0].unsqueeze(0),
                sample['query_masks'][0].unsqueeze(0).bool(),
            )
            return query_lprobs, cand_lprobs
        else:
            best_idx = cand_lprobs.argmax().item()
            full_cand = sample['candidate_tokens'][0][best_idx]
            mask = sample['candidate_masks'][0][best_idx].bool()
            toks = full_cand[mask]
            return self.bpe.decode(self.source_dictionary.string(toks)).strip()

    @property
    def source_dictionary(self):
        return self.vocab

    @property
    def target_dictionary(self):
        return self.vocab

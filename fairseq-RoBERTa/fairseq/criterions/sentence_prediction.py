# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F

from fairseq import utils

from . import FairseqCriterion, register_criterion
import pdb

@register_criterion('sentence_prediction')
class SentencePredictionCriterion(FairseqCriterion):

    @staticmethod
    def add_args(parser):
        # fmt: off
        parser.add_argument('--save-predictions', metavar='FILE',
                            help='file to save predictions to')
        parser.add_argument('--reduction', default="sum", type=str)
        parser.add_argument('--reg-weight', default=0, type=float,
                            help='weight for the smoothness regularization')
        # fmt: on

    def sym_kld(self, net_out_1, net_out_2):
        P = F.softmax(net_out_1, dim=-1, dtype=torch.float32)
        Q = F.softmax(net_out_2, dim=-1, dtype=torch.float32)

        logP = F.log_softmax(net_out_1, dim=-1, dtype=torch.float32)
        logQ = F.log_softmax(net_out_2, dim=-1, dtype=torch.float32)

        # taking sum directly, since the reduction method is sum
        sym_kld = 0.5 * torch.sum((P - Q) * (logP - logQ))

        return sym_kld

    def forward(self, model, sample, token_embed=None, init_dp=True, reduce=True, yopo=False, clean_logits=None):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        assert hasattr(model, 'classification_heads') and \
            'sentence_classification_head' in model.classification_heads, \
            "model must provide sentence classification head for --criterion=sentence_prediction"

        logits, _ = model(
            **sample['net_input'],
            token_embed=token_embed, # use this to pass token embeddings
            init_dp=init_dp,
            features_only=True,
            classification_head_name='sentence_classification_head',
            yopo=yopo
        )

        targets = model.get_targets(sample, [logits]).view(-1)
        sample_size = targets.numel()

        if not self.args.regression_target:
            loss = F.nll_loss(
                F.log_softmax(logits, dim=-1, dtype=torch.float32),
                targets,
                reduction="sum",
            )
            reg_val = self.sym_kld(clean_logits, logits) if clean_logits is not None else 0

        else:
            logits = logits.squeeze().float()
            targets = targets.float()
            loss = F.mse_loss(
                logits,
                targets,
                reduction="sum",
            )
            # use l2 loss for the regression task
            reg_val = 0.5 * torch.sum((logits - clean_logits) ** 2) if clean_logits is not None else 0

        if self.args.reg_weight > 0:
            loss = loss + self.args.reg_weight * reg_val

        if self.args.reduction == "mean":
            loss = loss / self.args.max_sentences / self.args.update_freq[0]
        # pdb.set_trace()
        logging_output = {
            'loss': utils.item(loss.data) if reduce else loss.data,
            'ntokens': sample['ntokens'],
            'nsentences': sample_size,
            'sample_size': sample_size,
            'labels': targets,
            'smooth_reg': utils.item(reg_val)
        }

        if not self.args.regression_target:
            preds = logits.max(dim=1)[1]
            logging_output.update(
                ncorrect=(preds == targets).sum().item()
            )
            # keep the prediction of the model
            logging_output.update(preds=preds.detach())
        else:
            logging_output.update(preds=logits.view(-1).detach())

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        nsentences = sum(log.get('nsentences', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        smooth_reg = sum(log.get('smooth_reg', 0) for log in logging_outputs)

        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
            'ntokens': ntokens,
            'nsentences': nsentences,
            'sample_size': sample_size,
            'labels': logging_outputs[0]['labels'],
            'preds': logging_outputs[0]['preds'],
            'smooth_reg': smooth_reg / sample_size
        }

        if len(logging_outputs) > 0 and 'ncorrect' in logging_outputs[0]:
            ncorrect = sum(log.get('ncorrect', 0) for log in logging_outputs)
            agg_output.update(accuracy=ncorrect/nsentences) # gets the accuracy!!!

        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output

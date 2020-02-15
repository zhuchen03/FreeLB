# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
)


class TransformerSentenceEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        activation_fn: str = 'relu',
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        export: bool = False,
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.self_attn = MultiheadAttention(
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=True
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = LayerNorm(self.embedding_dim, export=export)
        self.fc1 = nn.Linear(self.embedding_dim, ffn_embedding_dim)
        self.fc2 = nn.Linear(ffn_embedding_dim, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = LayerNorm(self.embedding_dim, export=export)

    def forward(
        self,
        x: torch.Tensor,
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,
        init_dp: bool = True,
        store_dp_list: bool = False,
        dp_idx: int = 0,
        need_attn: bool = False,
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        residual = x
        x, attn = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=self_attn_padding_mask,
            need_weights=need_attn,
            attn_mask=self_attn_mask,
            init_dp=init_dp,
            store_dp_list=store_dp_list, dp_idx=dp_idx
        )
        # if not init_dp:
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        if self.training and self.dropout > 0:
            if store_dp_list:
                # for WSC, which use two forward passes to compute the loss
                if init_dp:
                    mask = torch.zeros_like(x).bernoulli_(1 - self.dropout) / (1 - self.dropout)
                    if dp_idx == 0:
                        self.mask_list1 = [mask]
                    else:
                        self.mask_list1.append(mask)
                x = self.mask_list1[dp_idx] * x
            else:
                if init_dp:
                    self.mask1 = torch.zeros_like(x).bernoulli_(1 - self.dropout) / (1 - self.dropout)
                x = self.mask1 * x

        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        # if not ignore_dp:
        #     x = F.dropout(x, p=self.activation_dropout, training=self.training)
        if self.training and self.activation_dropout > 0:
            # if init_dp:
            #     self.mask2 = torch.zeros_like(x).bernoulli_(1 - self.activation_dropout) / (1 - self.activation_dropout)
            # x = self.mask2 * x
            if store_dp_list:
                # for WSC, which use two forward passes to compute the loss
                if init_dp:
                    mask = torch.zeros_like(x).bernoulli_(1 - self.activation_dropout) / (1 - self.activation_dropout)
                    if dp_idx == 0:
                        self.mask_list2 = [mask]
                    else:
                        self.mask_list2.append(mask)
                x = self.mask_list2[dp_idx] * x
            else:
                if init_dp:
                    self.mask2 = torch.zeros_like(x).bernoulli_(1 - self.activation_dropout) / (1 - self.activation_dropout)
                x = self.mask2 * x

        x = self.fc2(x)
        # if not ignore_dp:
        #     x = F.dropout(x, p=self.dropout, training=self.training)

        if self.training and self.dropout > 0:
            if store_dp_list:
                if init_dp:
                    mask = torch.zeros_like(x).bernoulli_(1 - self.dropout) / (1 - self.dropout)
                    if dp_idx == 0:
                        self.mask_list3 = [mask]
                    else:
                        self.mask_list3.append(mask)
                x = self.mask_list3[dp_idx] * x
            else:
                if init_dp:
                    self.mask3 = torch.zeros_like(x).bernoulli_(1 - self.dropout) / (1 - self.dropout)
                x = self.mask3 * x

        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn

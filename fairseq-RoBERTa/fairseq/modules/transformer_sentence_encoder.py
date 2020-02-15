# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq.modules import (
    LayerNorm,
    MultiheadAttention,
    PositionalEmbedding,
    TransformerSentenceEncoderLayer,
)
import pdb

def init_bert_params(module):
    """
    Initialize the weights specific to the BERT Model.
    This overrides the default initializations depending on the specified arguments.
        1. If normal_init_linear_weights is set then weights of linear
           layer will be initialized using the normal distribution and
           bais will be set to the specified value.
        2. If normal_init_embed_weights is set then weights of embedding
           layer will be initialized using the normal distribution.
        3. If normal_init_proj_weights is set then weights of
           in_project_weight for MultiHeadAttention initialized using
           the normal distribution (to be validated).
    """

    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02)
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)
        module.weight.data[module.padding_idx].zero_()
    if isinstance(module, MultiheadAttention):
        module.in_proj_weight.data.normal_(mean=0.0, std=0.02)


class TransformerSentenceEncoder(nn.Module):
    """
    Implementation for a Bi-directional Transformer based Sentence Encoder used
    in BERT/XLM style pre-trained models.

    This first computes the token embedding using the token embedding matrix,
    position embeddings (if specified) and segment embeddings
    (if specified). After applying the specified number of
    TransformerEncoderLayers, it outputs all the internal states of the
    encoder as well as the final representation associated with the first
    token (usually CLS token).

    Input:
        - tokens: B x T matrix representing sentences
        - segment_labels: B x T matrix representing segment label for tokens

    Output:
        - a tuple of the following:
            - a list of internal model states used to compute the
              predictions where each tensor has shape B x T x C
            - sentence representation associated with first input token
              in format B x C.
    """

    def __init__(
        self,
        padding_idx: int,
        vocab_size: int,
        num_encoder_layers: int = 6,
        embedding_dim: int = 768,
        ffn_embedding_dim: int = 3072,
        num_attention_heads: int = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        max_seq_len: int = 256,
        num_segments: int = 2,
        use_position_embeddings: bool = True,
        offset_positions_by_padding: bool = True,
        encoder_normalize_before: bool = False,
        apply_bert_init: bool = False,
        activation_fn: str = "relu",
        learned_pos_embedding: bool = True,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        embed_scale: float = None,
        freeze_embeddings: bool = False,
        n_trans_layers_to_freeze: int = 0,
        export: bool = False,
    ) -> None:

        super().__init__()
        self.padding_idx = padding_idx
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.max_seq_len = max_seq_len
        self.embedding_dim = embedding_dim
        self.num_segments = num_segments
        self.use_position_embeddings = use_position_embeddings
        self.apply_bert_init = apply_bert_init
        self.learned_pos_embedding = learned_pos_embedding

        self.embed_tokens = nn.Embedding(
            self.vocab_size, self.embedding_dim, self.padding_idx
        )
        self.embed_scale = embed_scale

        self.segment_embeddings = (
            nn.Embedding(self.num_segments, self.embedding_dim, padding_idx=None)
            if self.num_segments > 0
            else None
        )

        self.embed_positions = (
            PositionalEmbedding(
                self.max_seq_len,
                self.embedding_dim,
                padding_idx=(self.padding_idx if offset_positions_by_padding else None),
                learned=self.learned_pos_embedding,
            )
            if self.use_position_embeddings
            else None
        )

        self.layers = nn.ModuleList(
            [
                TransformerSentenceEncoderLayer(
                    embedding_dim=self.embedding_dim,
                    ffn_embedding_dim=ffn_embedding_dim,
                    num_attention_heads=num_attention_heads,
                    dropout=self.dropout,
                    attention_dropout=attention_dropout,
                    activation_dropout=activation_dropout,
                    activation_fn=activation_fn,
                    add_bias_kv=add_bias_kv,
                    add_zero_attn=add_zero_attn,
                    export=export,
                )
                for _ in range(num_encoder_layers)
            ]
        )

        if encoder_normalize_before:
            self.emb_layer_norm = LayerNorm(self.embedding_dim, export=export)
        else:
            self.emb_layer_norm = None

        # Apply initialization of model params after building the model
        if self.apply_bert_init:
            self.apply(init_bert_params)

        def freeze_module_params(m):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad = False

        if freeze_embeddings:
            freeze_module_params(self.embed_tokens)
            freeze_module_params(self.segment_embeddings)
            freeze_module_params(self.embed_positions)
            freeze_module_params(self.emb_layer_norm)

        for layer in range(n_trans_layers_to_freeze):
            freeze_module_params(self.layers[layer])

    def Hamiltonian_fwd(self,
        p_var: torch.Tensor,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        positions: Optional[torch.Tensor] = None,
        token_embed: Optional[torch.Tensor] = None,
        init_dp: Optional[bool] = True,
        store_embed: Optional[bool] = False, # doesn't matter if token_embed is None
        dp_idx: Optional[int] = 0,
        ) -> Tuple[torch.Tensor, torch.Tensor]:

        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        if token_embed is None:
            # pdb.set_trace()
            if store_embed:
                token_embed = self.embed_tokens(tokens)
                token_embed.requires_grad_()
                token_embed.retain_grad()
                # pdb.set_trace()
                if dp_idx == 0:
                    self.token_embed_cache = [token_embed]
                else:
                    self.token_embed_cache.append(token_embed)
                x = token_embed
            else:
                self.token_embed = self.embed_tokens(tokens)
                self.token_embed.requires_grad_()
                self.token_embed.retain_grad()
                x = self.token_embed
        else:
            x = token_embed

        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        if self.training and self.dropout > 0:
            if store_embed:
                # we are in the WSC mode
                if init_dp:
                    if dp_idx == 0:
                        self.mask_list1 = [torch.zeros_like(x).bernoulli_(1 - self.dropout) / (1 - self.dropout)]
                    else:
                        self.mask_list1.append(torch.zeros_like(x).bernoulli_(1 - self.dropout) / (1 - self.dropout))
                x = self.mask_list1[dp_idx] * x
            else:
                if init_dp:
                    self.mask1 = torch.zeros_like(x).bernoulli_(1 - self.dropout) / (1 - self.dropout)
                x = self.mask1 * x

        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        x, _ = self.layers[0](x, self_attn_padding_mask=padding_mask, init_dp=init_dp, store_dp_list=store_embed,
                        dp_idx=dp_idx, need_attn=False)
        return torch.sum(x * p_var)


    def forward(
        self,
        tokens: torch.Tensor,
        segment_labels: torch.Tensor = None,
        last_state_only: bool = False,
        positions: Optional[torch.Tensor] = None,
        token_embed: Optional[torch.Tensor] = None,
        init_dp: Optional[bool] = True,
        store_embed: Optional[bool] = False, # doesn't matter if token_embed is None
        dp_idx: Optional[int] = 0,
        need_attn: Optional[bool] = False,
        yopo: Optional[bool] = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # compute padding mask. This is needed for multi-head attention
        padding_mask = tokens.eq(self.padding_idx)
        if not padding_mask.any():
            padding_mask = None

        if token_embed is None:
            # pdb.set_trace()
            if store_embed:
                token_embed = self.embed_tokens(tokens)
                token_embed.requires_grad_()
                token_embed.retain_grad()
                # pdb.set_trace()
                if dp_idx == 0:
                    self.token_embed_cache = [token_embed]
                else:
                    self.token_embed_cache.append(token_embed)
                x = token_embed
            else:
                self.token_embed = self.embed_tokens(tokens)
                self.token_embed.requires_grad_()
                self.token_embed.retain_grad()
                x = self.token_embed
        else:
            x = token_embed

        if self.embed_scale is not None:
            x *= self.embed_scale

        if self.embed_positions is not None:
            x += self.embed_positions(tokens, positions=positions)

        if self.segment_embeddings is not None and segment_labels is not None:
            x += self.segment_embeddings(segment_labels)

        if self.emb_layer_norm is not None:
            x = self.emb_layer_norm(x)

        if self.training and self.dropout > 0:
            if store_embed:
                # we are in the WSC mode
                if init_dp:
                    if dp_idx == 0:
                        self.mask_list1 = [torch.zeros_like(x).bernoulli_(1 - self.dropout) / (1 - self.dropout)]
                    else:
                        self.mask_list1.append(torch.zeros_like(x).bernoulli_(1 - self.dropout) / (1 - self.dropout))
                x = self.mask_list1[dp_idx] * x
            else:
                if init_dp:
                    self.mask1 = torch.zeros_like(x).bernoulli_(1 - self.dropout) / (1 - self.dropout)
                x = self.mask1 * x



        # account for padding while computing the representation
        if padding_mask is not None:
            x *= 1 - padding_mask.unsqueeze(-1).type_as(x)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        inner_states = []
        if not last_state_only:
            inner_states.append(x)

        attn_list = []
        for natt, layer in enumerate(self.layers):
            x, attn = layer(x, self_attn_padding_mask=padding_mask, init_dp=init_dp, store_dp_list=store_embed,
                            dp_idx=dp_idx, need_attn=need_attn)
            if natt == 0 and yopo:
                self.first_layer_out = x
                self.first_layer_out.requires_grad_()
                self.first_layer_out.retain_grad()
                x = self.first_layer_out
            if need_attn:
                attn_list.append(attn.detach())
            if not last_state_only:
                inner_states.append(x)

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)

        sentence_rep = x[:, 0, :]

        if last_state_only:
            inner_states = [x]

        if need_attn:
            return inner_states, sentence_rep, attn_list
        else:
            return inner_states, sentence_rep

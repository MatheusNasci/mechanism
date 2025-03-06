import os
import warnings
import copy
import math
import numpy as np
import torch
from torch import nn

import esm
from torchdrug import core, layers, utils, data, models
from torchdrug.utils import comm, cuda, cat
from torchdrug.layers import functional
from torchdrug.core import Registry as R

from module import layers as newlayers


@R.register("models.MoleculeTransformer")
class MoleculeTransformer(nn.Module, core.Configurable):
    """
    Transformer Encoder-Decoder.
    
    Parameters:
        num_layers (int): number of transformer layers
        embed_dim (int): hidden dimension
        ffn_embed_dim (int): hidden dimension of feed-forward layers
        attention_heads (int): number of attention heads
        output_dim (int): output dimension
        dropout (float): dropout rate
    """
    # 这个部分还需要改，tokenize的部分，我还没仔细看
    residue_symbol2id = {
    "C": 1, "O": 2, "N": 3, "H": 4, "c": 5, "n": 6, 
    "(": 7, ")": 8, "=": 9, "#": 10, "[": 11, "]": 12, 
    "1": 13, "2": 14, "3": 15, "4": 16, "5": 17, 
    "p": 18, "e": 19, "m": 20, "c": 21
    }

    id2residue_symbol = {v: k for k, v in residue_symbol2id.items()}

    def __init__(self, num_layers=8, embed_dim=512, ffn_embed_dim=2048, attention_heads=8, output_dim=512, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.output_dim = output_dim
        
        self.cls_idx = self.residue_symbol2id["c"]
        self.padding_idx = self.residue_symbol2id["p"]
        self.eos_idx = self.residue_symbol2id["e"]
        self.mask_idx = self.residue_symbol2id["m"]
        self.alphabet_size = len(self.residue_symbol2id)

        self.dropout_module = nn.Dropout(dropout)

        # Embedding
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.embed_dim, padding_idx=self.padding_idx
        )
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.embed_dim ** -0.5)
        nn.init.constant_(self.embed_tokens.weight[self.padding_idx], 0)
        self.embed_scale = math.sqrt(self.embed_dim)

        # Position Embedding
        self.embed_positions = newlayers.SinusoidalPositionalEmbedding(
            self.embed_dim, self.padding_idx,
        )

        # Transformer Encoder
        self.encoder_layers = nn.ModuleList(
            [
                newlayers.TransformerEncoderLayer(
                    self.embed_dim,
                    self.ffn_embed_dim,
                    self.attention_heads,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # Transformer Decoder
        self.decoder_layers = nn.ModuleList(
            [
                newlayers.TransformerDecoderLayer(
                    self.embed_dim,
                    self.ffn_embed_dim,
                    self.attention_heads,
                    no_encoder_attn=False,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(embed_dim)

        # Output projection
        self.output_projection = nn.Linear(
            self.embed_dim, self.alphabet_size, bias=False
        )
        nn.init.normal_(
            self.output_projection.weight, mean=0, std=self.embed_dim ** -0.5
        )

    def forward(self, graph, src, tgt, all_loss=None, metric=None, incremental_state=None):
        """
        Parameters:
            graph (Graph): input graph which is not used in this function
            src (Tensor): SMILES input (batch_size, src_len)
            tgt (Tensor): target SMILES (batch_size, tgt_len)
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
            incremental_state (dict): used for fast sampling during inference
        
        Returns:
            dict with the following fields:
                logits
        """
        
        bs, src_len = src.size()
        _, tgt_len = tgt.size()

        # === ENCODER ===
        src_padding_mask = src.eq(self.padding_idx)
        src_positions = self.embed_positions(src)

        src_emb = self.embed_scale * self.embed_tokens(src) + src_positions
        src_emb = self.dropout_module(src_emb).transpose(0, 1)  # (src_len, batch, embed_dim)

        for layer in self.encoder_layers:
            src_emb = layer(src_emb, self_attn_padding_mask=src_padding_mask)[0]

        encoder_out = self.encoder_norm(src_emb)

        # === DECODER ===
        tgt_padding_mask = tgt.eq(self.padding_idx)
        tgt_positions = self.embed_positions(tgt)

        tgt_emb = self.embed_scale * self.embed_tokens(tgt) + tgt_positions
        tgt_emb = self.dropout_module(tgt_emb).transpose(0, 1)  # (tgt_len, batch, embed_dim)

        for layer in self.decoder_layers:
            tgt_emb, _, _ = layer(
                tgt_emb,
                encoder_out=encoder_out,
                encoder_padding_mask=src_padding_mask,
                self_attn_padding_mask=tgt_padding_mask,
                need_attn=False,
                need_head_weights=False,
            )

        decoder_out = self.decoder_norm(tgt_emb)

        logits = self.output_projection(decoder_out).transpose(0, 1)  # (batch, tgt_len, vocab_size)
        
        return {"logits": logits}

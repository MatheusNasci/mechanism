import os
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchdrug import core
from torchdrug.core import Registry as R
from torchdrug.layers import functional

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from module import layers as newlayers
from module.data.molecule_sequence import MoleculeSequence


@R.register("models.MoleculeTransformer")
class MoleculeTransformer(nn.Module, core.Configurable):
    """
    Transformer Encoder-Decoder for molecule sequence modeling.
    
    Parameters:
        num_layers (int): number of transformer layers
        embed_dim (int): hidden dimension
        ffn_embed_dim (int): hidden dimension of feed-forward layers
        attention_heads (int): number of attention heads
        output_dim (int): output dimension
        dropout (float): dropout rate
    """

    token2id = MoleculeSequence.token2id
    id2token = {v: k for k, v in token2id.items()}

    def __init__(self, num_layers=8, embed_dim=512, ffn_embed_dim=2048, attention_heads=8, output_dim=512, dropout=0.1):
        super().__init__()
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.ffn_embed_dim = ffn_embed_dim
        self.attention_heads = attention_heads
        self.output_dim = output_dim
        
        self.cls_idx = self.token2id["CLS"]
        self.padding_idx = self.token2id["PAD"]
        self.eos_idx = self.token2id["EOS"]
        self.mask_idx = self.token2id["MASK"]
        self.alphabet_size = len(self.token2id)

        self.dropout_module = nn.Dropout(dropout)

        # **Token Embedding**
        self.embed_tokens = nn.Embedding(
            self.alphabet_size, self.embed_dim, padding_idx=self.padding_idx
        )
        nn.init.normal_(self.embed_tokens.weight, mean=0, std=self.embed_dim ** -0.5)
        nn.init.constant_(self.embed_tokens.weight[self.padding_idx], 0)
        self.embed_scale = math.sqrt(self.embed_dim)

        # **Position Embedding**
        self.embed_positions = newlayers.SinusoidalPositionalEmbedding(
            self.embed_dim, self.padding_idx,
        )

        # **Transformer Encoder**
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

        # **Transformer Decoder**
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

        # **Output Projection**
        self.output_projection = nn.Linear(
            self.embed_dim, self.alphabet_size, bias=False
        )
        nn.init.normal_(
            self.output_projection.weight, mean=0, std=self.embed_dim ** -0.5
        )

    def forward(self, graph, src, tgt, all_loss=None, metric=None, incremental_state=None):
        """
        Parameters:
            graph (Graph): input graph (not used)
            src (Tensor): input molecule sequence (batch_size, src_len)
            tgt (Tensor): target molecule sequence (batch_size, tgt_len)
            all_loss (Tensor, optional): if specified, add loss to this tensor
            metric (dict, optional): if specified, output metrics to this dict
            incremental_state (dict): used for fast sampling during inference
        
        Returns:
            dict with the following fields:
                logits
        """

        bs, src_len = src.size()
        _, tgt_len = tgt.size()

        if src.dim() == 1 or src.shape[1] != src_len:
            src_size = torch.tensor([len(seq) for seq in src], device=src.device)  
            src, _ = functional.variadic_to_padded(src, src_size, value=self.padding_idx)

        if tgt.dim() == 1 or tgt.shape[1] != tgt_len:
            tgt_size = torch.tensor([len(seq) for seq in tgt], device=tgt.device)
            tgt, _ = functional.variadic_to_padded(tgt, tgt_size, value=self.padding_idx)

        # === **ENCODER** ===
        src_padding_mask = src.eq(self.padding_idx) 
        src_positions = self.embed_positions(src)

        src_emb = self.embed_scale * self.embed_tokens(src) + src_positions
        src_emb = self.dropout_module(src_emb).transpose(0, 1)  # (src_len, batch, embed_dim)

        for layer in self.encoder_layers:
            src_emb = layer(src_emb, self_attn_padding_mask=src_padding_mask)[0]

        encoder_out = self.encoder_norm(src_emb)

        # === **DECODER** ===
        tgt_padding_mask = tgt.eq(self.padding_idx)
        tgt_positions = self.embed_positions(tgt)

        tgt_emb = self.embed_scale * self.embed_tokens(tgt) + tgt_positions
        tgt_emb = self.dropout_module(tgt_emb).transpose(0, 1) 

        if incremental_state is None:
            self_attn_mask = torch.triu(
                torch.ones(tgt_len, tgt_len, device=tgt.device) * float("-inf"), diagonal=1
            )
        else:
            self_attn_mask = None

        for layer in self.decoder_layers:
            tgt_emb, _, _ = layer(
                tgt_emb,
                encoder_out=encoder_out,
                encoder_padding_mask=src_padding_mask,
                self_attn_padding_mask=tgt_padding_mask,
                self_attn_mask=self_attn_mask,
                need_attn=False,
                need_head_weights=False,
            )

        decoder_out = self.decoder_norm(tgt_emb)

        logits = self.output_projection(decoder_out).transpose(0, 1)  # (batch, tgt_len, vocab_size)
        
        return {"logits": logits}

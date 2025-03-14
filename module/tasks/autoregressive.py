import os, sys
import math
import numpy as np
import copy
from collections import defaultdict

import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter_mean

from torchdrug import core, data, layers, tasks, metrics, models
from torchdrug.core import Registry as R
from torchdrug.layers import functional
from torchdrug.utils import cuda, download
from torchdrug.data import constant


@R.register("tasks.MoleculeGenerationTask")
class MoleculeGenerationTask(tasks.Task, core.Configurable):
    """
    Autoregressive Transformer Task for Molecule-to-Molecule generation.

    This task models molecule translation from input (A) to output (B).

    Parameters:
        model (nn.Module): The Transformer model (MoleculeTransformer)
    """

    def __init__(self, model, criterion="ce", metric=["acc"], num_mlp_layer=2, **kwargs):
        super(MoleculeGenerationTask, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer

    def get_target(self, batch):
        """
        Extracts the target sequences (B column) from batch and applies EOS + padding.
        """
        graph = batch["target"]["B"] 
        target = graph.node_feature 
        
        size_ext = graph.num_nodes
        
        if len(size_ext.shape) == 0:  
            size_ext = size_ext.unsqueeze(0)

        if (size_ext == 0).any():
            print("Warning: Some sequences have zero length!")

        eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.model.eos_idx 

        target, size_ext = functional._extend(target, size_ext, eos, torch.ones_like(size_ext))  

        target = functional.variadic_to_padded(target, size_ext, value=self.model.padding_idx)[0]

        assert target.numel() > 0, "Error: target is empty after variadic_to_padded()!"

        return target

    def predict_and_target(self, batch, all_loss=None, metric=None):
        """
        Compute model predictions and target sequences for loss computation.
        """
        graph = batch["target"]["B"] 
        
        input = functional.variadic_to_padded(graph.node_feature, graph.num_nodes, value=self.model.padding_idx)[0]
        
        target = self.get_target(batch)  

        output = self.model(graph, input, target, all_loss, metric) 
        pred = output["logits"]

        mask = (target != self.model.padding_idx).view(-1)
        pred = pred.reshape(-1, pred.size(-1))[mask]
        target = target.reshape(-1)[mask]

        return pred, target

    def evaluate(self, pred, target):
        """
        Compute accuracy for evaluation.

        Parameters:
            pred (Tensor): Model predictions.
            target (Tensor): Ground truth token indices.

        Returns:
            metric (dict): Computed accuracy metric.
        """
        metric = {}
        accuracy = (pred.argmax(dim=-1) == target).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        """
        Compute loss and evaluation metrics for training.

        Parameters:
            batch (dict): Batch data containing input and target SMILES.

        Returns:
            all_loss (Tensor): Computed loss.
            metric (dict): Evaluation metrics.
        """
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.cross_entropy(pred, target, ignore_index=self.model.padding_idx)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss
        return all_loss, metric

    def score(self, batch):
        """
        Compute log-probabilities of target sequences.

        Parameters:
            batch (dict): Batch data containing input and target SMILES.

        Returns:
            log_prob (Tensor): Log-probabilities of sequences.
            token_count (Tensor): Number of non-PAD tokens.
        """
        src = batch["input"].token_ids
        tgt = self.get_target(batch)

        src, src_size = functional.variadic_to_padded(src, value=self.model.padding_idx)
        tgt, tgt_size = functional.variadic_to_padded(tgt, value=self.model.padding_idx)

        output = self.model(graph=None, src=src, tgt=tgt)
        pred = output["logits"]

        log_prob = torch.log_softmax(pred, dim=-1)
        log_prob = torch.gather(log_prob, dim=-1, index=tgt.unsqueeze(-1)).squeeze(-1)
        mask = (tgt != self.model.padding_idx)
        log_prob = (log_prob * mask).sum(-1)

        return log_prob, mask.sum(-1)

    def seq_to_id(self, seq):
        """
        Convert SMILES sequence to token ID sequence.

        Parameters:
            seq (str): SMILES string.

        Returns:
            List[int]: Token IDs.
        """
        return [self.model.token2id.get(c, self.model.mask_idx) for c in seq]

    def sample(self, partial_seqs, max_length=50, temperature=1.0):
        """
        Sample sequences using autoregressive decoding.

        Parameters:
            partial_seqs (List[str]): Input SMILES sequences (prefix).
            max_length (int): Maximum length for generation.
            temperature (float): Sampling temperature.

        Returns:
            List[str]: Generated SMILES sequences.
        """
        assert len(set([len(seq) for seq in partial_seqs])) == 1

        sampled_tokens = [self.seq_to_id(seq) + [self.model.mask_idx] * (max_length - len(seq)) for seq in partial_seqs]
        sampled_tokens = torch.tensor(sampled_tokens, dtype=torch.long, device=self.device)

        incremental_state = {}
        for i in range(max_length):
            if sampled_tokens[0, i].item() != self.model.mask_idx:
                continue
            output = self.model(graph=None, src=sampled_tokens[:, :i], tgt=sampled_tokens[:, :i], incremental_state=incremental_state)
            logits = output["logits"].squeeze(1) / temperature
            probs = torch.softmax(logits, dim=-1)
            sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)

        sampled_seq = sampled_tokens.tolist()
        final_seqs = []
        for seq in sampled_seq:
            smi_seq = ''.join([self.model.id2token[aa] for aa in seq])
            smi_seq = smi_seq.split(self.model.id2token[self.model.eos_idx])[0]  
            final_seqs.append(smi_seq)

        return final_seqs
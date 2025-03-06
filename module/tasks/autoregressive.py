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


@R.register("tasks.AutoregressiveLanguageModel")
class AutoregressiveLanguageModel(tasks.Task, core.Configurable):
    """
    Autoregressive Language Model for protein language model pretraining.
    
    Parameters:
        model: the model
    """
    def __init__(self, model):
        super(AutoregressiveLanguageModel, self).__init__()
        self.model = model

    def get_target(self, graph):
        target = graph.residue_type
        size_ext = graph.num_residues
        eos = torch.ones(graph.batch_size, dtype=torch.long, device=self.device) * self.model.eos_idx
        target, size_ext = functional._extend(target, size_ext, eos, torch.ones_like(size_ext))
        target = functional.variadic_to_padded(target, size_ext, value=self.model.padding_idx)[0]
        return target
    
    def predict_and_target(self, batch, all_loss=None, metric=None):
        # Get the graph.
        graph = batch["graph"]
        # Get the input by padding graph.residue_type.
        input = functional.variadic_to_padded(graph.residue_type, graph.num_residues, value=self.model.padding_idx)[0]
        # Get the target.
        target = self.get_target(graph)
        
        output = self.model(graph, input, all_loss, metric)
        pred = output["logits"]

        mask = (target != self.model.padding_idx).view(-1)
        pred = pred.reshape(-1, pred.size(-1))[mask]
        target = target.reshape(-1)[mask]

        return pred, target
    
    def evaluate(self, pred, target):
        metric = {}
        accuracy = (pred.argmax(dim=-1) == target).float().mean()

        name = tasks._get_metric_name("acc")
        metric[name] = accuracy

        return metric

    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        pred, target = self.predict_and_target(batch, all_loss, metric)
        metric.update(self.evaluate(pred, target))

        loss = F.cross_entropy(pred, target)
        name = tasks._get_criterion_name("ce")
        metric[name] = loss

        all_loss += loss

        return all_loss, metric
    
    def score(self, batch):
        graph = batch["graph"]
        input = functional.variadic_to_padded(graph.residue_type, graph.num_residues, value=self.model.padding_idx)[0]
        target = self.get_target(graph)

        output = self.model(graph, input)
        pred = output["logits"]

        log_prob = torch.log_softmax(pred, dim=-1)
        log_prob = torch.gather(log_prob, dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
        mask = (target != self.model.padding_idx)
        log_prob = (log_prob * mask).sum(-1)

        return log_prob, mask.sum(-1)
    
    def seq_to_id(self, seq):
        return [self.model.residue_symbol2id[c] for c in seq]
    
    def sample(self, partial_seqs, max_length, temperature=1.0):
        """
        Sample sequences.
        
        Parameters:
            partial_seqs (List of Str): a set of prefix and the lengths should be the same
            max_length (int): maximum length
            temperature (float): temperature of sampling
        Returns:
            List of Str, generated sequences
        """
        assert len(set([len(seq) for seq in partial_seqs])) == 1

        sampled_tokens = [self.seq_to_id(seq) + [self.model.mask_idx] * (max_length - len(seq)) for seq in partial_seqs]
        sampled_tokens = torch.tensor(sampled_tokens, dtype=torch.long, device=self.device)
        # Save incremental states for faster sampling
        incremental_state = dict()
        # Decode one token at a time
        for i in range(max_length):
            if sampled_tokens[0, i].item() != self.model.mask_idx:
                continue
            output = self.model(
                graph=None, 
                input=sampled_tokens[:, :i], 
                incremental_state=incremental_state,
            )
            logits = output["logits"].squeeze(1)
            logits /= temperature
            probs = torch.softmax(logits, dim=-1)
            sampled_tokens[:, i] = torch.multinomial(probs, 1).squeeze(-1)
        
        sampled_seq = sampled_tokens.tolist()

        final_seqs = []
        for seq in sampled_seq:
            aa_seq = ''.join([self.model.id2residue_symbol[aa] for aa in seq])
            aa_seq = aa_seq.split("e")[0]
            final_seqs += [aa_seq]
        return final_seqs

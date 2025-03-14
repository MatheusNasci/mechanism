import os
import sys
import pickle
from collections import defaultdict

import torch
from torch import nn
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from torchdrug.data import constant, Graph, PackedGraph
from torchdrug.core import Registry as R
from torchdrug.utils import pretty

# 读取 vocab.pkl
_vocab_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "vocab.pkl")
with open(_vocab_path, "rb") as fin:
    _vocab = pickle.load(fin)

_token2id = _vocab
_id2token = {v: k for k, v in _token2id.items()}


@R.register("data.MoleculeSequence")
class MoleculeSequence(Graph):
    """
    Representation of a single SMILES sequence.

    This class treats a SMILES string as a token sequence and converts it into an ID sequence 
    using a pre-built vocabulary (module/data/vocab.pkl).
    
    At the same time, the token sequence is considered as a simple linear graph: 
    each token is a node, and there is an edge between token i and i+1 (relation type 0).

    Attributes:
        token_ids (Tensor): 1D tensor storing the token IDs in the SMILES sequence.
        num_tokens (int): Sequence length (number of tokens).
    """

    token2id = _token2id
    id2token = _id2token

    def __init__(self, token_ids, **kwargs):
        """
        Parameters:
            token_ids (list[int]): A token ID sequence that can be converted to a 1D LongTensor.
        """
        token_ids = torch.as_tensor(token_ids, dtype=torch.long)
        self.token_ids = token_ids
        self.num_tokens = token_ids.size(0)

        # **构造线性 `Graph` 结构**
        if self.num_tokens > 1:
            nodes = torch.arange(self.num_tokens, dtype=torch.long)
            src = nodes[:-1]
            tgt = nodes[1:]
            edge_list = torch.stack([src, tgt, torch.zeros(src.size(0), dtype=torch.long)], dim=1)
        else:
            edge_list = torch.zeros((0, 3), dtype=torch.long)

        # **继承 `Graph`**
        super(MoleculeSequence, self).__init__(
            edge_list=edge_list, num_node=self.num_tokens, num_relation=1, **kwargs
        )

        # **直接存储 `sequence`，而不是 `context`**
        self.sequence = token_ids  # ✅ 直接存储，不使用 `context`

    def sequence(self):
        return self.context("sequence")

    @classmethod
    def tokenize(cls, smiles):
        """
        Tokenize a SMILES string.
        
        If the string contains spaces, split by space; otherwise, split by character.
        
        Parameters:
            smiles (str): SMILES string.

        Returns:
            list[str]: Tokenized SMILES sequence.
        """
        if " " in smiles:
            tokens = smiles.split()
        else:
            tokens = list(smiles)
        return tokens

    @classmethod
    def from_smiles(cls, smiles):
        """
        Convert a SMILES string to a `MoleculeSequence` object.

        Parameters:
            smiles (str): SMILES string.

        Returns:
            MoleculeSequence: Converted SMILES sequence.
        """
        tokens = cls.tokenize(smiles)
        token_ids = [cls.token2id.get(token, cls.token2id.get("UNK", 0)) for token in tokens]
        return cls(token_ids)

    def to_graph(self):
        """
        Convert `MoleculeSequence` to `Graph`.
        """
        return Graph(
            edge_list=self.edge_list, num_node=self.num_tokens, num_relation=1,
            node_feature=self.token_ids  
        )




    def __repr__(self):
        fields = ["num_tokens=%d" % self.num_tokens]
        return "MoleculeSequence(%s)" % ", ".join(fields)


@R.register("data.PackedMoleculeSequence")
class PackedMoleculeSequence(PackedGraph):
    """
    A packed container for variable-length SMILES sequences.

    This is used for batch processing multiple `MoleculeSequence` objects.
    
    Attributes:
        token_ids (Tensor): A 1D tensor containing all concatenated token IDs.
        num_tokens (Tensor): A tensor containing the length of each sequence.
        num_edges (Tensor): A tensor containing the number of edges in each sequence.
        num_nodes (Tensor): A tensor containing the number of nodes in each sequence.
    """

    unpacked_type = Graph

    def __init__(self, edge_list=None, num_tokens=None, num_edges=None, num_nodes=None, token_ids=None, **kwargs):
        """
        Parameters:
            edge_list (Tensor): The concatenated edge list.
            num_tokens (Tensor): Number of tokens per sequence.
            num_edges (Tensor): Number of edges per sequence.
            num_nodes (Tensor): Number of nodes per sequence.
            token_ids (Tensor): Concatenated token IDs of all sequences.
        """
        if num_tokens is None or num_edges is None or num_nodes is None or token_ids is None:
            raise ValueError("All attributes `num_tokens`, `num_edges`, `num_nodes`, and `token_ids` must be provided")

        self.num_tokens = torch.as_tensor(num_tokens, dtype=torch.long)
        self.num_edges = torch.as_tensor(num_edges, dtype=torch.long)
        self.num_nodes = torch.as_tensor(num_nodes, dtype=torch.long)
        self.token_ids = torch.as_tensor(token_ids, dtype=torch.long)

        if edge_list is None:
            edge_list = torch.zeros((0, 3), dtype=torch.long)

        super(PackedMoleculeSequence, self).__init__(
            edge_list=edge_list, num_nodes=self.num_nodes.sum().item(), num_edges=self.num_edges.sum().item(), **kwargs
        )

    @classmethod
    def from_smiles(cls, smiles_list):
        """
        Convert a list of SMILES strings into a `PackedMoleculeSequence`.

        Parameters:
            smiles_list (list[str]): List of SMILES strings.

        Returns:
            PackedMoleculeSequence: Packed batch of sequences.
        """
        sequences = [MoleculeSequence.from_smiles(smiles) for smiles in smiles_list]
        return cls.pack(sequences)

    @classmethod
    def pack(cls, sequences):
        """
        Pack multiple `MoleculeSequence` objects into a single `PackedMoleculeSequence`.

        Parameters:
            sequences (list[MoleculeSequence]): List of molecule sequences.

        Returns:
            PackedMoleculeSequence: Packed batch of sequences.
        """
        num_tokens = [seq.num_tokens for seq in sequences]
        token_ids = torch.cat([seq.token_ids for seq in sequences], dim=0)

        edge_list_list = []
        num_edges = []
        num_nodes = []
        offset = 0

        for seq in sequences:
            n = seq.num_tokens
            num_nodes.append(n)
            if n > 1:
                nodes = torch.arange(n, dtype=torch.long) + offset
                src = nodes[:-1]
                tgt = nodes[1:]
                local_edges = torch.stack([src, tgt, torch.zeros(n - 1, dtype=torch.long)], dim=1)
                edge_list_list.append(local_edges)
                num_edges.append(n - 1)
            else:
                num_edges.append(0)
            offset += n

        edge_list = torch.cat(edge_list_list, dim=0) if edge_list_list else torch.zeros((0, 3), dtype=torch.long)

        return cls(
            edge_list=edge_list,
            num_tokens=num_tokens,
            num_edges=num_edges,
            num_nodes=num_nodes,
            token_ids=token_ids,
        )

    def __repr__(self):
        return f"PackedMoleculeSequence(batch_size={self.batch_size})"

    @property
    def batch_size(self):
        return self.num_tokens.size(0)

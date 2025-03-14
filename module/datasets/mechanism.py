import os
import sys
import csv
import warnings
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.utils import data as torch_data
from torchdrug.core import Registry as R
from torchdrug import core, utils
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from module.data import molecule_sequence

@R.register("datasets.MechanismDataset")
class MechanismDataset(torch_data.Dataset, core.Configurable):
    """
    Dataset for mechanism-based molecular transformations.
    This dataset is used for SMILES-to-SMILES transformation tasks, where both input and target are SMILES sequences.
    """

    target_fields = ["B"]
    splits = ["train", "valid", "test"]

    def __init__(self, path, target_fields=None, verbose=1, **kwargs):
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            os.makedirs(path)
        self.path = path
        if target_fields is not None:
            self.target_fields = target_fields

        csv_file = "/home/user/students/gyt/major_product/internal_dataset/trans_new.csv"
        self.load_csv(csv_file, smiles_field="A", target_fields=self.target_fields, verbose=verbose, **kwargs)

    def __len__(self):
        return len(self.data)

    def load_smiles(self, input_smiles, target_smiles, sample_splits, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Load the dataset from input SMILES and target SMILES.

        Parameters:
            input_smiles (list of str): Input SMILES sequences.
            target_smiles (dict of list of str): Dictionary mapping target field names to lists of target SMILES.
            sample_splits (list of str): Dataset split labels for each sample.
            transform (Callable, optional): Data transformation function.
            lazy (bool, optional): If True, molecules are processed in the dataloader (reducing memory usage but slowing data loading).
            verbose (int, optional): Output verbosity level.
            **kwargs: Additional arguments for data processing.
        """
        num_sample = len(input_smiles)
        if num_sample > 1000000:
            warnings.warn("Preprocessing a large dataset may consume significant CPU memory and time. "
                          "Use load_smiles(lazy=True) to construct molecule sequences in the dataloader instead.")

        # Ensure the target list sizes match input samples
        for field, target_list in target_smiles.items():
            if len(target_list) != num_sample:
                raise ValueError(f"Number of targets `{field}` does not match number of molecule sequences. "
                                 f"Expected {num_sample}, but found {len(target_list)}.")

        self.transform = transform
        self.lazy = lazy
        self.kwargs = kwargs
        self.input_smiles_list = []
        self.target_smiles_dict = defaultdict(list)
        self.data = []
        self.sample_splits = sample_splits

        if verbose:
            input_smiles = list(tqdm(input_smiles, desc="Processing input SMILES"))
            for field in target_smiles:
                target_smiles[field] = list(tqdm(target_smiles[field], desc=f"Processing target SMILES for {field}"))

        for i, smiles in enumerate(input_smiles):
            input_molecule = molecule_sequence.MoleculeSequence.from_smiles(smiles)  # Tokenize SMILES before conversion
            self.input_smiles_list.append(input_molecule)
            self.data.append(input_molecule)
            if sample_splits:
                self.sample_splits.append(sample_splits[i])

            for field, target_list in target_smiles.items():
                target_molecule = molecule_sequence.MoleculeSequence.from_smiles(target_list[i])  # Tokenize target
                self.target_smiles_dict[field].append(target_molecule)

    def load_csv(self, csv_file, smiles_field, target_fields, transform=None, lazy=False, verbose=0, **kwargs):
        """
        Load dataset from a CSV file.
        This method correctly parses `train, valid, test` columns which contain binary values (`0/1`).
        """
        input_smiles = []
        target_smiles = {field: [] for field in target_fields}
        sample_splits_list = []

        with open(csv_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                input_smiles.append(row[smiles_field])
                for field in target_fields:
                    target_smiles[field].append(row[field])

                if row["train"] == "1":
                    sample_splits_list.append("train")
                elif row["valid"] == "1":
                    sample_splits_list.append("valid")
                elif row["test"] == "1":
                    sample_splits_list.append("test")
                else:
                    print(f"Warning: Sample `{row}` does not belong to any split. Assigning to `train` by default.")
                    sample_splits_list.append("train")

        print(f"Loaded {len(input_smiles)} samples, Sample Splits count: {len(sample_splits_list)}")

        self.load_smiles(input_smiles, target_smiles, sample_splits_list, transform, lazy, verbose, **kwargs)


    def split(self):
        """
        Get dataset splits for training, validation, and testing.

        Returns:
            List of torch_data.Subset: [train_subset, valid_subset, test_subset]
        """
        if self.sample_splits is None:
            return [torch_data.Subset(self, list(range(len(self.data))))]

        train_indices = [i for i in range(len(self.data)) if self.sample_splits[i] == "train"]
        valid_indices = [i for i in range(len(self.data)) if self.sample_splits[i] == "valid"]
        test_indices = [i for i in range(len(self.data)) if self.sample_splits[i] == "test"]

        return [
            torch_data.Subset(self, train_indices),
            torch_data.Subset(self, valid_indices),
            torch_data.Subset(self, test_indices)
        ]

    def get_item(self, index):
        """
        Get a sample from the dataset.

        Parameters:
            index (int): Sample index.

        Returns:
            dict: {"input": Graph, "target": {field_name: Graph, ...}}
        """
        input_molecule = self.input_smiles_list[index]
        target_molecules = {field: self.target_smiles_dict[field][index] for field in self.target_fields}

        input_graph = input_molecule.to_graph()
        target_graphs = {field: target_molecules[field].to_graph() for field in self.target_fields}

        item = {"input": input_graph, "target": target_graphs}

        if self.transform:
            item = self.transform(item)

        return item

    def __getitem__(self, index):
        """
        Retrieve a dataset sample.

        Parameters:
            index (int or list or slice): Index of the sample.

        Returns:
            dict: A single sample if index is an int, otherwise a list of samples.
        """
        if isinstance(index, int):
            return self.get_item(index)

        index = self._standardize_index(index, len(self))
        return [self.get_item(i) for i in index]

    def _standardize_index(self, index, count):
        """
        Convert index formats (slice, list) into a standardized list.

        Parameters:
            index (slice, list, or int): Indexing format.
            count (int): Total number of samples.

        Returns:
            list: Standardized list of indices.
        """
        if isinstance(index, slice):
            start = index.start or 0
            if start < 0:
                start += count
            stop = index.stop or count
            if stop < 0:
                stop += count
            step = index.step or 1
            return list(range(start, stop, step))
        elif isinstance(index, list):
            return index
        else:
            raise ValueError(f"Unknown index format: {index}")

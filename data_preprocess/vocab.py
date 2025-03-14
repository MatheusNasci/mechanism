import pandas as pd
import re
import pickle
from rdkit import Chem

def smi_tokenizer(smi: str):
    """Tokenize a SMILES molecule or reaction."""
    pattern = r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
    regex = re.compile(pattern)
    tokens = regex.findall(smi)
    assert smi == "".join(tokens), f"Tokenization error: {smi}"
    return tokens

def canonicalize_smiles(smiles: str, remove_atom_number: bool = True):
    """Convert a SMILES string to its canonical form using RDKit."""
    smiles = "".join(smiles.split()) 
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""  

    if remove_atom_number:
        for atom in mol.GetAtoms():
            atom.ClearProp("molAtomMapNumber") 

    cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)

    mol = Chem.MolFromSmiles(cano_smiles)
    if mol:
        cano_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    
    return cano_smiles

def build_vocab(df, col_A="A", col_B="B"):
    vocab = set()

    for smiles in pd.concat([df[col_A], df[col_B]]).dropna():  
        cano_smiles = canonicalize_smiles(smiles)  
        if cano_smiles:  
            tokens = smi_tokenizer(cano_smiles)  
            vocab.update(tokens)  

    special_tokens = ["PAD", "CLS", "EOS", "MASK"]
    vocab_dict = {token: idx for idx, token in enumerate(special_tokens)}

    for token in sorted(vocab):
        vocab_dict[token] = len(vocab_dict)

    return vocab_dict

if __name__ == "__main__":
    csv_path = "internal_dataset/trans_new.csv"
    df = pd.read_csv(csv_path)

    vocab_dict = build_vocab(df)

    pkl_path = "module/data/vocab.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(vocab_dict, f)

    print(vocab_dict)

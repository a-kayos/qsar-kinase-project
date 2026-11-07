import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    """Convert a SMILES string into a binary Morgan fingerprint numpy array."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((0,), dtype=np.float32)
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr

def prepare_data(csv_path):
    """Load dataset and extract features (X) and targets (y)."""
    df = pd.read_csv(csv_path)
    # Assumes columns are named 'smiles' and 'pIC50'
    X = np.array([smiles_to_fingerprint(s) for s in df['smiles']])
    y = df['pIC50'].values
    return X, y

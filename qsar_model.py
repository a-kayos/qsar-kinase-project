{\rtf1\ansi\ansicpg1252\cocoartf2909
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fswiss\fcharset0 Helvetica;}
{\colortbl;\red255\green255\blue255;}
{\*\expandedcolortbl;;}
\paperw11900\paperh16840\margl1440\margr1440\vieww11520\viewh8400\viewkind0
\pard\tx720\tx1440\tx2160\tx2880\tx3600\tx4320\tx5040\tx5760\tx6480\tx7200\tx7920\tx8640\pardirnatural\partightenfactor0

\f0\fs24 \cf0 import pandas as pd\
import numpy as np\
from rdkit import Chem\
from rdkit.Chem import AllChem\
\
def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):\
    """Convert a SMILES string into a binary Morgan fingerprint numpy array."""\
    mol = Chem.MolFromSmiles(smiles)\
    if mol is None:\
        return np.zeros(n_bits)\
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)\
    arr = np.zeros((0,), dtype=np.float32)\
    AllChem.DataStructs.ConvertToNumpyArray(fp, arr)\
    return arr\
\
def prepare_data(csv_path):\
    """Load dataset and extract features (X) and targets (y)."""\
    df = pd.read_csv(csv_path)\
    # Assumes columns are named 'smiles' and 'pIC50'\
    X = np.array([smiles_to_fingerprint(s) for s in df['smiles']])\
    y = df['pIC50'].values\
    return X, y}
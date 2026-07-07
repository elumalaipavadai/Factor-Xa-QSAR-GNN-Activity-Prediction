"""
featurize.py — molecular featurization.

VALIDATED with RDKit 2026.03 (modern, non-deprecated MorganGenerator API).
Two featurizations:
  * morgan_fp()    -> 2048-bit ECFP4-style fingerprint (for RF/XGB/LightGBM baselines)
  * descriptors()  -> 8 physicochemical descriptors (optional baseline block)
Both return None on an unparseable SMILES so you can drop bad rows cleanly.
"""
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator, Descriptors, DataStructs

# Build the Morgan generator ONCE at import (fast, and avoids the deprecated
# GetMorganFingerprintAsBitVect call that warns on RDKit 2023+).
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def morgan_fp(smiles: str, n_bits: int = 2048):
    """Return an int8 numpy array of length n_bits, or None if SMILES is invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = _MORGAN.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# A compact, interpretable descriptor block. Add/remove as you like.
_DESC_FUNCS = [
    ("MolWt", Descriptors.MolWt),
    ("MolLogP", Descriptors.MolLogP),
    ("TPSA", Descriptors.TPSA),
    ("NumHDonors", Descriptors.NumHDonors),
    ("NumHAcceptors", Descriptors.NumHAcceptors),
    ("NumRotatableBonds", Descriptors.NumRotatableBonds),
    ("FractionCSP3", Descriptors.FractionCSP3),
    ("NumAromaticRings", Descriptors.NumAromaticRings),
]
DESC_NAMES = [n for n, _ in _DESC_FUNCS]


def descriptors(smiles: str):
    """Return a float numpy array of the descriptor block, or None if invalid."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array([f(mol) for _, f in _DESC_FUNCS], dtype=float)

"""
filters.py — synthesis-aware + drug-likeness filtering for generated molecules.

Validated with RDKit 2026.03. Covers the three things that gate generated output:
  * synthetic accessibility (SA score, RDKit contrib sascorer)
  * structural alerts (PAINS)
  * drug-like property windows (MW, logP, TPSA, HBD, HBA, QED)
"""
import os, sys
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, RDConfig, FilterCatalog

# --- SA score lives in RDKit's contrib dir, not the main namespace (common gotcha) ---
sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
import sascorer  # noqa: E402  (must come after the path append)

# --- PAINS catalog, built once ---
_p = FilterCatalog.FilterCatalogParams()
_p.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
_PAINS = FilterCatalog.FilterCatalog(_p)

# Drug-like windows (Lipinski/Veber-ish; tune to your program)
WINDOWS = dict(MolWt=(150, 550), MolLogP=(-1, 5), TPSA=(0, 140),
               NumHDonors=(0, 5), NumHAcceptors=(0, 10))


def sa_score(mol):
    """1 (easy) .. 10 (hard). Common cutoff: keep < 4."""
    return sascorer.calculateScore(mol)


def has_pains(mol):
    return _PAINS.HasMatch(mol)


def passes(smiles, sa_max=4.0, qed_min=0.5):
    """Return (ok: bool, info: dict) for a SMILES against all filters."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False, {"reason": "unparseable"}
    info = {
        "SA": sa_score(mol),
        "QED": QED.qed(mol),
        "PAINS": has_pains(mol),
        "MolWt": Descriptors.MolWt(mol),
        "MolLogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "NumHDonors": Descriptors.NumHDonors(mol),
        "NumHAcceptors": Descriptors.NumHAcceptors(mol),
    }
    ok = (info["SA"] < sa_max and info["QED"] >= qed_min and not info["PAINS"])
    for k, (lo, hi) in WINDOWS.items():
        ok = ok and (lo <= info[k] <= hi)
    return bool(ok), info

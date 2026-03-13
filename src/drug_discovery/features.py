"""
Shared molecular feature engineering utilities for the Drug Discovery portfolio.

Functions here mirror the inline helpers used in each Jupyter notebook so that
they can be tested independently (pytest) and re-used without copy-pasting.
"""

from __future__ import annotations

from collections import defaultdict
from typing import List, Optional, Tuple

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors

try:
    from rdkit.Chem import QED
except ImportError:  # older RDKit builds use different path
    from rdkit.Chem.QED import qed as _qed  # type: ignore

    class _QEDShim:
        @staticmethod
        def qed(mol):
            return _qed(mol)

    QED = _QEDShim()  # type: ignore

# ── Constants ─────────────────────────────────────────────────────────────────
DESC_NAMES: List[str] = [
    "MW",
    "LogP",
    "TPSA",
    "HBD",
    "HBA",
    "RotBonds",
    "ArRings",
    "RadElec",
]

DESC_NAMES_9: List[str] = DESC_NAMES + ["NumRings"]


# ── Fingerprints ──────────────────────────────────────────────────────────────

def smiles_to_morgan(
    smiles: str,
    radius: int = 2,
    n_bits: int = 2048,
) -> np.ndarray:
    """Convert a SMILES string to a Morgan fingerprint bit vector.

    Parameters
    ----------
    smiles:
        SMILES string for the molecule.
    radius:
        Morgan algorithm radius (2 ≡ ECFP4).
    n_bits:
        Length of the bit vector.

    Returns
    -------
    np.ndarray of shape (n_bits,), dtype float64.
    Returns an all-zero vector for invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
    arr = np.zeros(n_bits)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


# ── Descriptors ───────────────────────────────────────────────────────────────

def smiles_to_descriptors(smiles: str) -> List[float]:
    """Compute 8 physicochemical RDKit descriptors for a SMILES string.

    Returns
    -------
    List of 8 values: [MW, LogP, TPSA, HBD, HBA, RotBonds, ArRings, RadElec].
    Returns a list of 8 NaN values for invalid SMILES.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return [float("nan")] * 8
    return [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        rdMolDescriptors.CalcNumHBD(mol),
        rdMolDescriptors.CalcNumHBA(mol),
        rdMolDescriptors.CalcNumRotatableBonds(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        Descriptors.NumRadicalElectrons(mol),
    ]


def compute_features(
    smiles_list: List[str],
    n_bits: int = 2048,
    radius: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Morgan fingerprints + 9 descriptors for a list of SMILES.

    Parameters
    ----------
    smiles_list:
        List of SMILES strings.
    n_bits:
        Fingerprint length.
    radius:
        Morgan radius.

    Returns
    -------
    fps : np.ndarray, shape (n, n_bits)  — binary fingerprint matrix.
    descs : np.ndarray, shape (n, 9)     — descriptor matrix.
    """
    fps: List[np.ndarray] = []
    descs: List[List[float]] = []

    for sm in smiles_list:
        mol = Chem.MolFromSmiles(sm)
        if mol is None:
            fps.append(np.zeros(n_bits))
            descs.append([float("nan")] * 9)
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=n_bits)
        arr = np.zeros(n_bits)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)
        descs.append(
            [
                Descriptors.MolWt(mol),
                Descriptors.MolLogP(mol),
                Descriptors.TPSA(mol),
                rdMolDescriptors.CalcNumHBD(mol),
                rdMolDescriptors.CalcNumHBA(mol),
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                rdMolDescriptors.CalcNumAromaticRings(mol),
                Descriptors.NumRadicalElectrons(mol),
                rdMolDescriptors.CalcNumRings(mol),
            ]
        )

    if not fps:
        return np.empty((0, n_bits)), np.empty((0, 9))

    return np.array(fps), np.array(descs)


# ── Drug-likelihood filters ────────────────────────────────────────────────────

def lipinski_filter(smiles: str) -> bool:
    """Apply Lipinski's Rule of Five filter.

    Returns True only when all four criteria are satisfied:
    MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10.
    Returns False for invalid or empty SMILES.
    """
    if not smiles or not smiles.strip():
        return False
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mw   = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd  = rdMolDescriptors.CalcNumHBD(mol)
    hba  = rdMolDescriptors.CalcNumHBA(mol)
    return mw <= 500 and logp <= 5 and hbd <= 5 and hba <= 10


def get_mol_properties(smiles: str) -> Optional[dict]:
    """Return a dictionary of physicochemical properties for a molecule.

    Includes: SMILES, MW, LogP, TPSA, HBD, HBA, RotBonds, ArRings, QED, NumRings.
    Returns None for invalid or empty SMILES.
    """
    if not smiles or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "SMILES":   smiles,
        "MW":       round(Descriptors.MolWt(mol), 2),
        "LogP":     round(Descriptors.MolLogP(mol), 3),
        "TPSA":     round(Descriptors.TPSA(mol), 2),
        "HBD":      rdMolDescriptors.CalcNumHBD(mol),
        "HBA":      rdMolDescriptors.CalcNumHBA(mol),
        "RotBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "ArRings":  rdMolDescriptors.CalcNumAromaticRings(mol),
        "QED":      round(QED.qed(mol), 4),
        "NumRings": rdMolDescriptors.CalcNumRings(mol),
    }


# ── Data splitting ────────────────────────────────────────────────────────────

def scaffold_split(
    smiles_list: List[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    """Scaffold-aware train/test split using Bemis–Murcko scaffolds.

    Molecules sharing the same Murcko scaffold always go to the same split,
    preventing data leakage when testing generalisation to new scaffolds.

    Parameters
    ----------
    smiles_list:
        Input SMILES strings (in any order).
    test_size:
        Fraction of indices to reserve for the test set.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    train_idx, test_idx : sorted lists of integer indices.
    """
    from rdkit.Chem.Scaffolds import MurckoScaffold

    scaffolds: defaultdict = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            scaffold_smi = ""
        else:
            core = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(core) if core is not None else ""
        scaffolds[scaffold_smi].append(i)

    rng = np.random.default_rng(seed)
    scaffold_groups = list(scaffolds.values())
    rng.shuffle(scaffold_groups)

    n_test = max(1, int(len(smiles_list) * test_size))
    train_idx: List[int] = []
    test_idx: List[int] = []

    for group in scaffold_groups:
        if len(test_idx) < n_test:
            test_idx.extend(group)
        else:
            train_idx.extend(group)

    return sorted(train_idx), sorted(test_idx)

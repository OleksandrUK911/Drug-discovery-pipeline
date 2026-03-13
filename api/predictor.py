"""
Model registry and inference functions for the Drug Discovery REST API.

Models are loaded ONCE at application startup via ModelRegistry.initialize()
and then cached in memory for low-latency inference.
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np

# Make src/ importable
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from rdkit import Chem  # noqa: E402
from rdkit.Chem import AllChem, Descriptors  # noqa: E402

try:
    from rdkit.Chem import QED as _QED_mod

    def _qed(mol):
        return _QED_mod.qed(mol)

except Exception:  # pragma: no cover
    def _qed(mol):  # type: ignore[misc]
        return 0.5

from drug_discovery.features import smiles_to_morgan  # noqa: E402

# ── Constants ──────────────────────────────────────────────────────────────────

TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

MODELS_DIR_TOX21 = ROOT / "01_Toxicity_Prediction" / "models"
MODELS_DIR_ACT = ROOT / "03_Activity_Classification" / "models"


# ── Model registry ─────────────────────────────────────────────────────────────

class ModelRegistry:
    """Singleton — holds all loaded ML models."""

    _instance: Optional[ModelRegistry] = None

    def __new__(cls) -> ModelRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
            # Safe defaults so properties/predict functions work before initialize()
            cls._instance.tox21_models = {}
            cls._instance.activity_model = None
        return cls._instance

    def initialize(self) -> None:
        if self._initialized:
            return
        self.tox21_models: Dict[str, object] = {}
        self.activity_model: Optional[object] = None
        self._load_tox21()
        self._load_activity()
        self._initialized = True

    def _load_tox21(self) -> None:
        if not MODELS_DIR_TOX21.exists():
            return
        for task in TOX21_TASKS:
            path = MODELS_DIR_TOX21 / f"xgb_{task.replace('-', '_')}.pkl"
            if path.exists():
                with open(path, "rb") as f:
                    self.tox21_models[task] = pickle.load(f)

    def _load_activity(self) -> None:
        path = MODELS_DIR_ACT / "xgb_egfr.pkl"
        if path.exists():
            with open(path, "rb") as f:
                self.activity_model = pickle.load(f)

    @property
    def tox21_loaded(self) -> int:
        return len(self.tox21_models)

    @property
    def activity_loaded(self) -> bool:
        return self.activity_model is not None


registry = ModelRegistry()


# ── Inference helpers ─────────────────────────────────────────────────────────

def predict_toxicity(smiles: str) -> dict:
    """Return Tox21 predictions for a SMILES string."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"valid": False, "predictions": {}, "n_toxic_tasks": 0, "overall_risk": "unknown"}

    fp = smiles_to_morgan(smiles)
    predictions: Dict[str, dict] = {}

    for task, model in registry.tox21_models.items():
        try:
            proba = float(model.predict_proba([fp])[0][1])
            distance = abs(proba - 0.5)
            confidence = "high" if distance > 0.3 else "medium" if distance > 0.15 else "low"
            predictions[task] = {
                "probability": round(proba, 4),
                "label": int(proba >= 0.5),
                "confidence": confidence,
            }
        except Exception:
            predictions[task] = {"probability": 0.0, "label": 0, "confidence": "low"}

    n_toxic = sum(1 for v in predictions.values() if v["label"] == 1)
    overall_risk = "high" if n_toxic >= 4 else "medium" if n_toxic >= 2 else "low"

    return {
        "valid": True,
        "predictions": predictions,
        "n_toxic_tasks": n_toxic,
        "overall_risk": overall_risk,
    }


def predict_admet(smiles: str) -> dict:
    """Return RDKit-computed physicochemical ADMET properties."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {}

    mw = round(Descriptors.MolWt(mol), 2)
    logp = round(Descriptors.MolLogP(mol), 3)
    tpsa = round(Descriptors.TPSA(mol), 2)
    hbd = int(Descriptors.NumHDonors(mol))
    hba = int(Descriptors.NumHAcceptors(mol))
    rotb = int(Descriptors.NumRotatableBonds(mol))
    qed = round(_qed(mol), 4)

    violations: list[str] = []
    if mw > 500:
        violations.append(f"MW={mw:.0f} > 500")
    if logp > 5:
        violations.append(f"LogP={logp:.1f} > 5")
    if hbd > 5:
        violations.append(f"HBD={hbd} > 5")
    if hba > 10:
        violations.append(f"HBA={hba} > 10")

    return {
        "molecular_weight": mw,
        "logP": logp,
        "solubility_esol": None,
        "TPSA": tpsa,
        "HBD": hbd,
        "HBA": hba,
        "RotatableBonds": rotb,
        "QED": qed,
        "lipinski_pass": len(violations) == 0,
        "lipinski_violations": violations,
    }


def predict_activity(smiles: str) -> dict:
    """Return EGFR activity prediction."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not registry.activity_loaded:
        return {
            "target": "EGFR",
            "active_probability": None,
            "label": "unknown",
            "pIC50_predicted": None,
            "IC50_uM_predicted": None,
        }

    fp = smiles_to_morgan(smiles)
    proba = float(registry.activity_model.predict_proba([fp])[0][1])  # type: ignore[union-attr]
    label = "active" if proba >= 0.5 else "inactive"

    # Heuristic mapping: p_active → pIC50 range [4, 8]
    pic50 = round(4.0 + 4.0 * proba, 2)
    ic50_um = round(10 ** (-pic50) * 1e6, 6)

    return {
        "target": "EGFR",
        "active_probability": round(proba, 4),
        "label": label,
        "pIC50_predicted": pic50,
        "IC50_uM_predicted": ic50_um,
    }

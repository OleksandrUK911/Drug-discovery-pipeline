"""
Pydantic request/response schemas for the Drug Discovery REST API.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# Import RDKit lazily to avoid hard dependency at import time
try:
    from rdkit import Chem

    _RDKIT_OK = True
except ImportError:  # pragma: no cover
    _RDKIT_OK = False


# ── Request models ─────────────────────────────────────────────────────────────

class SMILESRequest(BaseModel):
    smiles: str = Field(
        ...,
        description="SMILES string of the molecule",
        example="CC(=O)Oc1ccccc1C(=O)O",
    )

    @field_validator("smiles")
    @classmethod
    def validate_smiles(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("SMILES string cannot be empty")
        if _RDKIT_OK:
            mol = Chem.MolFromSmiles(v)
            if mol is None:
                raise ValueError(f"Invalid SMILES: '{v}' — RDKit could not parse it")
        return v


class BatchSMILESRequest(BaseModel):
    smiles_list: List[str] = Field(
        ...,
        description="List of SMILES strings (max 100)",
        max_length=100,
    )


# ── Response models ────────────────────────────────────────────────────────────

class ToxicityTaskPrediction(BaseModel):
    probability: float = Field(..., ge=0.0, le=1.0)
    label: int = Field(..., description="0 = non-toxic, 1 = toxic (threshold 0.5)")
    confidence: str = Field(..., description="high / medium / low")


class ToxicityResponse(BaseModel):
    smiles: str
    valid: bool
    predictions: Dict[str, ToxicityTaskPrediction]
    n_toxic_tasks: int
    overall_risk: str = Field(..., description="low / medium / high")


class ADMETResponse(BaseModel):
    smiles: str
    molecular_weight: float
    logP: float
    solubility_esol: Optional[float] = Field(None, description="Predicted aqueous solubility (rule-based)")
    TPSA: float
    HBD: int
    HBA: int
    RotatableBonds: int
    QED: float = Field(..., description="Quantitative Estimate of Drug-likeness [0–1]")
    lipinski_pass: bool
    lipinski_violations: List[str]


class ActivityResponse(BaseModel):
    smiles: str
    target: str = Field(..., description="Protein target name")
    active_probability: Optional[float] = Field(None, ge=0.0, le=1.0)
    label: str = Field(..., description="active / inactive / unknown")
    pIC50_predicted: Optional[float] = Field(None, description="Predicted pIC50 (heuristic)")
    IC50_uM_predicted: Optional[float] = Field(None, description="Predicted IC50 in µM (heuristic)")


class FullPredictionResponse(BaseModel):
    smiles: str
    toxicity: ToxicityResponse
    admet: ADMETResponse
    activity: ActivityResponse
    overall_score: float = Field(
        ...,
        description="Composite score: 0.4×(1−tox) + 0.3×QED + 0.3×p_active [0–1]",
    )


class BatchResultItem(BaseModel):
    smiles: str
    valid: bool
    n_toxic_tasks: Optional[int] = None
    overall_risk: Optional[str] = None
    QED: Optional[float] = None
    lipinski_pass: Optional[bool] = None
    error: Optional[str] = None


class BatchResponse(BaseModel):
    results: List[BatchResultItem]
    count: int


class MoleculeInfoResponse(BaseModel):
    smiles: str
    valid: bool
    molecular_formula: Optional[str] = None
    molecular_weight: Optional[float] = None
    exact_mass: Optional[float] = None
    logP: Optional[float] = None
    TPSA: Optional[float] = None
    HBD: Optional[int] = None
    HBA: Optional[int] = None
    RotatableBonds: Optional[int] = None
    AromaticRings: Optional[int] = None
    QED: Optional[float] = None
    lipinski_pass: Optional[bool] = None
    lipinski_violations: Optional[List[str]] = None
    InChIKey: Optional[str] = None
    error: Optional[str] = None

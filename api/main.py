"""
Drug Discovery ML Portfolio — Unified REST API
===============================================
Start:   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
Docs:    http://localhost:8000/docs
ReDoc:   http://localhost:8000/redoc

Endpoints
---------
GET  /                  — service info
GET  /health            — model load status
POST /predict/toxicity  — Tox21 12-target classification
POST /predict/admet     — ADMET physicochemical properties
POST /predict/activity  — EGFR inhibitor probability
POST /predict/full      — all predictions + composite score
POST /batch/predict     — bulk predictions (max 100 molecules)
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import (
    ADMETResponse,
    ActivityResponse,
    BatchResponse,
    BatchResultItem,
    BatchSMILESRequest,
    FullPredictionResponse,
    MoleculeInfoResponse,
    SMILESRequest,
    ToxicityResponse,
    ToxicityTaskPrediction,
)
from .predictor import TOX21_TASKS, predict_activity, predict_admet, predict_toxicity, registry


# ── Lifespan: load models once at startup ─────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    registry.initialize()
    yield


# ── Application ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Drug Discovery ML API",
    description="""
## Drug Discovery Machine Learning Portfolio — REST API

Unified prediction API exposing all trained models from the portfolio:

- 🧪 **Toxicity** — Tox21 multi-label (12 targets) via XGBoost
- 💧 **ADMET** — Physicochemical properties (MW, LogP, TPSA, QED, Lipinski Ro5)
- 🎯 **Activity** — EGFR kinase inhibitor probability via XGBoost
- 🔬 **Full** — All predictions + composite drug candidate score
- 📋 **Batch** — Bulk predictions for up to 100 molecules

### Example SMILES:
| Molecule | SMILES |
|----------|--------|
| Aspirin | `CC(=O)Oc1ccccc1C(=O)O` |
| Erlotinib (EGFR) | `COCCOC1=C(OCCO)C=C2C(=CC=NC2=C1)NC1=CC=CC(C#C)=C1` |
| Caffeine | `Cn1cnc2c1c(=O)n(C)c(=O)n2C` |
    """,
    version="1.0.0",
    contact={"name": "Drug Discovery ML Portfolio"},
    license_info={"name": "MIT"},
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/", tags=["Health"], summary="Service info")
def root() -> dict:
    return {
        "service": "Drug Discovery ML API",
        "version": "1.0.0",
        "endpoints": [
            "/predict/toxicity",
            "/predict/admet",
            "/predict/activity",
            "/predict/full",
            "/batch/predict",
        ],
        "models_loaded": {
            "tox21_tasks": registry.tox21_loaded,
            "tox21_expected": len(TOX21_TASKS),
            "activity_egfr": registry.activity_loaded,
        },
        "docs": "/docs",
    }


@app.get("/health", tags=["Health"], summary="Detailed health check")
def health() -> dict:
    return {
        "status": "ok",
        "tox21_models_loaded": registry.tox21_loaded,
        "tox21_models_expected": len(TOX21_TASKS),
        "activity_model_loaded": registry.activity_loaded,
        "tox21_ready": registry.tox21_loaded == len(TOX21_TASKS),
    }


# ── Predictions ────────────────────────────────────────────────────────────────

@app.post(
    "/predict/toxicity",
    response_model=ToxicityResponse,
    tags=["Predictions"],
    summary="Tox21 multi-label toxicity prediction (12 targets)",
)
def predict_tox(req: SMILESRequest) -> ToxicityResponse:
    result = predict_toxicity(req.smiles)
    if not result.get("valid"):
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {req.smiles}")

    task_preds = {
        task: ToxicityTaskPrediction(**vals)
        for task, vals in result["predictions"].items()
    }
    return ToxicityResponse(
        smiles=req.smiles,
        valid=True,
        predictions=task_preds,
        n_toxic_tasks=result["n_toxic_tasks"],
        overall_risk=result["overall_risk"],
    )


@app.post(
    "/predict/admet",
    response_model=ADMETResponse,
    tags=["Predictions"],
    summary="ADMET physicochemical properties (RDKit)",
)
def predict_admet_endpoint(req: SMILESRequest) -> ADMETResponse:
    result = predict_admet(req.smiles)
    if not result:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {req.smiles}")
    return ADMETResponse(smiles=req.smiles, **result)


@app.post(
    "/predict/activity",
    response_model=ActivityResponse,
    tags=["Predictions"],
    summary="EGFR kinase inhibitor activity prediction",
)
def predict_act(req: SMILESRequest) -> ActivityResponse:
    result = predict_activity(req.smiles)
    return ActivityResponse(smiles=req.smiles, **result)


@app.post(
    "/predict/full",
    response_model=FullPredictionResponse,
    tags=["Predictions"],
    summary="Full pipeline: toxicity + ADMET + activity + composite score",
)
def predict_full(req: SMILESRequest) -> FullPredictionResponse:
    tox_raw = predict_toxicity(req.smiles)
    adm_raw = predict_admet(req.smiles)
    act_raw = predict_activity(req.smiles)

    if not tox_raw.get("valid") or not adm_raw:
        raise HTTPException(status_code=422, detail=f"Invalid SMILES: {req.smiles}")

    # Composite score: safety + drug-likeness + activity
    tox_score = 1.0 - (tox_raw["n_toxic_tasks"] / max(len(TOX21_TASKS), 1))
    qed_score = float(adm_raw.get("QED") or 0.5)
    act_score = float(act_raw.get("active_probability") or 0.5)
    overall = round(0.4 * tox_score + 0.3 * qed_score + 0.3 * act_score, 4)

    task_preds = {
        task: ToxicityTaskPrediction(**vals)
        for task, vals in tox_raw["predictions"].items()
    }
    tox_resp = ToxicityResponse(
        smiles=req.smiles, valid=True,
        predictions=task_preds,
        n_toxic_tasks=tox_raw["n_toxic_tasks"],
        overall_risk=tox_raw["overall_risk"],
    )
    adm_resp = ADMETResponse(smiles=req.smiles, **adm_raw)
    act_resp = ActivityResponse(smiles=req.smiles, **act_raw)

    return FullPredictionResponse(
        smiles=req.smiles,
        toxicity=tox_resp,
        admet=adm_resp,
        activity=act_resp,
        overall_score=overall,
    )


@app.get(
    "/molecule/info",
    response_model=MoleculeInfoResponse,
    tags=["Molecules"],
    summary="Molecular descriptors without ML prediction (RDKit)",
)
def molecule_info(smiles: str) -> MoleculeInfoResponse:
    """Return rich molecular information for a SMILES string.

    Accepts SMILES as a query parameter: /molecule/info?smiles=CCO
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors, inchi

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return MoleculeInfoResponse(smiles=smiles, valid=False, error=f"Invalid SMILES: {smiles!r}")

    from .predictor import _qed

    mw = round(Descriptors.MolWt(mol), 3)
    logp = round(Descriptors.MolLogP(mol), 3)
    tpsa = round(Descriptors.TPSA(mol), 2)
    hbd = int(rdMolDescriptors.CalcNumHBD(mol))
    hba = int(rdMolDescriptors.CalcNumHBA(mol))
    rotb = int(rdMolDescriptors.CalcNumRotatableBonds(mol))
    aro = int(rdMolDescriptors.CalcNumAromaticRings(mol))
    qed = round(_qed(mol), 4)
    formula = rdMolDescriptors.CalcMolFormula(mol)
    exact = round(Descriptors.ExactMolWt(mol), 5)

    violations: list[str] = []
    if mw > 500:
        violations.append(f"MW={mw:.0f} > 500")
    if logp > 5:
        violations.append(f"LogP={logp:.1f} > 5")
    if hbd > 5:
        violations.append(f"HBD={hbd} > 5")
    if hba > 10:
        violations.append(f"HBA={hba} > 10")

    try:
        inchikey = inchi.MolToInchiKey(mol)
    except Exception:
        inchikey = None

    return MoleculeInfoResponse(
        smiles=smiles,
        valid=True,
        molecular_formula=formula,
        molecular_weight=mw,
        exact_mass=exact,
        logP=logp,
        TPSA=tpsa,
        HBD=hbd,
        HBA=hba,
        RotatableBonds=rotb,
        AromaticRings=aro,
        QED=qed,
        lipinski_pass=len(violations) == 0,
        lipinski_violations=violations,
        InChIKey=inchikey,
    )


@app.post(
    "/batch/predict",
    response_model=BatchResponse,
    tags=["Predictions"],
    summary="Bulk predictions for up to 100 molecules",
)
def predict_batch(req: BatchSMILESRequest) -> BatchResponse:
    results: list[BatchResultItem] = []
    for smi in req.smiles_list:
        try:
            tox = predict_toxicity(smi.strip())
            adm = predict_admet(smi.strip())
            results.append(
                BatchResultItem(
                    smiles=smi,
                    valid=tox.get("valid", False),
                    n_toxic_tasks=tox.get("n_toxic_tasks"),
                    overall_risk=tox.get("overall_risk"),
                    QED=adm.get("QED"),
                    lipinski_pass=adm.get("lipinski_pass"),
                )
            )
        except Exception as exc:
            results.append(BatchResultItem(smiles=smi, valid=False, error=str(exc)))

    return BatchResponse(results=results, count=len(results))

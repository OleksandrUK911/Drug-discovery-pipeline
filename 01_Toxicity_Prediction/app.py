#!/usr/bin/env python
"""FastAPI app — Tox21 multi-label toxicity prediction service.

Install:  pip install fastapi uvicorn rdkit
Run:      uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Docs:     http://localhost:8000/docs
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List
import pickle, numpy as np, os
from rdkit import Chem
from rdkit.Chem import AllChem

app = FastAPI(title="Tox21 Toxicity Predictor", version="1.0")

# ── Load models at startup ────────────────────────────────────────────────────
MODELS_DIR = "models"
TOX21_TASKS = [
    "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
    "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
    "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53",
]

xgb_models: Dict[str, object] = {}
for task in TOX21_TASKS:
    path = os.path.join(MODELS_DIR, f"xgb_{task.replace('-','_')}.pkl")
    if os.path.exists(path):
        with open(path, "rb") as f:
            xgb_models[task] = pickle.load(f)

print(f"Loaded {len(xgb_models)}/{len(TOX21_TASKS)} task models.")

def smiles_to_fp(smi: str, radius=2, nbits=2048):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits))

# ── Request / Response schemas ────────────────────────────────────────────────
class PredictRequest(BaseModel):
    smiles: str

class TaskPrediction(BaseModel):
    probability: float
    label: int      # 0 = non-toxic, 1 = toxic (threshold 0.5)

class PredictResponse(BaseModel):
    smiles:      str
    valid:       bool
    predictions: Dict[str, TaskPrediction]

# ── Endpoints ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"service": "Tox21 Predictor", "tasks": TOX21_TASKS}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    fp = smiles_to_fp(req.smiles)
    if fp is None:
        return PredictResponse(smiles=req.smiles, valid=False, predictions={})

    X = fp.reshape(1, -1)
    preds: Dict[str, TaskPrediction] = {}
    for task, model in xgb_models.items():
        prob  = float(model.predict_proba(X)[0, 1])
        preds[task] = TaskPrediction(probability=round(prob, 4), label=int(prob >= 0.5))

    return PredictResponse(smiles=req.smiles, valid=True, predictions=preds)

@app.post("/predict/batch")
def predict_batch(requests: List[PredictRequest]):
    return [predict(r) for r in requests]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

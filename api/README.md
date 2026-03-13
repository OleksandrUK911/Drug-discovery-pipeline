# Drug Discovery ML Portfolio — Unified REST API

Start the API server:

```bash
# From the project root
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

Then open:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **Health check:** http://localhost:8000/health

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info + loaded models status |
| GET | `/health` | Detailed health check |
| GET | `/molecule/info?smiles=...` | Molecular descriptors — formula, MW, QED, Lipinski, InChIKey |
| POST | `/predict/toxicity` | SMILES → Tox21 12-target predictions |
| POST | `/predict/admet` | SMILES → ADMET physicochemical properties |
| POST | `/predict/activity` | SMILES → EGFR activity probability |
| POST | `/predict/full` | SMILES → All predictions + composite score |
| POST | `/batch/predict` | List[SMILES] → Bulk predictions (max 100) |

## Example request

```bash
curl -X POST http://localhost:8000/predict/full \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'
```

## Models used

| Endpoint | Model | Source |
|----------|-------|--------|
| `/predict/toxicity` | XGBoost (12 models) | `01_Toxicity_Prediction/models/` |
| `/predict/activity` | XGBoost EGFR | `03_Activity_Classification/models/` |
| `/predict/admet` | RDKit descriptors (rule-based) | No saved model needed |

> **Note:** Models must be trained and saved from notebooks NB01 and NB03 before running the API.

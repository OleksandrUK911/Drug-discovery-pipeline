# Drug Discovery — Machine Learning Portfolio

<!-- Replace YOUR_USERNAME with your actual GitHub handle before publishing -->
![CI](https://github.com/YOUR_USERNAME/drug-discovery/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Models](https://img.shields.io/badge/models-XGBoost%20%7C%20PyTorch%20%7C%20GCN%20%7C%20VAE%20%7C%20ChemBERTa-orange)

> **7 end-to-end ML projects across the full drug discovery pipeline** — from molecular property prediction and generative chemistry to structure-based design and drug repurposing. Each notebook is production-grade: scaffold-aware splits, SHAP explainability, Optuna tuning, MLflow tracking, and a unified FastAPI serving layer.

---

## 🏆 Results at a Glance

| # | Project | Dataset | Best Model | Key Metric |
|---|---------|---------|------------|------------|
| 1 | [Toxicity Prediction](01_Toxicity_Prediction/) | Tox21 (7 831 mol.) | XGBoost | Mean AUC **0.849** across 12 targets |
| 2 | [ADMET Properties](02_ADMET_Properties/) | ESOL (1 128 mol.) | Ridge | RMSE 0.308 · R² **0.966** |
| 3 | [Activity Classification](03_Activity_Classification/) | ChEMBL EGFR (19 149) | XGBoost | AUC **0.969** |
| 4 | [Molecule Generation](04_Molecule_Generation/) | FDA drugs + ZINC 250k | LSTM · VAE · CVAE | Validity **98.2%** |
| 5 | [Molecular Clustering](05_Molecular_Clustering/) | ChEMBL drug-like | KMeans + HDBSCAN | Best DrugScore **0.849** |
| 6 | [Structure-Based Design](06_Structure_Based_Design/) | EGFR PDB 1IEP / 3W2O | AutoDock-Vina | Pearson r **> 0.70** dock vs IC50 |
| 7 | [Drug Repurposing](07_Drug_Repurposing/) | ChEMBL approved drugs | XGBoost + DTI network | Integrated repurposing score |

---

## 🚀 Quick Start

### Docker — one command

```bash
docker-compose up --build
# Streamlit dashboard → http://localhost:8501
# FastAPI Swagger UI  → http://localhost:8000/docs
```

### Streamlit Dashboard (local)

```bash
pip install -r requirements_dashboard.txt
streamlit run dashboard/app.py
# → http://localhost:8501
```

The dashboard provides:
- Live toxicity, ADMET and activity predictions for any SMILES string
- Interactive py3Dmol 3D molecule viewer (rotate in-browser)
- 3D UMAP chemical space explorer
- Tanimoto similarity graph
- Collapsible science explainers on every page

### REST API

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
# Swagger UI → http://localhost:8000/docs
```

```bash
# Example — full prediction pipeline for aspirin
curl -X POST http://localhost:8000/predict/full \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CC(=O)Oc1ccccc1C(=O)O"}'
```

### Jupyter Notebooks

```bash
pip install -r requirements.txt
jupyter lab
```

### HTML Reports (no Jupyter required)

Open `reports/index.html` in any browser — all 7 notebooks as static HTML.

### Unit Tests

```bash
pytest tests/ -v
# 96 passed
```

---

## Projects

### 01 · Toxicity Prediction
**Task:** Multi-label classification across 12 Tox21 nuclear-receptor and stress-response assays  
**Models:** Random Forest · XGBoost (Optuna-tuned) · Graph Convolutional Network (PyG)  
**Highlights:** Scaffold-aware split · SHAP beeswarm + bar · 12-task Plotly AUC heatmap · calibrated probabilities · per-molecule radar chart

### 02 · ADMET Properties
**Task:** Aqueous solubility (ESOL) and lipophilicity regression + ADMET profiling  
**Models:** Ridge · RF · XGBoost · Multi-task DNN · AttentiveFP (DeepChem)  
**Highlights:** LOF applicability domain · MC-Dropout uncertainty · stacking ensemble · parity plot with confidence bands

### 03 · Activity Classification
**Task:** Binary EGFR kinase inhibitor activity + pIC50 regression (ChEMBL)  
**Models:** RF · XGBoost · DNN · ChemBERTa (HuggingFace fine-tune)  
**Highlights:** SMILES augmentation · fingerprint benchmark (Morgan/FCFP4/MACCS/AtomPair) · SHAP waterfall

### 04 · Molecule Generation
**Task:** De-novo molecule generation with property control  
**Models:** LSTM language model · VAE · CVAE (conditioned on QED/SA) · REINVENT RL (reward = QED)  
**Highlights:** ZINC250k + KL annealing · SELFIES encoding · UMAP latent space · latent interpolation GIF

### 05 · Molecular Clustering
**Task:** Unsupervised chemical space mapping + hit-diversity selection  
**Models:** KMeans · DBSCAN · HDBSCAN · sphere exclusion  
**Highlights:** ChEMBL API pipeline · SA Score · SDF export · interactive Plotly scatter · NetworkX similarity graph · scaffold centrality

### 06 · Structure-Based Design
**Task:** Molecular docking of EGFR inhibitors (Gefitinib, Erlotinib, Afatinib, Osimertinib, Lapatinib)  
**Tools:** AutoDock-Vina · ProLIF interaction fingerprints · py3Dmol 3D viewer  
**Highlights:** WT vs T790M selectivity analysis · dock score vs IC50 correlation (r > 0.70) · interactive HTML dashboards

### 07 · Drug Repurposing
**Task:** Identify repurposing candidates for approved drugs via ML scoring + DTI graph  
**Models:** XGBoost (EGFR similarity) · ChEMBL max_phase=4 screening · pyvis DTI network  
**Highlights:** Integrated repurposing score · interactive knowledge graph · similarity matrix heatmap

---

## Architecture

```
drug-discovery/
│
├── 01_Toxicity_Prediction/       ← Tox21 multi-label
├── 02_ADMET_Properties/          ← ESOL + Lipophilicity regression
├── 03_Activity_Classification/   ← EGFR binary + pIC50
├── 04_Molecule_Generation/       ← LSTM · VAE · CVAE · REINVENT
├── 05_Molecular_Clustering/      ← KMeans · HDBSCAN · scaffold analysis
├── 06_Structure_Based_Design/    ← AutoDock-Vina · ProLIF · py3Dmol
├── 07_Drug_Repurposing/          ← ChEMBL · DTI network · pyvis
│
├── api/                          ← FastAPI REST — all models unified
│   ├── main.py                   ← 8 endpoints + lifespan model loading
│   ├── models.py                 ← Pydantic v2 request/response schemas
│   └── predictor.py              ← ModelRegistry singleton
│
├── dashboard/app.py              ← Streamlit 8-page interactive app
├── reports/                      ← Static HTML exports of all notebooks
├── src/drug_discovery/           ← Shared feature engineering (importable)
├── tests/                        ← 96 unit tests (pytest)
│
├── Dockerfile                    ← Production Docker image
├── docker-compose.yml            ← API + Dashboard orchestration
├── Makefile                      ← make dashboard / test / reports / api
├── pyproject.toml                ← black · isort · flake8 config
└── .github/workflows/ci.yml      ← CI: pytest on Python 3.10/3.11/3.12
```

---

## REST API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Service info + model status |
| GET | `/health` | Detailed health check |
| GET | `/molecule/info` | Molecular descriptors (MW, QED, Lipinski, InChIKey) |
| POST | `/predict/toxicity` | SMILES → Tox21 12-target probabilities |
| POST | `/predict/admet` | SMILES → physicochemical profile |
| POST | `/predict/activity` | SMILES → EGFR probability + pIC50 |
| POST | `/predict/full` | SMILES → toxicity + ADMET + activity (unified) |
| POST | `/batch/predict` | List[SMILES] → bulk predictions (max 100) |

---

## Tech Stack

### Molecular Libraries
| Library | Purpose |
|---------|---------|
| **RDKit** | Descriptors, Morgan fingerprints, molecule drawing |
| **DeepChem** | MoleculeNet datasets (Tox21, ESOL, Lipophilicity) |
| **chembl-webresource-client** | ChEMBL bioactivity API |
| **mols2grid** | Interactive molecule grid in Jupyter |
| **py3Dmol** | 3D conformer viewer (inline Jupyter + Streamlit) |

### Machine Learning / Deep Learning
| Library | Purpose |
|---------|---------|
| **scikit-learn** | RF, KMeans, DBSCAN, metrics, cross-validation |
| **XGBoost** | Gradient boosting — classification and regression |
| **PyTorch** | DNN, LSTM, VAE, CVAE |
| **torch-geometric** | Graph Convolutional Networks (GCN) |
| **transformers** | ChemBERTa fine-tuning on SMILES |
| **SHAP** | Explainability — feature importance and waterfall plots |
| **optuna** | Bayesian hyperparameter search |
| **selfies** | SELFIES tokenisation for generative models |

### Deployment and Tooling
| Tool | Purpose |
|------|---------|
| **FastAPI + Uvicorn** | REST API serving |
| **Streamlit** | Interactive web dashboard |
| **Docker + Compose** | Reproducible deployment |
| **MLflow** | Experiment tracking |
| **pytest + pytest-cov** | 96 unit tests + coverage |
| **pre-commit** | black · isort · flake8 hooks |

---

## Skills Demonstrated

```
Molecular representation: Morgan fingerprints, RDKit descriptors, SMILES, SELFIES
Multi-label classification (Tox21), binary classification (EGFR), regression (ADMET)
Graph Neural Networks: GCN on molecular graphs (PyTorch Geometric)
Generative models: LSTM, VAE, CVAE, REINVENT RL with QED reward
Unsupervised learning: KMeans, DBSCAN, HDBSCAN, UMAP, t-SNE, PCA
Transformer fine-tuning: ChemBERTa on SMILES sequences
Explainability: SHAP values, feature importance, waterfall plots
Scaffold-aware train/test split — no data leakage
Probability calibration + MC-Dropout uncertainty quantification
Hyperparameter tuning: Optuna Bayesian search
Structure-based drug design: docking, interaction fingerprints, 3D visualisation
Drug repurposing: ChEMBL screening + DTI knowledge graph
Production API: FastAPI, Pydantic v2, async lifespan, CORS
CI/CD: GitHub Actions — pytest on Python 3.10/3.11/3.12
Containerisation: Docker multi-stage build + docker-compose
Data: Tox21, ChEMBL, ESOL, MoleculeNet, ZINC250k, PDB
```

---

## Model Files

Large Random Forest models (~10–35 MB each) are excluded from the repository.
To regenerate them, run `01_Toxicity_Prediction/Toxicity_Prediction.ipynb` top-to-bottom.

Small XGBoost and PyTorch models are committed and ready to use immediately:

| File | Size | Used by |
|------|------|---------|
| `01_Toxicity_Prediction/models/xgb_*.pkl` | ~0.4 MB each | `/predict/toxicity` |
| `02_ADMET_Properties/models/multi_task_admet.pt` | 4.6 MB | ADMET dashboard page |
| `03_Activity_Classification/models/xgb_egfr.pkl` | 0.6 MB | `/predict/activity` |

---

## Key References

1. Gomez-Bombarelli et al. (2018) — *Automatic Chemical Design Using a Data-Driven Continuous Representation of Molecules* (VAE)
2. Kipf & Welling (2017) — *Semi-Supervised Classification with Graph Convolutional Networks* (GCN)
3. Wu et al. (2018) — *MoleculeNet: A Benchmark for Molecular Machine Learning*
4. Xiong et al. (2020) — *Pushing the Boundaries of Molecular Representation for Drug Discovery with the Graph Attention Mechanism* (AttentiveFP)
5. Chithrananda et al. (2020) — *ChemBERTa: Large-Scale Self-Supervised Pretraining for Molecular Property Prediction*
6. Olivecrona et al. (2017) — *Molecular de-novo design through deep reinforcement learning* (REINVENT)
7. Krenn et al. (2020) — *Self-Referencing Embedded Strings (SELFIES)*
8. Bickerton et al. (2012) — *Quantifying the chemical beauty of drugs* (QED)

---

## License

[MIT](LICENSE) © 2026 Oleksandr

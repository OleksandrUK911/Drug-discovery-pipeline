# Project 7: Drug Repurposing via Molecular Similarity & Network Analysis

## Objective

Identify **FDA-approved drugs** that could be repurposed as novel EGFR inhibitors using a computational pipeline combining:
1. Structural similarity search (Tanimoto / Morgan fingerprints)
2. ML activity prediction (XGBoost model from Project 3)
3. Drug-Target Interaction (DTI) network analysis

> **AstraZeneca connection:** AZ actively pursues drug repurposing — **Ticagrelor (Brilinta)** was investigated for COVID-19, **Dapagliflozin** expanded from diabetes to heart failure/CKD. Computational prioritization is the first step in every repurposing program.

## Scientific Background

Drug repurposing finds new therapeutic uses for approved drugs. Key advantages:
- **Safety profile already known** → Phase I trials can be skipped
- **Development timeline:** ~3–5 years vs ~12–15 years de novo
- **Cost:** ~$300M vs ~$2.6B for new molecular entity
- **Success rate:** ~25% vs ~5% for new drugs

### Approaches used in this notebook:

| Method | Description |
|--------|-------------|
| Tanimoto similarity | Structural similarity to known EGFR inhibitors |
| ML prediction | Apply NB03 XGBoost model to approved drugs |
| Integrated score | Combine similarity + probability + QED |
| DTI network | Visualize drug-target connections |

## Dataset

| Source | Content | Size |
|--------|---------|------|
| ChEMBL (max_phase=4) | FDA-approved small molecules | ~2500 |
| Fallback (curated) | 20 well-known kinase inhibitors | 20 |

## Methods

| Step | Tool |
|------|------|
| ChEMBL API query | `chembl-webresource-client` |
| Morgan fingerprints | `RDKit` (radius=2, 2048 bits) |
| Tanimoto similarity | `DataStructs.BulkTanimotoSimilarity` |
| ML prediction | XGBoost from NB03 (loaded via pickle) |
| Network analysis | `NetworkX` + `pyvis` |
| Visualization | `Plotly` (bar, heatmap, scatter) |
| Experiment tracking | `MLflow` |

## Expected Results

| Metric | Expected |
|--------|----------|
| Approved drugs screened | ~2,500 (or 20 fallback) |
| Tanimoto ≥ 0.3 to EGFR | ~5–15% |
| ML-predicted active | ~3–8% |
| Top repurposing score | > 0.75 |

Likely top candidates:
- Other kinase inhibitors (Imatinib, Dasatinib analogs)
- Compounds with quinazoline or anilinopyrimidine scaffold

## Files

```
07_Drug_Repurposing/
├── Drug_Repurposing.ipynb               ← main notebook (TODO: implement)
├── README.md                            ← this file
└── data/
    ├── approved_drugs_chembl.csv         ← ChEMBL cache (auto-downloaded)
    ├── repurposing_candidates.csv         ← top scoring compounds
    ├── dti_network.html                   ← interactive DTI network
    ├── similarity_matrix.html             ← pairwise similarity heatmap
    └── repurposing_candidates.html        ← ranked candidates plot
```

## Dependencies

```bash
pip install chembl-webresource-client>=0.10
# Already in requirements.txt: pyvis, networkx, plotly, rdkit
```

## Status

🔴 **TODO** — Notebook not yet implemented. See [TODO/NB07_Drug_Repurposing.md](../TODO/NB07_Drug_Repurposing.md) for the full step-by-step implementation plan with code.

## Connection to Other Notebooks

| Notebook | Connection |
|----------|-----------|
| NB01 Toxicity Prediction | Tox21 models → safety screening of repurposing candidates |
| NB03 Activity Classification | XGBoost EGFR model → ML-based repurposing |
| NB05 Molecular Clustering | Pre-screen approved drugs in chemical space |
| NB06 Structure-Based Design | Dock top repurposing candidates into EGFR active site |

## Why this matters for AstraZeneca

AstraZeneca Cambridge Centre for Drug Discovery uses exactly this workflow:
1. Computational screen (similarity + ML) → prioritize ~20 candidates
2. SBDD validation (docking) → narrow to top 5
3. *In vitro* IC50 assay → confirm 1–3 hits
4. Lead optimization → clinical candidate

This notebook demonstrates **step 1** of that pipeline.

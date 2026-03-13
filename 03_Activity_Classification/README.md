# Project 3: Molecular Activity Classification Against Protein Targets

## Objective
Predict whether a molecule will be **active** against specific protein targets using bioactivity data from ChEMBL / PubChem BioAssay. Focus target: **EGFR (Epidermal Growth Factor Receptor)** — a major oncology target.

## Dataset
- **ChEMBL** bioactivity data for EGFR (target CHEMBL203)
- ~10,000+ compounds with IC50 / Ki measurements
- Binary labels: Active (IC50 < 1μM) / Inactive (IC50 > 10μM)
- Additional targets: BACE1, HIV Protease, SARS-CoV-2 inhibitors (via MoleculeNet)

## Models Used
| Model | Features | Notes |
|---|---|---|
| Random Forest | Morgan fingerprints | Baseline |
| XGBoost | Morgan + RDKit descriptors | Best classical ML |
| Deep Neural Network | Morgan fingerprints | 3-layer MLP with Dropout |
| Graph Convolutional Network | Molecular graph | End-to-end graph learning |

## Key Skills Demonstrated
- **ChEMBL API** data retrieval and preprocessing
- IC50-to-binary label conversion with activity cliffs awareness
- **Class imbalance handling** (SMOTE, class weights, threshold tuning)
- ROC-AUC, PR-AUC, F1, MCC evaluation metrics
- **Scaffold-based train/test split** (no information leakage)
- Feature importance with SHAP + molecular substructure matching

## Results Summary
- ROC-AUC, PR-AUC, F1, MCC per model
- Confusion matrices
- Activity cliff analysis
- Top structural features vs inactive compounds

## How to Run
```bash
pip install rdkit chembl-webresource-client xgboost shap imbalanced-learn scikit-learn
jupyter notebook Activity_Classification.ipynb
```

## Project Structure
```
03_Activity_Classification/
├── README.md
├── Activity_Classification.ipynb
└── data/
```

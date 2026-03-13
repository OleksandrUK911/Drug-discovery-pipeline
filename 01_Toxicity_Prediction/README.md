# Project 1: Toxicity Prediction of Chemical Compounds

## Objective
Build a machine learning model that predicts the toxicity of molecules across 12 biological targets using the **Tox21** dataset. The project progresses from classical ML to a Graph Neural Network (GCN) for molecular representation.

## Dataset
- **Tox21** — ~8,000 compounds tested against 12 toxicological targets (e.g., NR-AR, SR-MMP, NR-AhR)
- Downloaded automatically via `deepchem` or `MoleculeNet`
- Molecules represented as SMILES strings → converted to Morgan fingerprints and molecular graphs

## Models Used
| Model | Type | Notes |
|---|---|---|
| Random Forest | Baseline classifier | Morgan fingerprints as features |
| XGBoost | Boosting classifier | Morgan fingerprints as features |
| Graph Convolutional Network (GCN) | Deep Learning | Molecular graph as input |

## Key Skills Demonstrated
- Working with real bioactivity datasets (Tox21 / MoleculeNet)
- Molecular feature engineering with **RDKit** (Morgan fingerprints, descriptors)
- Multi-label binary classification
- Model evaluation: ROC-AUC, F1, confusion matrix
- Graph Neural Networks for chemistry (via DeepChem / PyTorch Geometric)
- SHAP feature importance for model explainability

## Results Summary
See the notebook for full results. Key metrics reported per toxicity target:
- ROC-AUC per task
- Mean ROC-AUC across all 12 targets
- Feature importance (SHAP values for RF/XGBoost)
- GCN attention visualization

## How to Run
```bash
pip install rdkit deepchem xgboost shap matplotlib seaborn scikit-learn
jupyter notebook Toxicity_Prediction.ipynb
```

## Project Structure
```
01_Toxicity_Prediction/
├── README.md
├── Toxicity_Prediction.ipynb   # Main notebook
└── data/                        # Auto-downloaded by deepchem
```

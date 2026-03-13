# Project 2: ADMET Properties Prediction

## Objective
Predict key pharmacokinetic (ADMET) properties of drug candidate molecules:
- **Aqueous Solubility** (logS) — ESOL dataset
- **Lipophilicity** (logP) — Lipophilicity dataset
- **Blood-Brain Barrier Permeability** (BBBP) — MoleculeNet

## Dataset
| Property | Dataset | Size | Task |
|---|---|---|---|
| Solubility (logS) | ESOL / MoleculeNet | 1,128 compounds | Regression |
| Lipophilicity (logP) | Lipophilicity / MoleculeNet | 4,200 compounds | Regression |
| BBB Permeability | BBBP | 2,053 compounds | Classification |

All datasets are loaded via `deepchem` / `MoleculeNet` or as fallback CSV.

## Models Used
| Model | Type | Notes |
|---|---|---|
| Random Forest Regressor | Baseline | Morgan fingerprints |
| Gradient Boosting (XGBoost) | Boosting | Morgan + descriptors |
| Deep Neural Network (DNN) | Neural Network | 3-layer MLP |
| Graph Convolutional Network | Graph DL | Molecular graph input |

## Key Skills Demonstrated
- **Regression tasks** with molecular data (RMSE, R², MAE)
- Physicochemical descriptor engineering (MW, LogP, TPSA, HBD, HBA)
- **Cross-validation** with scaffold splitting (realistic drug discovery evaluation)
- Neural network design with PyTorch for molecular property prediction
- **Parity plots** (predicted vs actual), residual analysis
- Feature importance visualization

## Results Summary
Evaluation metrics reported for each property:
- **RMSE** (Root Mean Squared Error)
- **R²** (Coefficient of determination)
- **MAE** (Mean Absolute Error)
- Parity plots and residual distributions

## How to Run
```bash
pip install rdkit deepchem xgboost torch scikit-learn matplotlib seaborn
jupyter notebook ADMET_Properties_Prediction.ipynb
```

## Project Structure
```
02_ADMET_Properties/
├── README.md
├── ADMET_Properties_Prediction.ipynb
└── data/
```

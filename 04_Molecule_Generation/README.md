# Project 4: De Novo Molecule Generation with VAE and LSTM

## Objective
Build **generative models** that design novel drug-like molecules with desired properties:
1. **LSTM on SMILES** — train a language model to sample new SMILES strings
2. **Variational Autoencoder (VAE)** — learn a continuous latent molecular space and perform property-guided optimization

## Dataset
- **ChEMBL** drug-like compounds (MW < 500, LogP < 5) in SMILES format
- ~200,000 filtered molecules used for training
- Alternatively: ZINC250k subset (250k drug-like molecules)

## Models Used
| Model | Description |
|---|---|
| Character-level LSTM | Sequences model trained on SMILES tokens |
| VAE (RNN-based) | Encoder-Decoder with KL-divergence regularisation |
| Property predictor | Regression head on latent vector for constrained generation |

## Key Skills Demonstrated
- **SMILES tokenization** and vocabulary building
- Sequence modelling with PyTorch LSTM/GRU
- **Variational Autoencoder** training (ELBO loss = reconstruction + KL)
- Latent space interpolation between two molecules
- **Bayesian Optimization** in latent space for property optimization
- Validity / Uniqueness / Novelty (VUN) metrics for molecular generation
- RDKit validity checks and property filtering

## Metrics
| Metric | Description |
|---|---|
| Validity | % generated SMILES parseable by RDKit |
| Uniqueness | % unique among valid molecules |
| Novelty | % not in training set |
| KL Divergence | Property distribution vs training set |

## How to Run
```bash
pip install rdkit torch torchvision pandas numpy matplotlib tqdm
jupyter notebook Molecule_Generation_VAE.ipynb
```

## Project Structure
```
04_Molecule_Generation/
├── README.md
├── Molecule_Generation_VAE.ipynb
└── data/
    └── chembl_smiles.csv     # auto-downloaded
```

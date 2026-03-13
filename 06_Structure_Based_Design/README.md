# Project 6: Structure-Based Drug Design (SBDD) & Molecular Docking

## Objective

Apply **Structure-Based Drug Design (SBDD)** methods to the EGFR kinase target — connecting the computational pipeline from Projects 1–5 to the 3D protein structure. This notebook demonstrates how drug discovery companies like **AstraZeneca** design and prioritize compounds using protein crystal structures.

> **AstraZeneca connection:** AZ's blockbuster drug **Osimertinib (Tagrisso)** — a ~$4B/year EGFR inhibitor — was designed using exactly this SBDD pipeline against the T790M resistance mutation (PDB: 3W2O).

## Why Structure-Based?

| Approach | What it uses | Example |
|----------|-------------|---------|
| Ligand-Based (NB01-NB03) | Known actives → ML model | QSAR, fingerprints, GCN |
| Structure-Based (NB06) | Protein 3D structure → docking | Explicit protein-ligand binding |
| Combined (NB06 + NB03) | Both | Correlation: docking score vs IC50 |

## Target: EGFR Kinase

| PDB ID | Description | Resolution | Ligand |
|--------|-------------|------------|--------|
| `1IEP` | EGFR wildtype + erlotinib analog | 1.65 Å | 4-anilinoquinazoline |
| `3W2O` | EGFR T790M mutant | 2.5 Å | WZ4003 (resistance model) |

**Why T790M?** The T790M gatekeeper mutation causes resistance to erlotinib/gefitinib. AstraZeneca developed **Osimertinib** specifically to overcome this — it covalently binds Cys797 and retains affinity for T790M. This notebook simulates that discovery logic.

## Methods

| Step | Method | Tool |
|------|--------|------|
| PDB download | RCSB REST API | `requests` |
| Protein preparation | Remove water/HETATM, add H | `pdbfixer` / manual |
| Ligand 3D conformer | ETKDG + MMFF94 | `RDKit` |
| Molecular docking | AutoDock-Vina scoring | `vina` Python API |
| Interaction analysis | Protein-Ligand Interaction Fingerprints | `ProLIF` |
| Visualization | Interactive 3D | `py3Dmol` |
| Correlation analysis | Docking ΔG vs experimental IC50 | `scipy`, `plotly` |
| Experiment tracking | Per-compound logging | `MLflow` |

## Compounds Docked

| Compound | Type | IC50 (nM) | AZ relevance |
|----------|------|-----------|-------------|
| Erlotinib | 1st gen EGFR inhibitor | 1 | Precursor |
| Gefitinib | 1st gen EGFR inhibitor | 5.6 | Precursor |
| **Osimertinib** | 3rd gen (T790M) | 0.84 | **AZ Tagrisso** |
| Lapatinib | Dual EGFR/HER2 | 11 | Comparison |
| Afatinib | 2nd gen covalent | 0.5 | Comparison |
| Top NB05 candidates | Novel drug-like | Unknown | **Repurposing targets** |

## Key Results

See `Structure_Based_Design.ipynb` for full results.

| Metric | Expected |
|--------|----------|
| Docking Score vs IC50 Pearson r | > 0.70 |
| Best docking compound | Osimertinib (ΔG ~ −10 to −11 kcal/mol) |
| Key binding interactions | H-bond: Thr790, Met793 (hinge) |
| T790M resistance insight | Reduced affinity for 1st gen, retained for Osimertinib |

## Files

```
06_Structure_Based_Design/
├── Structure_Based_Design.ipynb     ← main notebook (TODO: implement)
├── README.md                        ← this file
└── data/
    ├── 1IEP.pdb                     ← auto-downloaded from RCSB
    ├── 3W2O.pdb                     ← auto-downloaded from RCSB
    ├── 1IEP_clean.pdb               ← prepared protein
    ├── Erlotinib_3d.sdf             ← 3D conformers (generated)
    ├── Osimertinib_3d.sdf
    ├── docking_results.csv           ← docking scores
    └── docking_vs_ic50.html         ← interactive correlation plot
```

## Dependencies

```bash
pip install vina>=1.2 prolif>=2.0 meeko>=0.5 biopython>=1.81
# Optional (protein preparation with OpenMM):
conda install -c conda-forge pdbfixer openmm
```

## Status

🔴 **TODO** — Notebook not yet implemented. See [TODO/NB06_Structure_Based_Design.md](../TODO/NB06_Structure_Based_Design.md) for the full implementation plan with code.

## Connection to Other Notebooks

| Notebook | Connection |
|----------|-----------|
| NB03 Activity Classification | EGFR IC50 data → correlation with docking scores |
| NB05 Molecular Clustering | Top candidates → dock into EGFR active site |
| NB07 Drug Repurposing | Docking validates repurposing candidates |

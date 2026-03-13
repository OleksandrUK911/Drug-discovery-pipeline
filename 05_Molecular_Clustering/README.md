# Project 5: Molecular Clustering & Chemical Space Exploration

## Objective
Analyze a large molecular library, discover clusters of structurally similar compounds, and identify **potential drug candidates** based on their positioning in chemical space. This is an unsupervised learning approach for lead compound discovery.

## Dataset
- **ChEMBL** — drug-like compounds (Lipinski-filtered)
- **ZINC** — commercially available compounds
- ~10,000–50,000 molecules analyzed

## Methods Used
| Step | Method | Purpose |
|---|---|---|
| Feature extraction | Morgan fingerprints (2048-bit) | Molecular representation |
| Dimensionality reduction | PCA → t-SNE / UMAP | 2D visualization of chemical space |
| Clustering | KMeans + DBSCAN | Grouping structurally similar molecules |
| Visualization | Scatter plots + mols2grid | Interactive exploration |
| Scoring | Lipinski / QED / SA Score | Drug-likeness ranking |

## Key Skills Demonstrated
- Large-scale molecular dataset processing with **RDKit**
- **Tanimoto similarity** and fingerprint-based distance matrices
- Dimensionality reduction comparison: PCA vs t-SNE vs UMAP
- **Silhouette score** and Elbow method for cluster number selection
- DBSCAN for density-based clustering and outlier detection
- Chemical space visualization and medicinal chemistry insights
- **QED (Quantitative Estimate of Drug-likeness)** scoring
- Interactive molecular grid with `mols2grid`

## Results
- 2D chemical space maps colored by cluster, MW, LogP, QED
- Cluster statistics (size, mean properties, drug-likeness)
- Ranked list of top drug-candidate molecules per cluster
- CSV export of top candidates with SMILES + properties

## How to Run
```bash
pip install rdkit umap-learn scikit-learn matplotlib seaborn mols2grid pandas
jupyter notebook Molecular_Clustering.ipynb
```

## Project Structure
```
05_Molecular_Clustering/
├── README.md
├── Molecular_Clustering.ipynb
├── top_candidates.csv         # output
└── data/
```

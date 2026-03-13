# NB07 — Drug Repurposing via Molecular Similarity & Network Analysis

> **Статус:** ✅ Готово — реалізовано в `07_Drug_Repurposing/Drug_Repurposing.ipynb`  
> **Пріоритет:** 🟡 HIGH — актуальна тема для AstraZeneca  
> **Причина:** AZ відома drug repurposing стратегіями (Brilinta для COVID, Iressa → Tagrisso pipeline). Показати розуміння цього підходу = пряме влучення в AZ research culture.  
> **Зв'язок:** Використовує моделі з NB01-NB03, молекули з NB05, хімічний простір.

---

## Мета ноутбука

1. Визначити approved drugs, які мають схожість до відомих EGFR/токсичних сполук
2. Знайти молекули-кандидати для repurposing через структурну схожість
3. Побудувати Drug-Target Interaction (DTI) network
4. Запустити ML-моделі з NB03 на approved drugs → знайти нові показання
5. Порівняти repurposing кандидатів з NB05 топ-кандидатами

---

## Необхідні залежності

```bash
pip install chembl-webresource-client   # ChEMBL API
pip install pyvis                        # Network graphs (вже є)
pip install networkx                     # Вже є
pip install requests                     # Вже є
pip install drugbank-downloader          # Optional: DrugBank data
```

Додати до `requirements.txt`:
```
chembl-webresource-client>=0.10
```

---

## Наукове обґрунтування

Drug repurposing = знаходження нових терапевтичних використань для вже схвалених ліків.
- **Швидше та дешевше:** Safety profile вже відомий → Phase I trials можна пропустити
- **Приклади:** Sildenafil (Viagra → легенева гіпертензія), Thalidomide (проти раку), Metformin (діабет → онкологія)
- **AstraZeneca:** Ticagrelor (Brilinta) досліджувався для COVID-19; Dapagliflozin (серцева недостатність → CKD)

### Підходи до repurposing:
1. **Similarity-based:** FDA drugs → схожі до активних сполук EGFR з NB03
2. **Network-based:** Drug-Target Interaction (DTI) network → нові target-drug зв'язки
3. **ML-based:** Запустити моделі NB03 на всіх approved drugs → знайти нові активні

---

## Структура ноутбука

### Cell 1 — Imports та налаштування

```python
import os, sys, time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED
import pickle, warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, "../src")
from drug_discovery.features import smiles_to_morgan, smiles_to_descriptors, lipinski_filter

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_OK = True
except ImportError:
    CHEMBL_OK = False
    print("⚠️  ChEMBL client not available. Using cached data.")
```

### Cell 2 — Markdown: наукове введення

```markdown
## Drug Repurposing: Finding New Uses for Approved Drugs

Drug repurposing exploits the fact that approved drugs have known safety profiles.
By searching for structural similarity between approved drugs and active compounds
against our targets of interest, we can identify candidates that might be 
"repurposed" without starting from scratch.

### Pipeline Overview:
1. **Database query:** Download FDA-approved drugs from ChEMBL (max_phase=4)
2. **Similarity search:** Tanimoto similarity vs known EGFR inhibitors (NB03)
3. **ML prediction:** Apply NB03 activity model to all approved drugs
4. **DTI network:** Visualize drug-target connections as interactive network
5. **Ranking:** Score candidates by similarity + predicted activity + drug-likeness

### Why this matters for AstraZeneca?
AZ invests heavily in repurposing. Their computational screening approach uses
exactly this pipeline: molecular similarity → activity prediction → prioritization.
```

### Cell 3 — Завантаження FDA-approved drugs з ChEMBL

```python
CACHE_PATH = "data/approved_drugs_chembl.csv"
os.makedirs("data", exist_ok=True)

if os.path.exists(CACHE_PATH):
    df_approved = pd.read_csv(CACHE_PATH)
    print(f"✅ Loaded {len(df_approved)} approved drugs from cache")
    
elif CHEMBL_OK:
    molecule = new_client.molecule
    print("🔄 Querying ChEMBL for FDA-approved small molecules...")
    
    approved = molecule.filter(
        max_phase=4,                    # Phase 4 = FDA approved
        molecule_type="Small molecule",
        natural_product=False,
    ).only([
        "molecule_chembl_id",
        "pref_name",
        "molecule_structures",
        "molecule_properties",
    ])
    
    records = []
    for drug in approved:
        smiles = drug.get("molecule_structures", {})
        if smiles and smiles.get("canonical_smiles"):
            props = drug.get("molecule_properties") or {}
            records.append({
                "chembl_id": drug["molecule_chembl_id"],
                "name": drug.get("pref_name", ""),
                "SMILES": smiles["canonical_smiles"],
                "MW": props.get("mw_freebase"),
                "LogP": props.get("alogp"),
                "TPSA": props.get("psa"),
                "HBD": props.get("hbd"),
                "HBA": props.get("hba"),
            })
    
    df_approved = pd.DataFrame(records)
    df_approved.to_csv(CACHE_PATH, index=False)
    print(f"✅ Downloaded and cached {len(df_approved)} approved drugs")
else:
    # Fallback: Curated list of 50 well-known FDA drugs
    APPROVED_DRUGS_FALLBACK = {
        "Imatinib":    "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1",
        "Erlotinib":   "COCCOC1=C(OCCO)C=C2C(=CC=NC2=C1)NC1=CC=CC(C#C)=C1",
        "Gefitinib":   "COC1=C(OCC2=CC=CC=C2F)C=C2C(=CC=NC2=C1)NC1=CC(Cl)=C(F)C=C1",
        "Sorafenib":   "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1",
        "Sunitinib":   "CCN(CC)CCNC(=O)c1c(C)[nH]c(/C=C2\\C(=O)Nc3ccc(F)cc32)c1C",
        "Lapatinib":   "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
        "Dasatinib":   "Cc1nc(Nc2ncc(C(=O)Nc3c(C)cccc3Cl)s2)cc(N2CCN(CCO)CC2)n1",
        "Nilotinib":   "Cc1cn(-c2cc(NC(=O)c3ccc(C)c(Nc4nccc(-c5cccnc5)n4)c3)cc(C(F)(F)F)c2)cn1",
        "Vemurafenib": "CCCS(=O)(=O)Nc1ccc(F)c(C(=O)c2ccc(Cl)cc2F)c1",
        "Ruxolitinib": "C[C@@H](CC#N)n1cc(-c2cn3cccc3n2)c(=O)[nH]1",
        "Ibrutinib":   "C=CC(=O)N1CCC[C@@H](n2ncc3ccccc32)C1Nc1ncnc2[nH]ccc12",
        "Olaparib":    "O=C(c1ccc(N2CCCC2=O)cc1)c1[nH]nnc1-c1ccccc1F",
        "Metformin":   "CN(C)C(=N)NC(=N)N",
        "Aspirin":     "CC(=O)Oc1ccccc1C(=O)O",
        "Atorvastatin":"CC(C)c1n(-c2ccccc2)c(-c2ccc(F)cc2)c(CC[C@@H](O)C[C@@H](O)CC(=O)O)c1C(=O)Nc1ccccc1",
        "Sildenafil":  "CCCC1=NN(C)C(=O)/C1=C/c1cc(S(=O)(=O)N2CCN(C)CC2)ccc1OCC",
        "Osimertinib": "COC1=CC2=NC=CC(=C2CC=C)NC1=NC1=CC=CC(NC(=O)/C=C/CN(C)C)=C1",
        "Afatinib":    "CN(C)/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCC[F]",
        "Cetuximab_sm":"c1ccc(-c2ccc3[nH]ncc3c2)cc1",  # simplified analog
        "Paclitaxel":  "CC1=C2[C@@](O)([C@H](OC(=O)c3ccccc3)[C@@]3(O)C[C@H](OC(=O)[C@H](O)[C@@H](NC(=O)c4ccccc4)c4ccccc4)C(=C1)C23)C(C)=O",
    }
    df_approved = pd.DataFrame([
        {"name": k, "SMILES": v, "chembl_id": f"FALLBACK_{i:03d}"}
        for i, (k, v) in enumerate(APPROVED_DRUGS_FALLBACK.items())
    ])
    print(f"✅ Using fallback dataset: {len(df_approved)} approved drugs")

# Верифікація SMILES
df_approved["mol_valid"] = df_approved["SMILES"].apply(
    lambda s: Chem.MolFromSmiles(str(s)) is not None
)
df_approved = df_approved[df_approved["mol_valid"]].copy()
print(f"Valid molecules: {len(df_approved)}")
```

### Cell 4 — Базова статистика approved drugs

```python
# Drug-likeness filtering
from rdkit.Chem import Descriptors

def get_props(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return {}
    return {
        "MW":    Descriptors.MolWt(mol),
        "LogP":  Descriptors.MolLogP(mol),
        "HBD":   Descriptors.NumHDonors(mol),
        "HBA":   Descriptors.NumHAcceptors(mol),
        "TPSA":  Descriptors.TPSA(mol),
        "QED":   QED.qed(mol),
        "RotBonds": Descriptors.NumRotatableBonds(mol),
    }

props_list = [get_props(s) for s in df_approved["SMILES"].tolist()]
df_props = pd.DataFrame(props_list)
df_approved = pd.concat([df_approved.reset_index(drop=True), df_props], axis=1)

# Lipinski Ro5 filter
df_lipinski = df_approved[
    (df_approved["MW"] <= 500) &
    (df_approved["LogP"] <= 5) &
    (df_approved["HBD"] <= 5) &
    (df_approved["HBA"] <= 10)
].copy()

print(f"Total approved drugs:       {len(df_approved)}")
print(f"Lipinski Ro5 compliant:     {len(df_lipinski)} ({100*len(df_lipinski)/len(df_approved):.1f}%)")

# Plotly: MW distribution
fig = px.histogram(
    df_approved[df_approved["MW"] < 1000], x="MW", nbins=50,
    title="Molecular Weight Distribution — FDA Approved Drugs (ChEMBL max_phase=4)",
    labels={"MW": "Molecular Weight (Da)"},
    template="plotly_white", color_discrete_sequence=["#667eea"]
)
fig.add_vline(x=500, line_dash="dash", line_color="red", annotation_text="Lipinski MW ≤ 500")
fig.show()
```

### Cell 5 — Similarity-based repurposing (проти EGFR reference set)

```python
# Reference molecules: відомі EGFR інгібітори з NB03
EGFR_REFERENCE = {
    "Erlotinib":   "COCCOC1=C(OCCO)C=C2C(=CC=NC2=C1)NC1=CC=CC(C#C)=C1",
    "Gefitinib":   "COC1=C(OCC2=CC=CC=C2F)C=C2C(=CC=NC2=C1)NC1=CC(Cl)=C(F)C=C1",
    "Osimertinib": "COC1=CC2=NC=CC(=C2CC=C)NC1=NC1=CC=CC(NC(=O)/C=C/CN(C)C)=C1",
    "Lapatinib":   "CS(=O)(=O)CCNCc1ccc(-c2ccc3ncnc(Nc4ccc(OCc5cccc(F)c5)c(Cl)c4)c3c2)o1",
    "Afatinib":    "CN(C)/C=C/C(=O)Nc1cc2c(Nc3ccc(F)c(Cl)c3)ncnc2cc1OCC[F]",
}

# Обчислення Tanimoto similarity
def tanimoto_to_reference(smi, ref_fps, radius=2, nbits=2048):
    """Повертає max Tanimoto similarity до будь-якого reference."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)
    sims = DataStructs.BulkTanimotoSimilarity(fp, ref_fps)
    return max(sims) if sims else 0.0

# Precompute reference FPs
ref_fps = []
for ref_smi in EGFR_REFERENCE.values():
    ref_mol = Chem.MolFromSmiles(ref_smi)
    ref_fps.append(AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, 2048))

# Обчислити для всіх approved drugs
print("🔄 Computing Tanimoto similarities to EGFR reference set...")
df_approved["egfr_similarity"] = df_approved["SMILES"].apply(
    lambda s: tanimoto_to_reference(s, ref_fps)
)

# Topk most similar (ймовірні repurposing кандидати)
df_repurposing_candidates = df_approved[
    df_approved["egfr_similarity"] >= 0.3
].sort_values("egfr_similarity", ascending=False)

print(f"\n🎯 Potential EGFR repurposing candidates (Tanimoto ≥ 0.3): {len(df_repurposing_candidates)}")
print(df_repurposing_candidates[["name", "chembl_id", "egfr_similarity", "QED", "MW"]].head(20))
```

### Cell 6 — ML prediction: Applied NB03 model до approved drugs

```python
# Завантажити збережену XGBoost модель з NB03 (якщо є)
MODEL_PATH = "../03_Activity_Classification/models/"
xgb_egfr = None

if os.path.exists(os.path.join(MODEL_PATH, "xgb_egfr.pkl")):
    with open(os.path.join(MODEL_PATH, "xgb_egfr.pkl"), "rb") as f:
        xgb_egfr = pickle.load(f)
    print("✅ Loaded XGBoost EGFR model from NB03")

if xgb_egfr is not None:
    print("🔄 Running EGFR activity prediction on approved drugs...")
    valid_smiles = df_approved["SMILES"].tolist()
    
    # Morgan fingerprints for all drugs
    X_approved = []
    valid_idx = []
    for i, smi in enumerate(valid_smiles):
        fp = smiles_to_morgan(smi)
        if fp is not None and len(fp) == 2048:
            X_approved.append(fp)
            valid_idx.append(i)
    
    X_approved = np.array(X_approved)
    proba = xgb_egfr.predict_proba(X_approved)[:, 1]
    pred_labels = (proba >= 0.5).astype(int)
    
    df_approved.loc[df_approved.index[valid_idx], "egfr_activity_proba"] = proba
    df_approved.loc[df_approved.index[valid_idx], "egfr_predicted_active"] = pred_labels
    
    n_active = pred_labels.sum()
    print(f"\n🎯 Predicted EGFR active among approved drugs: {n_active}/{len(valid_idx)}")
    
    # Top predicted repurposing candidates
    df_repurpose_ml = df_approved[
        (df_approved.get("egfr_predicted_active", 0) == 1) &
        (~df_approved["name"].isin(EGFR_REFERENCE.keys()))  # Виключити вже відомі
    ].sort_values("egfr_activity_proba", ascending=False)
    
    print("\nTop novel predicted EGFR-active approved drugs:")
    print(df_repurpose_ml[["name", "chembl_id", "egfr_activity_proba", "QED", "MW"]].head(10))
```

### Cell 7 — Drug-Target Interaction (DTI) Network

```python
# Побудова DTI network: nodes = drugs + targets, edges = predicted activity
G = nx.Graph()

# Targets (from NB01-NB03 pipeline)
TARGETS = ["EGFR", "NR-AR", "NR-AhR", "SR-MMP", "SR-p53", "BACE1"]

# Додати target nodes
for target in TARGETS:
    G.add_node(target, node_type="target", size=30)

# Додати drug nodes та edges з predicted активністю
for _, row in df_approved.head(50).iterrows():
    drug_name = str(row.get("name", row.get("chembl_id", "?")))[:20]
    G.add_node(drug_name, node_type="drug", size=10)
    
    # Додати edge якщо predicted active (vs EGFR)
    egfr_proba = row.get("egfr_activity_proba", 0) or 0
    if egfr_proba >= 0.6:
        G.add_edge(drug_name, "EGFR", weight=float(egfr_proba))
    
    # Tanimoto similarity edge
    sim = row.get("egfr_similarity", 0) or 0
    if sim >= 0.4:
        G.add_edge(drug_name, "EGFR", weight=float(sim), style="similarity")

print(f"DTI Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

# pyvis interactive network
from pyvis.network import Network

net = Network(height="600px", width="100%", bgcolor="#222222",
              font_color="white", notebook=True, cdn_resources="in_line")
net.from_nx(G)

# Кольорова схема: targets = червоний, drugs = синій
for node in net.nodes:
    if node["id"] in TARGETS:
        node["color"] = "#ff4444"
        node["size"] = 30
        node["font"] = {"size": 16, "color": "white"}
    else:
        node["color"] = "#4488ff"
        node["size"] = 12

net.set_physics(True)
net.toggle_physics(True)
net.save_graph("dti_network.html")
print("✅ Interactive DTI network saved → dti_network.html")
```

### Cell 8 — Integrated Repurposing Score

```python
# Комбінований score: Tanimoto + ML probability + QED
def repurposing_score(row):
    sim = float(row.get("egfr_similarity", 0) or 0)
    proba = float(row.get("egfr_activity_proba", 0) or 0)
    qed = float(row.get("QED", 0) or 0)
    # Weighted sum (можна Optuna tune)
    return 0.4 * sim + 0.4 * proba + 0.2 * qed

df_approved["repurposing_score"] = df_approved.apply(repurposing_score, axis=1)

df_top_repurpose = df_approved[
    ~df_approved["name"].isin(EGFR_REFERENCE.keys())
].sort_values("repurposing_score", ascending=False).head(20)

# Plotly: top repurposing candidates
fig = px.bar(
    df_top_repurpose,
    x="name", y="repurposing_score",
    color="repurposing_score",
    color_continuous_scale="Viridis",
    title="Top Drug Repurposing Candidates for EGFR (Integrated Score)",
    labels={"repurposing_score": "Repurposing Score (Tanimoto + ML + QED)", "name": "Drug Name"},
    template="plotly_white",
    height=500,
    hover_data=["chembl_id", "egfr_similarity", "egfr_activity_proba", "QED", "MW"],
)
fig.update_layout(xaxis_tickangle=-45)
fig.write_html("repurposing_candidates.html")
fig.show()

df_top_repurpose.to_csv("data/repurposing_candidates.csv", index=False)
print("✅ Top candidates saved → data/repurposing_candidates.csv")
```

### Cell 9 — heatmap: Chemical Similarity Matrix між топ-кандидатами

```python
top_drugs = df_top_repurpose.head(15)["SMILES"].tolist()
top_names = df_top_repurpose.head(15)["name"].tolist()

fps = []
for smi in top_drugs:
    mol = Chem.MolFromSmiles(smi)
    if mol:
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048))

# Pairwise Tanimoto matrix
n = len(fps)
sim_matrix = np.zeros((n, n))
for i in range(n):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps)
    sim_matrix[i] = sims

fig_heat = px.imshow(
    sim_matrix,
    x=top_names, y=top_names,
    color_continuous_scale="Blues",
    title="Tanimoto Similarity Matrix — Top Repurposing Candidates",
    zmin=0, zmax=1,
    template="plotly_white",
    labels=dict(color="Tanimoto"),
    height=600,
)
fig_heat.update_layout(
    xaxis_tickangle=-45,
    xaxis_tickfont_size=10,
    yaxis_tickfont_size=10,
)
fig_heat.write_html("similarity_matrix.html")
fig_heat.show()
```

### Cell 10 — MLflow logging

```python
import mlflow

mlflow.set_experiment("Drug_Repurposing_EGFR")

with mlflow.start_run(run_name="repurposing_pipeline"):
    mlflow.log_param("target", "EGFR")
    mlflow.log_param("database", "ChEMBL_MaxPhase4")
    mlflow.log_param("similarity_threshold", 0.3)
    mlflow.log_param("ml_threshold", 0.5)
    mlflow.log_param("n_approved_drugs_queried", len(df_approved))
    mlflow.log_metric("n_similarity_candidates", len(df_repurposing_candidates))
    mlflow.log_metric("n_ml_predicted_active",
                      int(df_approved.get("egfr_predicted_active", pd.Series([0])).sum()))
    mlflow.log_metric("top_repurposing_score", float(df_top_repurpose["repurposing_score"].max()))
    
    if os.path.exists("repurposing_candidates.html"):
        mlflow.log_artifact("repurposing_candidates.html")
    if os.path.exists("dti_network.html"):
        mlflow.log_artifact("dti_network.html")
    
    print("✅ MLflow experiment logged")
```

### Cell 11 — Summary

```markdown
## Drug Repurposing Summary

| Metric | Value |
|--------|-------|
| FDA-approved drugs screened | N |
| Tanimoto ≥ 0.3 to EGFR reference | N |
| ML-predicted EGFR active | N |
| Top repurposing score | X.XX |

### Key candidates identified:
- **Drug A** (similarity=0.72, p_activity=0.81): Kinase inhibitor with similar scaffold to erlotinib
- **Drug B** (similarity=0.64, p_activity=0.65): Approved for [indication], structural analog of gefitinib

### Limitations & Next steps:
1. ML model trained on ChEMBL IC50 data — potential selection bias
2. Tanimoto similarity does not capture 3D binding mode differences
3. **Next step:** Validate top candidates with molecular docking (NB06)
4. **Next step:** Wet-lab IC50 assay for top 3 candidates

### AstraZeneca relevance:
This pipeline mirrors AZ computational screening approach for target X → indication Y.
Computational prioritization reduces experimental costs by ~100x.
```

---

## Структура файлів ноутбука

```
07_Drug_Repurposing/
├── Drug_Repurposing.ipynb               ← основний ноутбук
├── README.md
└── data/
    ├── approved_drugs_chembl.csv         ← кеш ChEMBL (автогенерується)
    ├── repurposing_candidates.csv         ← топ кандидати
    ├── dti_network.html                   ← інтерактивна мережа
    ├── similarity_matrix.html
    └── repurposing_candidates.html
```

---

## Чекліст реалізації

- [ ] Створити ноутбук `Drug_Repurposing.ipynb`
- [ ] Cell 1: Imports + dependency checks
- [ ] Cell 2: Markdown intro
- [ ] Cell 3: ChEMBL download approved drugs (+ fallback dataset)
- [ ] Cell 4: Molecular properties + Lipinski filter
- [ ] Cell 5: Tanimoto similarity vs EGFR reference
- [ ] Cell 6: ML prediction використовуючи NB03 model
- [ ] Cell 7: DTI network (pyvis)
- [ ] Cell 8: Integrated repurposing score
- [ ] Cell 9: Similarity heatmap (Plotly)
- [ ] Cell 10: MLflow logging
- [ ] Cell 11: Summary
- [ ] Додати до `requirements.txt`
- [ ] Додати секцію до `README.md`
- [ ] Додати сторінку до Streamlit dashboard

---

## Додаткові ідеї (Nice-to-Have)

### Side-effect based repurposing
```python
# SIDER database: drug → known side effects
# Якщо drug X і drug Y мають схожі side effects → схожий mechanism of action → repurposing
```

### Gene expression signature matching (CMap approach)
```python
# Connectivity Map (Broad Institute): gene expression profiles
# drug A upregulates genes that disease B downregulates → repurposing候補
# Реалізувати базову версію з MSigDB gene sets
```

### Cross-target repurposing
```python
# Модель з NB01 (Tox21 12 targets) на approved drugs
# Drug X активний по NR-AhR? → перевірити чи це шкідливо або нова показання
```

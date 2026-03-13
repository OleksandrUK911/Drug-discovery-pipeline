# Drug Discovery Portfolio — TODO & Progress Tracker

> **Автор:** Олександр  
> **Останнє оновлення:** 10 березня 2026  
> **Статус портфоліо:** 7/7 ноутбуків готові ✅ | Розширення EXT-A/B/C/D ✅ | MLflow ✅ | Unit-тести ✅ | CI/CD ✅ | Інтерактивна візуалізація ✅ | Streamlit Dashboard ✅ (8 сторінок) | Educational expanders ✅ | Docker ✅ | Production FastAPI ✅ | Pre-commit ✅ | HTML Reports ✅ | Makefile ✅

---

## Загальний стан проєктів

| # | Ноутбук | Статус | Баги виправлено | Розширення | Рівень складності |
|---|---------|--------|-----------------|------------|-------------------|
| 01 | [Toxicity Prediction](./NB01_Toxicity_Prediction.md) | ✅ Готово | 4 баги | ✅ 🥇🥈🥉 реалізовані | ⭐⭐⭐ |
| 02 | [ADMET Properties](./NB02_ADMET_Properties.md) | ✅ Готово | 0 багів | ✅ 🥇🥈🥉 реалізовані | ⭐⭐⭐ |
| 03 | [Activity Classification](./NB03_Activity_Classification.md) | ✅ Готово | 1 баг | ✅ 🥇🥈🥉 реалізовані | ⭐⭐⭐ |
| 04 | [Molecule Generation](./NB04_Molecule_Generation.md) | ✅ Готово | 0 багів | ✅ 🥇🥈🥉 реалізовані | ⭐⭐⭐⭐⭐ |
| 05 | [Molecular Clustering](./NB05_Molecular_Clustering.md) | ✅ Готово | 2 баги | ✅ 🥇🥈🥉 реалізовані | ⭐⭐⭐⭐ |
| 06 | [Structure-Based Design](./NB06_Structure_Based_Design.md) | ✅ Готово | — | SBDD + Docking + ProLIF + py3Dmol | ⭐⭐⭐⭐⭐ |
| 07 | [Drug Repurposing](./NB07_Drug_Repurposing.md) | ✅ Готово | — | ChEMBL + DTI Network + ML + pyvis | ⭐⭐⭐⭐ |

---

## Що реалізовано в клітинках розширень

| Рівень | NB01 | NB02 | NB03 | NB04 | NB05 |
|--------|------|------|------|------|------|
| 🥇 EXT-A | Scaffold split + Optuna + model saving | BBBP/DeepChem + Multi-task DNN + saving | pIC50 regression + scaffold split + multi-target | ZINC250k + KL annealing + QED/SA scoring | ChEMBL API + SA score + SDF export |
| 🥈 EXT-B | Multi-task GCN + calibration + bootstrap CI | LOF applicability domain + MC-Dropout + stacking | FP benchmark + SMILES augmentation + SHAP waterfall | SELFIES + UMAP latent space + CVAE | HDBSCAN + sphere exclusion + Plotly HTML |
| 🥉 EXT-C | FastAPI REST server (`app.py`) | AttentiveFP on ESOL | ChemBERTa fine-tuning | REINVENT RL (QED reward) | NetworkX similarity graph + scaffold centrality |

---

## Виправлені баги (загальний список)

| # | Файл | Баг | Версія, що зламала | Виправлення |
|---|------|-----|---------------------|-------------|
| 1 | NB01 — Cell 4 | `use_label_encoder=False` → `TypeError` | XGBoost ≥ 2.0 | Аргумент видалено |
| 2 | NB01 — Cell 6 | `use_label_encoder=False` у SHAP-клітинці | XGBoost ≥ 2.0 | Аргумент видалено |
| 3 | NB01 — Cell 6 | Порожній `plt.subplots(1,2)` перед SHAP plots (SHAP створює власну фігуру) | — | Зайвий `fig, axes` прибрано |
| 4 | NB01 — Cell 7 | `from torch_geometric.data import DataLoader` → `ImportError` | PyG ≥ 2.0 | Перенесено до `torch_geometric.loader` з fallback |
| 5 | NB03 — Cell 4 | `use_label_encoder=False` → `TypeError` | XGBoost ≥ 2.0 | Аргумент видалено |
| 6 | NB05 — Cell 4 | `TSNE(n_iter=...)` → `TypeError` | sklearn ≥ 1.4 | Перейменовано в `max_iter` |
| 7 | NB05 — Cell 8 | CSV зберігався за абсолютним шляхом `'05_Molecular_Clustering/...'` — не існує в CWD ноутбука | — | Змінено на `'top_candidates.csv'` |

---

## Топ-пріоритети — статус виконання

### ✅ Реалізовано
- ✅ **Scaffold-aware split** (NB01 🥇, NB03 🥇) — власна реалізація через RDKit MurckoScaffold
- ✅ **Збереження моделей** — `pickle` + `torch.save` в усіх ноутбуках (NB01–NB03 🥇)
- ✅ **Optuna hyperparameter tuning** — NB01 🥇 (30 trials, 3-fold CV)
- ✅ **FastAPI REST server** — NB01 🥉 (записано `app.py`, інструкції run)
- ✅ **Multi-task DNN** — NB02 🥇 (logS + logP + BBB спільний backbone)
- ✅ **LOF Applicability Domain** — NB02 🥈
- ✅ **MC-Dropout uncertainty** — NB02 🥈
- ✅ **Stacking ensemble** — NB02 🥈
- ✅ **AttentiveFP graph model** — NB02 🥉
- ✅ **pIC50 regression** — NB03 🥇
- ✅ **Fingerprint benchmark** (Morgan/FCFP4/MACCS/AtomPair) — NB03 🥈
- ✅ **SMILES augmentation** — NB03 🥈
- ✅ **SHAP waterfall plot** — NB03 🥈
- ✅ **ChemBERTa fine-tuning** — NB03 🥉
- ✅ **ZINC250k dataset** download + setup — NB04 🥇
- ✅ **KL annealing schedule** — NB04 🥇
- ✅ **QED + SA scoring** — NB04 🥇
- ✅ **SELFIES encoding** — NB04 🥈
- ✅ **UMAP latent space visualization** — NB04 🥈
- ✅ **CVAE with property conditioning** — NB04 🥈
- ✅ **REINVENT RL** (QED reward, prior+agent) — NB04 🥉
- ✅ **ChEMBL API query** — NB05 🥇
- ✅ **SA score** calculation — NB05 🥇
- ✅ **SDF export** — NB05 🥇
- ✅ **HDBSCAN clustering** — NB05 🥈
- ✅ **Sphere exclusion** diversity picker — NB05 🥈
- ✅ **Plotly interactive HTML** — NB05 🥈
- ✅ **NetworkX similarity graph** + degree centrality — NB05 🥉
- ✅ **Scaffold centrality** analysis — NB05 🥉

### ✅ Додатково реалізовано
- ✅ **MLflow experiment tracking** — NB01 (`Toxicity_Prediction_Tox21`), NB02 (`ADMET_Properties_Prediction`), NB03 (`Activity_Classification_EGFR`) — per-model runs + per-epoch metrics
- ✅ **Unit-тести** (`tests/test_features.py`) — 51 тест для `smiles_to_morgan`, `smiles_to_descriptors`, `lipinski_filter`, `scaffold_split`, `compute_features`
- ✅ **GitHub Actions CI** (`.github/workflows/ci.yml`) — pytest + coverage на Python 3.10 / 3.11 (ubuntu-latest)
- ✅ `mlflow>=2.9`, `pytest>=7.4`, `pytest-cov>=4.1`, `plotly>=5.18`, `pyvis>=0.3`, `py3Dmol>=2.0`, `streamlit>=1.30` додані до `requirements.txt`

### ✅ Професійні покращення (PROFESSIONAL_UPGRADES.md)
- ✅ **Plotly інтерактивний heatmap + radar** — NB01 (ROC-AUC 12×2 + toxicity radar)
- ✅ **Structural alerts (mols2grid)** — NB01 (8 токсикофорних патернів + Grid/RDKit fallback)
- ✅ **Plotly parity plot** — NB02 (інтерактивний з hover SMILES + residual color)
- ✅ **Model comparison bar** — NB02 (RMSE/R²/MAE grouped bar chart)
- ✅ **Drug-likeness radar** — NB02 (Lipinski nормалізованй radar + Lipinski limit)
- ✅ **UMAP / PCA chemical space** — NB02 (colored by logS, Morgan FP projection)
- ✅ **py3Dmol 3D conformer viewer** — NB02 (ETKDG + MMFF optimisation)
- ✅ **Overlaid ROC-AUC + PR-AUC curves** — NB03 (Plotly, всі моделі на одному canvas)
- ✅ **Interactive confusion matrix** — NB03 (Plotly heatmap + threshold slider в dashboard)
- ✅ **Model scorecard grouped bar** — NB03 (ROC-AUC / PR-AUC / F1 / MCC)
- ✅ **VAE training curves** — NB04 (ELBO / Reconstruction / KL Plotly subplots)
- ✅ **Generated molecule grid + stats** — NB04 (QED/MW/LogP, RDKit grid image, validity %)
- ✅ **UMAP latent space** — NB04 (Training vs Generated, size∝QED, Plotly scatter)
- ✅ **py3Dmol 3D viewer для generated mols** — NB04 (ETKDG + spin animation)
- ✅ **3D UMAP chemical space** — NB05 (Plotly scatter_3d, color=cluster, size=QED)
- ✅ **pyvis similarity network** — NB05 (Tanimoto ≥ 0.4, cluster colors, HTML export)
- ✅ **Sunburst scaffold hierarchy** — NB05 (scaffold family → scaffold, size=count)
- ✅ **Top-15 scaffold bar chart** — NB05
- ✅ **Streamlit dashboard** — `dashboard/app.py` (6 сторінок: overview + NB01–NB05; live prediction, gauges, molecule inspector, 3D rendering, similarity network)
- ✅ **py3Dmol EXT-D interactive 3D gallery** — NB01 (4 молекули: cinnamaldehyde, aspirin, benzo[a]pyrene, epoxysqualene) + NB03 (4 EGFR drugs: gefitinib QED=0.518, erlotinib QED=0.418, osimertinib QED=0.311, lapatinib QED=0.179)
- ✅ **Latent interpolation GIF** — NB04 (Aspirin→Fluconazole + Caffeine→Ibuprofen, 10 frames each, imageio animated GIF)
- ✅ **Educational expanders в Streamlit dashboard** — 6 сторінок зі `st.expander()`: пояснення методології, метрик і науки (Tox21, ADMET, EGFR, VAE, кластеризація)
- ✅ **Docker контейнер** — `Dockerfile` + `docker-compose.yml` + `requirements_dashboard.txt` + `.dockerignore` + `.streamlit/config.toml`; одна команда `docker-compose up --build` запускає dashboard

---

## 🆕 Нові задачі — AstraZeneca рівень (10 березня 2026)

### 🔴 Критичні (повинні бути реалізовані)

#### NB06 — Structure-Based Drug Design (SBDD)
**Чому критично:** AZ Cambridge = лідер SBDD. Весь Medicinal Chemistry відділ будує молекули навколо кристалічних структур. Без цього ноутбука портфоліо неповне для фармацевтичної компанії.  
**Деталі:** [TODO/NB06_Structure_Based_Design.md](./NB06_Structure_Based_Design.md)  
**Що буде включено:**
- Download EGFR crystal structure (PDB: 1IEP + 3W2O T790M = AstraZeneca Tagrisso target)
- Protein preparation (pdbfixer / manual chain extraction)
- Ligand 3D conformer generation (RDKit ETKDG + MMFF94)
- Molecular docking (AutoDock-Vina Python API або precomputed scores)
- Protein-Ligand Interaction Fingerprints (ProLIF)
- Docking Score vs IC50 correlation (Pearson r, Plotly scatter)
- 3D py3Dmol complex visualization
- MLflow logging
- Cross-notebook зв'язок: EGFR з NB03 + топ-кандидати з NB05

#### NB07 — Drug Repurposing via Molecular Similarity & Network Analysis
**Чому важливо:** AZ відома drug repurposing стратегіями (Tagrisso, Brilinta for COVID). Показати розуміння цього підходу = пряме влучення в AZ research culture.  
**Деталі:** [TODO/NB07_Drug_Repurposing.md](./NB07_Drug_Repurposing.md)  
**Що буде включено:**
- ChEMBL query: всі FDA-approved drugs (max_phase=4)
- Tanimoto similarity vs EGFR reference set
- ML prediction: NB03 model на approved drugs → нові показання
- Drug-Target Interaction (DTI) Network (pyvis)
- Integrated repurposing score (similarity + ML + QED)
- Similarity heatmap між топ-кандидатами
- MLflow logging

### ✅ Виконано (підвищили рівень)

#### Production API — Consolidated FastAPI
**Поточний стан:** `api/` модуль повністю реалізовано (8 endpoints).  
**Мета:** Об'єднати ALL моделі (Tox21 + ADMET + Activity) в production-ready `api/` модуль.  
**Деталі:** [TODO/EXT_Production_API.md](./EXT_Production_API.md)  
**Статус:** ✅ Реалізовано — 8 endpoints: `/`, `/health`, `/predict/toxicity`, `/predict/admet`, `/predict/activity`, `/predict/full`, `/batch/predict`, `/molecule/info`

#### Pre-commit Hooks
**Файл:** `.pre-commit-config.yaml`  
**Що включає:** black (форматування), flake8 (linting), isort (imports), trailing-whitespace  
**Статус:** ✅ Реалізовано — `.pre-commit-config.yaml` (black, isort, flake8, trailing-whitespace, end-of-file-fixer)

### ✅ Вже виконано (підтверджено)
- ✅ **GitHub Actions CI/CD** — `.github/workflows/ci.yml` (pytest + coverage + Codecov, Python 3.10/3.11/3.12)
- ✅ **MLflow** — NB01, NB02, NB03 (per-model runs + per-epoch metrics)
- ✅ **FastAPI (Tox21 only)** — `01_Toxicity_Prediction/app.py`
- ✅ **Docker** — `Dockerfile` + `docker-compose.yml`
- ✅ **96 unit-тестів** — `tests/test_features.py` (51) + `tests/test_api.py` (45)

---

## Детальні файли

- 📄 [NB01_Toxicity_Prediction.md](./NB01_Toxicity_Prediction.md)
- 📄 [NB02_ADMET_Properties.md](./NB02_ADMET_Properties.md)
- 📄 [NB03_Activity_Classification.md](./NB03_Activity_Classification.md)
- 📄 [NB04_Molecule_Generation.md](./NB04_Molecule_Generation.md)
- 📄 [NB05_Molecular_Clustering.md](./NB05_Molecular_Clustering.md)
- 📄 [NB06_Structure_Based_Design.md](./NB06_Structure_Based_Design.md) 🆕 SBDD + Docking
- 📄 [NB07_Drug_Repurposing.md](./NB07_Drug_Repurposing.md) 🆕 Drug Repurposing
- 📄 [EXT_Production_API.md](./EXT_Production_API.md) 🆕 Consolidated FastAPI
- 📄 [PROFESSIONAL_UPGRADES.md](./PROFESSIONAL_UPGRADES.md) — Детальний аналіз і план проф. покращень (реалізовано)

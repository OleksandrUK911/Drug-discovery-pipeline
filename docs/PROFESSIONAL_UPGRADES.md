# Drug Discovery Portfolio — Professional Upgrades & Visualization Analysis

> **Дата аналізу:** 3 березня 2026  
> **Мета:** Зробити портфоліо максимально презентабельним і конкурентним на рівні industry-grade рішень  
> **Статус:** Аналіз виконано для всіх 5 ноутбуків + кросс-проєктних покращень

---

## 🔍 Загальна оцінка поточного стану

| # | Ноутбук | Поточна візуалізація | Рівень презентабельності |
|---|---------|---------------------|--------------------------|
| 01 | Toxicity Prediction | matplotlib/seaborn статичні PNG, SHAP bar/beeswarm | ⭐⭐⭐ Добре, але статично |
| 02 | ADMET Properties | Parity plot, residual, correlation heatmap (matplotlib) | ⭐⭐⭐ Добре, але немає інтерактиву |
| 03 | Activity Classification | SHAP bar, ROC-AUC, threshold F1 curve | ⭐⭐⭐ Добре, але беземоцій |
| 04 | Molecule Generation | VAE training curves (matplotlib), без структур молекул | ⭐⭐ Функціонально, але не WOW |
| 05 | Molecular Clustering | mols2grid + Plotly HTML scatter (частково) | ⭐⭐⭐⭐ Найближче до production |

**Головна проблема:** Всі ноутбуки використовують переважно `matplotlib` → статичні зображення, що виглядають як навчальний код, а не реальне дослідження.  
**Рішення:** Перейти на `Plotly` / `Bokeh` для інтерактиву + `py3Dmol` / `nglview` для 3D + `Streamlit` / `Dash` для дашборду + `mols2grid` скрізь де є молекули.

---

## 📦 Нові залежності (узагальнено)

```bash
pip install plotly kaleido          # Interactive + статичний експорт
pip install bokeh                   # Альтернативний інтерактив
pip install streamlit               # Web dashboard
pip install py3Dmol                 # 3D molecule viewer (inline Jupyter)
pip install pyvis                   # Інтерактивна network graph
pip install panel                   # Panel dashboard з ноутбука
pip install itables                 # Інтерактивні таблиці (DT.js)
pip install nbconvert               # HTML export ноутбуків
pip install quarto                  # Publication-quality reports
pip install ipywidgets              # Interactive sliders (вже є у requirements)
pip install cairosvg                # SVG → PNG export молекул (RDKit)
```

---

## 📗 NB01 — Toxicity Prediction: Конкретні апгрейди

### ❌ Що зараз не так
- Bar chart з 12 AUC задач — статичний, неможливо навести мишку щоб побачити точне значення
- SHAP beeswarm plot — є, але не прив'язаний до структур молекул
- GCN training curve — 2 графіки matplotlib, виглядають як tutorial
- Немає жодного відображення реальних молекул (структур)
- Confusion matrix / ROC-AUC відсутні взагалі

### ✅ Що додати

#### 1. Інтерактивний Plotly heatmap — 12 задач × 3 моделі
```python
import plotly.graph_objects as go

model_cols = ['RF_ROC_AUC', 'XGB_ROC_AUC', 'GCN_ROC_AUC']
fig = go.Figure(data=go.Heatmap(
    z=results_df[model_cols].values.T,
    x=results_df['Task'].tolist(),
    y=['Random Forest', 'XGBoost', 'GCN'],
    colorscale='RdYlGn', zmin=0.5, zmax=1.0,
    text=results_df[model_cols].values.T.round(3),
    texttemplate="%{text}",
    hovertemplate="Task: %{x}<br>Model: %{y}<br>AUC: %{z:.4f}",
))
fig.update_layout(title='Tox21 — ROC-AUC по всіх задачах і моделях',
                  height=350, width=900)
fig.write_html('tox21_auc_heatmap.html')
fig.show()
```
**Результат:** Відкривається HTML файл — рекрутер може клікати по кожній клітинці.

#### 2. Radar chart — профіль токсичності молекули
```python
# Радар-діаграма: ймовірності токсичності по 12 задачах для конкретної молекули
from plotly.graph_objects import Figure, Scatterpolar

def toxicity_radar(smiles, xgb_models, tasks):
    from rdkit.Chem import AllChem, Descriptors
    mol = Chem.MolFromSmiles(smiles)
    fp  = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048)
    ...
    fig = Figure()
    fig.add_trace(Scatterpolar(r=probas, theta=tasks, fill='toself'))
    fig.update_layout(title=f'Toxicity Profile: {smiles}')
    fig.show()
```
**Результат:** Для довільної молекули — «павутиноподібний» графік по 12 токсичним задачам. Виглядає як реальний фармацевтичний звіт.

#### 3. mols2grid — grid молекул з найвищим ризиком токсичності
```python
import mols2grid

# Топ-20 найтоксичніших молекул з датасету
toxic_mask = df_all[TOX21_TASKS].sum(axis=1, skipna=True) >= 4
df_toxic = df_all[toxic_mask].head(30).copy()
mols2grid.display(df_toxic, smiles_col='smiles',
                  subset=['smiles'] + TOX21_TASKS[:4],
                  tooltip=TOX21_TASKS,
                  style={'background-color': '#fff3f3'})
```
**Результат:** Клікабельний grid з 2D-структурами молекул і підсвіченими властивостями.

#### 4. RDKit — підсвічування токсичних субструктур
```python
from rdkit.Chem import Draw
from rdkit.Chem import FragmentMatcher

# Групи Алерт (Michael acceptors, epoxides, etc.)
STRUCTURAL_ALERTS = {
    'Epoxide':          '[OX2]1[CX4][CX4]1',
    'Michael acceptor': 'C=CC(=O)',
    'Quinone':          'O=C1C=CC(=O)C=C1',
    ...
}
# Малювати молекулу з підсвіченими небезпечними фрагментами
```
**Результат:** Виглядає як справжній toxicology report.

#### 5. Plotly — анімована крива навчання GCN
```python
fig = go.Figure()
fig.add_trace(go.Scatter(x=list(range(1, EPOCHS+1)), y=train_losses,
              mode='lines+markers', name='Train Loss', line=dict(color='steelblue')))
fig.add_trace(go.Scatter(x=list(range(1, EPOCHS+1)), y=test_aucs,
              mode='lines+markers', name='Test AUC', yaxis='y2', line=dict(color='green')))
fig.update_layout(title='GCN Training — SR-MMP', yaxis2=dict(overlaying='y', side='right'))
fig.write_html('gcn_training_interactive.html')
```

---

## 📘 NB02 — ADMET Properties: Конкретні апгрейди

### ❌ Що зараз не так
- Parity plot — статичний matplotlib, неможливо побачити яка саме молекула є outlier'ом
- Correlation heatmap — корисний але нудний
- Немає drug-likeness профілю у вигляді Radar chart (Lipinski + Veber + TPSA)
- Chemical space explorer відсутній (всі молекули в одному статичному scatter'і)

### ✅ Що додати

#### 1. Plotly parity plot з hover по SMILES та структурою
```python
import plotly.express as px
from rdkit.Chem import Draw
import base64, io

df_parity = pd.DataFrame({'True': y_te, 'Predicted': y_pred,
                           'SMILES': smiles_test, 'Residual': y_te - y_pred})

fig = px.scatter(df_parity, x='True', y='Predicted',
                 color='Residual', color_continuous_scale='RdBu',
                 hover_data=['SMILES'],
                 labels={'True': 'Actual logS', 'Predicted': 'Predicted logS'},
                 title='XGBoost ESOL Parity Plot — Hover для SMILES')
fig.add_shape(type='line', line=dict(color='red', dash='dash'), x0=-12, y0=-12, x1=2, y1=2)
fig.write_html('parity_interactive.html')
```
**Ефект:** Рекрутер може навести мишку на точку і побачити SMILES + помилку.

#### 2. Drug-likeness Radar chart (Lipinski spider)
```python
# Для кожного predicted molecule — radar of: MW/500, logP/5, HBD/5, HBA/10, TPSA/140
# Показати "drug-likeness space" vs Lipinski limits

categories = ['MW/500', 'logP/5', 'HBD/5', 'HBA/10', 'TPSA/140']
for i, smi in enumerate(top_smiles[:5]):
    mol = Chem.MolFromSmiles(smi)
    vals = [Descriptors.MolWt(mol)/500, Descriptors.MolLogP(mol)/5, ...]
    fig.add_trace(Scatterpolar(r=vals, theta=categories, fill='toself', name=f'Mol {i}'))
```
**Ефект:** Класичний фармацевтичний «radar plot» виглядає як slides з Big Pharma.

#### 3. Interactive Chemical Space explorer (UMAP + ADMET color)
```python
import umap
reducer = umap.UMAP(n_components=2, random_state=42)
X_2d = reducer.fit_transform(np.nan_to_num(X_esol))
df_space = pd.DataFrame(X_2d, columns=['UMAP_1', 'UMAP_2'])
df_space['logS'] = y_esol
df_space['SMILES'] = df_esol['smiles']

fig = px.scatter(df_space, x='UMAP_1', y='UMAP_2', color='logS',
                 color_continuous_scale='plasma', hover_data=['SMILES'],
                 title='ADMET Chemical Space (UMAP) — colored by logS')
fig.write_html('admet_chemical_space_interactive.html')
```
**Ефект:** Інтерактивна карта хімічного простору. Рекрутер «гуляє» зловити окремі молекули.

#### 4. itables — інтерактивна таблиця результатів моделей
```python
from itables import show as itable_show
itable_show(esol_df.style.background_gradient(cmap='RdYlGn_r', subset=['RMSE'])
                         .background_gradient(cmap='RdYlGn', subset=['R2']),
            caption="ESOL Model Comparison")
```
**Ефект:** Таблиця сортується, фільтрується прямо в ноутбуці.

---

## 📙 NB03 — Activity Classification: Конкретні апгрейди

### ❌ Що зараз не так
- Confusion matrix відсутня як візуалізація (тільки текстові метрики)
- ROC-AUC криві для різних моделей — не накладаються для порівняння
- SHAP plots не прив'язані до 2D структури молекул
- Немає відображення топ-активних і топ-неактивних молекул
- Threshold optimization — статичний matplotlib

### ✅ Що додати

#### 1. Накладені ROC-AUC + PR-AUC криві (Plotly)
```python
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, precision_recall_curve

fig_roc = go.Figure()
fig_pr  = go.Figure()

for name, r in results.items():
    fpr, tpr, _ = roc_curve(y_te, r['proba'])
    p, rec, _   = precision_recall_curve(y_te, r['proba'])
    fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f"{name} (AUC={r['ROC-AUC']:.3f})"))
    fig_pr.add_trace(go.Scatter(x=rec, y=p,   name=f"{name} (AP={r['PR-AUC']:.3f})"))

fig_roc.add_shape(type='line', x0=0, y0=0, x1=1, y1=1, line=dict(dash='dash', color='gray'))
fig_roc.update_layout(title='ROC-AUC — All Models (EGFR Activity)', 
                       xaxis_title='FPR', yaxis_title='TPR', width=700, height=500)
fig_roc.write_html('roc_auc_all_models.html')
```
**Ефект:** Порівняння 4 моделей на одному графіку, кожна крива клікабельна.

#### 2. Annotated confusion matrix (Plotly heatmap)
```python
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_te, (results['XGBoost']['proba'] >= best_t).astype(int))
fig = go.Figure(data=go.Heatmap(
    z=cm, colorscale='Blues',
    x=['Predicted Inactive', 'Predicted Active'],
    y=['True Inactive', 'True Active'],
    text=cm, texttemplate="%{text}",
    hovertemplate="Count: %{z}",
))
fig.update_layout(title=f'Confusion Matrix — XGBoost (threshold={best_t:.2f})')
fig.write_html('confusion_matrix_interactive.html')
```

#### 3. mols2grid — активні vs неактивні молекули
```python
df_model['Predicted_Prob'] = results['XGBoost']['proba']
df_model['Predicted_Class'] = (df_model['Predicted_Prob'] >= best_t).astype(int)

# Top 20 most confidently predicted active
top_active = df_model.nlargest(20, 'Predicted_Prob')
mols2grid.display(top_active, smiles_col='canonical_smiles',
                  subset=['canonical_smiles', 'Predicted_Prob', 'active'],
                  tooltip=['pIC50', 'MW', 'LogP'],
                  style={'Predicted_Prob': lambda v: 'color: green' if v > 0.8 else ''})
```

#### 4. SHAP Force plot для окремої молекули
```python
# Показати для конкретного compound чому модель вирішила active/inactive
shap.initjs()
idx_of_interest = np.argmax(results['XGBoost']['proba'])  # найбільш активна
shap.force_plot(explainer.expected_value, shap_values[idx_of_interest],
                feature_names=FEAT_NAMES)
```
**Ефект:** Waterfall / force plot — ABSOLUTELY standard в drug discovery ML презентаціях.

---

## 📕 NB04 — Molecule Generation (VAE): Конкретні апгрейди

### ❌ Що зараз не так
- VAE training curves — 3 статичних matplotlib графіки
- Згенеровані молекули відображаються тільки як SMILES рядки (print)
- Latent space — немає жодної 2D візуалізації (відсутня UMAP embedding)
- QED / SA distributions — базові гістограми без порівняння training vs generated
- RL reward curve — лінійний matplotlib

### ✅ Що додати

#### 1. RDKit molecule grid — generated molecules з властивостями
```python
from rdkit.Chem import Draw
from IPython.display import display, Image

# Показати топ-N згенерованих молекул по QED
top_mols = sorted(valid_generated, key=lambda s: QED.qed(Chem.MolFromSmiles(s)), reverse=True)[:20]
mols_rdkit = [Chem.MolFromSmiles(s) for s in top_mols]

# Підписи: QED + MW
legends = [f"QED={QED.qed(m):.2f}\nMW={Descriptors.MolWt(m):.0f}" for m in mols_rdkit]
img = Draw.MolsToGridImage(mols_rdkit, molsPerRow=5, subImgSize=(300,250),
                            legends=legends, returnPNG=False)
img.save('generated_molecules_grid.png', dpi=150)
display(img)
```
**Ефект:** Grid 4×5 молекул з підписами QED і MW — виглядає як Nature paper figure.

#### 2. UMAP 2D latent space (interactive Plotly)
```python
# Encode all training + generated molecules → get latent z
vae.eval()
with torch.no_grad():
    z_train = [vae.encode(torch.tensor(encode_smiles(s)).unsqueeze(0).to(DEVICE))[0]
               .squeeze().cpu().numpy() for s in ALL_SMILES[:200]]

z_all = np.vstack(z_train + z_generated_latent)
labels = ['Training']*200 + ['Generated']*len(z_generated_latent)
qeds   = [QED.qed(Chem.MolFromSmiles(s)) for s in ...]

reducer = umap.UMAP(n_components=2, random_state=42)
z_2d = reducer.fit_transform(z_all)

fig = px.scatter(x=z_2d[:,0], y=z_2d[:,1], color=labels,
                 hover_data={'QED': qeds, 'SMILES': all_smiles},
                 title='VAE Latent Space (UMAP) — Training vs Generated',
                 color_discrete_map={'Training': 'steelblue', 'Generated': 'tomato'})
fig.write_html('latent_space_umap.html')
```
**Ефект:** Інтерактивна карта латентного простору — ключова figure для будь-якої роботи з generative models.

#### 3. Plotly scatter: QED vs SA score (Training vs Generated)
```python
# Порівняти розподіли властивостей: чи генерує VAE drug-like молекули?
fig = go.Figure()
fig.add_trace(go.Scatter(x=train_qeds, y=train_sa,
              mode='markers', name='Training', opacity=0.5,
              marker=dict(color='steelblue', size=6)))
fig.add_trace(go.Scatter(x=gen_qeds, y=gen_sa,
              mode='markers', name='Generated', opacity=0.7,
              marker=dict(color='tomato', size=8, symbol='star')))
# Зони: "Drug-like region"
fig.add_shape(type='rect', x0=0.6, y0=1, x1=1.0, y1=4,
               fillcolor='green', opacity=0.1, line=dict(width=0))
fig.update_layout(title='QED vs SA Score — Training vs Generated Molecules',
                  xaxis_title='QED (higher = better)',
                  yaxis_title='SA Score (lower = more synthesisable, 1=easy)')
fig.write_html('qed_vs_sa_interactive.html')
```

#### 4. Latent space interpolation — анімація між двома молекулами
```python
# Interpolate між двома латентними точками → показати як молекула "трансформується"
import imageio, io

def interpolate_smiles(smiles_a, smiles_b, n_steps=10):
    za = encode_to_latent(smiles_a)
    zb = encode_to_latent(smiles_b)
    frames = []
    for t in np.linspace(0, 1, n_steps):
        z_interp = (1-t)*za + t*zb
        smi = decode_latent(z_interp)
        mol = Chem.MolFromSmiles(smi)
        if mol:
            img = Draw.MolToImage(mol, size=(300,200))
            frames.append(np.array(img))
    imageio.mimsave('latent_interpolation.gif', frames, fps=3)

# Показати як animted GIF у ноутбуці
from IPython.display import HTML
HTML('<img src="latent_interpolation.gif">')
```
**Ефект:** GIF інтерполяції між двома молекулами — АБСОЛЮТНИЙ WOW-ефект для портфоліо.

#### 5. py3Dmol — 3D конформер згенерованої молекули
```python
import py3Dmol
from rdkit.Chem import AllChem

def show_3d(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    AllChem.MMFFOptimizeMolecule(mol)
    mb = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=500, height=400)
    view.addModel(mb, 'mol')
    view.setStyle({'stick': {'colorscheme': 'cyanCarbon'}})
    view.zoomTo()
    return view.show()

# Показати топ-1 згенеровану молекулу у 3D
show_3d(top_mols[0])
```
**Ефект:** Інтерактивна 3D молекула у ноутбуці — обертається мишкою. Виглядає як справжній drug design tool.

---

## 📒 NB05 — Molecular Clustering: Конкретні апгрейди

### ❌ Що зараз не так
- mols2grid є, але виводить тільки 50 молекул
- Plotly scatter є, але без hover-структур молекул
- NetworkX графік — статичний, неможливо перетягувати вузли
- Scaffold pie chart — непрофесійний вигляд
- UMAP 2D є, але 3D відсутній

### ✅ Що додати

#### 1. Plotly 3D UMAP хімічного простору
```python
reducer_3d = umap.UMAP(n_components=3, random_state=42)
X_3d = reducer_3d.fit_transform(X_cluster)

fig = px.scatter_3d(
    x=X_3d[:,0], y=X_3d[:,1], z=X_3d[:,2],
    color=df_mol['KMeans_Cluster'].astype(str),
    size=df_mol['QED'],
    hover_data={'SMILES': df_mol['SMILES'], 'QED': df_mol['QED'].round(3),
                'DrugScore': df_mol['DrugScore'].round(3)},
    title='3D Chemical Space (UMAP) — Clusters colored by type, size by QED',
    opacity=0.7,
)
fig.write_html('chemical_space_3d.html')
```
**Ефект:** Інтерактивний 3D простір — обертається мишкою. Колір = кластер, розмір = QED.

#### 2. pyvis — інтерактивна граф схожості молекул
```python
from pyvis.network import Network

net = Network(height='600px', width='100%', notebook=True, cdn_resources='in_line')
net.set_options('{"physics": {"stabilization": {"iterations": 100}}}')

# Додати вузли (молекули) і ребра (схожість > поріг)
for idx, row in df_candidates.iterrows():
    net.add_node(idx, label=f"M{idx}", title=f"SMILES: {row['SMILES']}\nQED: {row['QED']:.2f}",
                 color='#3498db' if row['KMeans_Cluster'] == 0 else '#e74c3c', size=row['QED']*30)

for i, j in similarity_edges:  # де tanimoto > 0.4
    net.add_edge(i, j, weight=float(tanimoto[i,j]), title=f"Tanimoto: {tanimoto[i,j]:.2f}")

net.show('similarity_network.html')
```
**Ефект:** Інтерактивна граф-мережа — можна перетягувати вузли, клікати на зв'язки.

#### 3. Plotly sunburst — ієрархія скаффолдів
```python
# Scaffold → Sub-scaffold → Molecule ієрархія
fig = px.sunburst(
    df_scaffold_tree,
    path=['scaffold_family', 'scaffold', 'SMILES'],
    values='DrugScore',
    color='QED',
    color_continuous_scale='YlGn',
    title='Scaffold Hierarchy — Sunburst Chart',
)
fig.write_html('scaffold_sunburst.html')
```
**Ефект:** Сонячний ієрархічний chart виглядає як щось із фармацевтичного звіту.

#### 4. mols2grid — повна інтерактивна бібліотека
```python
# Повний mols2grid з фільтрами за кластером і DrugScore
mols2grid.display(
    df_mol[df_mol['DrugScore'] > 0.5],  # тільки drug-like
    smiles_col='SMILES',
    subset=['SMILES', 'MW', 'LogP', 'QED', 'DrugScore', 'KMeans_Cluster'],
    tooltip=['TPSA', 'HBD', 'HBA', 'RotBonds', 'Scaffold'],
    selection=True,        # checkbox selection
    n_rows=8, n_cols=6,
    transform={'QED': lambda v: round(v, 3)},
)
```

#### 5. Bokeh property correlation explorer
```python
from bokeh.plotting import figure, output_notebook, show
from bokeh.models import HoverTool, ColumnDataSource, ColorBar
from bokeh.transform import linear_cmap
from bokeh.palettes import Viridis256

source = ColumnDataSource(df_mol)
mapper = linear_cmap(field_name='QED', palette=Viridis256, low=0, high=1)
p = figure(title='MW vs LogP — Colored by QED', width=700, height=500,
           tools='pan,wheel_zoom,box_select,reset')
p.circle('MW', 'LogP', source=source, color=mapper, size=8, alpha=0.7)
p.add_tools(HoverTool(tooltips=[('SMILES','@SMILES'), ('QED','@QED'),
                                  ('Cluster','@KMeans_Cluster')]))
color_bar = ColorBar(color_mapper=mapper['transform'], width=8)
p.add_layout(color_bar, 'right')
output_notebook(); show(p)
```

---

## 🌐 Cross-Project: Streamlit Dashboard (НАЙВАЖЛИВІШИЙ АПГРЕЙД)

### Опис
Один `app.py` файл, який об'єднує всі 5 проєктів у web-дашборд.  
Рекрутер відкриває URL і бачить весь портфоліо без запуску Jupyter.

### Структура `dashboard/app.py`
```python
import streamlit as st
import pickle, pandas as pd, numpy as np, plotly.express as px
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, QED, AllChem

st.set_page_config(page_title='Drug Discovery ML Portfolio', layout='wide',
                   page_icon='💊')

st.sidebar.title('💊 Drug Discovery ML')
page = st.sidebar.radio('Project', [
    '🏠 Overview',
    '01 — Toxicity Prediction',
    '02 — ADMET Properties',
    '03 — Activity Classification',
    '04 — Molecule Generation',
    '05 — Molecular Clustering',
])

if page == '01 — Toxicity Prediction':
    st.title('Toxicity Prediction (Tox21)')
    smiles_input = st.text_input('Enter SMILES:', value='CC(=O)Oc1ccccc1C(=O)O')
    if st.button('Predict Toxicity'):
        # Load models, featurize, predict all 12 tasks
        # Show radar chart + mols2grid structure
        ...

elif page == '02 — ADMET Properties':
    st.title('ADMET Property Prediction')
    smiles_input = st.text_input('Enter SMILES:')
    if st.button('Predict ADMET'):
        # Show logS, logP, BBB pred + drug-likeness radar
        ...
```

### Команда запуску
```bash
streamlit run dashboard/app.py
# Відкривається на http://localhost:8501
```

### Що показувати у кожному розділі
| Розділ | Компоненти |
|--------|-----------|
| Overview | Project cards, metric summary, portfolio stats |
| NB01 | SMILES input → toxicity radar + molecule 2D + tox heatmap |
| NB02 | SMILES input → ADMET predictions + drug-likeness radar |
| NB03 | SMILES input → activity probability + SHAP force plot |
| NB04 | Sample/interpolate VAE → generated molecules grid |
| NB05 | Upload CSV → cluster analysis + interactive scatter |

**Де розмістити:** `dashboard/app.py` + `dashboard/requirements_dashboard.txt`

---

## 📄 Cross-Project: Quarto / nbconvert HTML звіти

### Проблема
Ноутбуки — це чудово для розробки, але рекрутер не завжди запустить Jupyter.  
HTML export дозволяє переглянути ноутбук як веб-сторінку зі всіма виходами без Python.

### Рішення 1: nbconvert (одноразовий скрипт)
```bash
# Конвертувати кожен ноутбук у HTML
jupyter nbconvert --to html "01_Toxicity_Prediction/Toxicity_Prediction.ipynb" \
    --output-dir="reports/" --template=lab --no-input
# --no-input = сховати код, показати тільки виходи/plots

# Або показати і код, і outputs:
jupyter nbconvert --to html "01_Toxicity_Prediction/Toxicity_Prediction.ipynb" \
    --output-dir="reports/"
```

### Рішення 2: Makefile для всіх звітів
```makefile
# Makefile
reports:
	@for nb in 01 02 03 04 05; do \
		jupyter nbconvert --to html \
			"0$$nb_*/0$$nb_*.ipynb" \
			--output-dir="reports/" --template=lab; \
	done
```

### Рішення 3: Quarto (найпрофесійніше)
```yaml
# _quarto.yml
project:
  type: website
  title: "Drug Discovery ML Portfolio"
format:
  html:
    theme: cosmo
    code-fold: true      # Згортати код за замовчуванням
    toc: true
    number-sections: true
execute:
  cache: true            # Кешувати результати
```
**Команда:** `quarto render` → генерує повний static site.  
**Де розмістити:** GitHub Pages або Netlify (безкоштовно).

---

## 📊 Cross-Project: Weights & Biases (wandb) — АЛЬТЕРНАТИВА MLflow

### Чому wandb > MLflow для портфоліо
| Feature | MLflow | W&B |
|---------|--------|-----|
| Хостинг | localhost (треба запускати сервер) | wandb.ai (хмарний, безкоштовно) |
| Shareable URL | ❌ Недоступне зовні | ✅ Публічне посилання для рекрутера |
| Model registry | Базовий | Просунутий |
| Автоматичні system metrics | Ні | CPU/GPU/RAM автоматично |
| Plots в UI | Базові | Інтерактивні, кастомні |

### Приклад інтеграції
```python
import wandb
wandb.init(project='drug-discovery-tox21', entity='your-username',
           config={'n_estimators': 200, 'max_depth': 6, 'fingerprint_bits': 2048})

for task, auc in xgb_results.items():
    wandb.log({f'roc_auc/{task}': auc})

# Після навчання — зберегти artifact
artifact = wandb.Artifact('tox21-xgboost', type='model')
artifact.add_file('models/xgb_SR_MMP.pkl')
wandb.log_artifact(artifact)
wandb.finish()
```
**Результат:** Публічне посилання `https://wandb.ai/your-name/drug-discovery-tox21`  
→ Вставити у LinkedIn / README / резюме.

---

## 🧬 Cross-Project: py3Dmol скрізь де є молекули

### Що це
py3Dmol — JavaScript-based 3D molecular viewer, що рендерить прямо у Jupyter notebook.  
Не потребує встановлення зовнішніх програм (Pymol, Chimera).

### Де додати (пріоритет)
| Ноутбук | Де використати |
|---------|---------------|
| NB01 | Показати 3D конформер найтоксичнішої молекули |
| NB02 | Показати 3D конформер молекули з найкращими ADMET |
| NB03 | Показати 3D активний інгібітор EGFR |
| NB04 | Показати 3D найкращу generated molecule |
| NB05 | Показати 3D центроїд кожного кластера |

### Шаблон (універсальний, в кожен ноутбук)
```python
import py3Dmol
from rdkit.Chem import AllChem

def view_molecule_3d(smiles, style='stick', width=500, height=400, colorscheme='cyanCarbon'):
    """Відобразити молекулу у 3D всередині Jupyter."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}"); return
    mol = Chem.AddHs(mol)
    result = AllChem.EmbedMolecule(mol, AllChem.ETKDGv3())
    if result != 0:
        AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    molblock = Chem.MolToMolBlock(mol)
    view = py3Dmol.view(width=width, height=height)
    view.addModel(molblock, 'mol')
    view.setStyle({style: {'colorscheme': colorscheme}})
    view.setBackgroundColor('#1a1a2e')    # темний фон виглядає круто
    view.zoomTo()
    view.show()

# Використання:
view_molecule_3d("CC(=O)Oc1ccccc1C(=O)O")  # Aspirin у 3D
```

---

## 📌 Пріоритизований план дій

| Пріоритет | Завдання | Зусилля | Impact |
|-----------|----------|---------|--------|
| 🔴 P1 | Streamlit dashboard (`dashboard/app.py`) | 4-6 год | ⭐⭐⭐⭐⭐ |
| 🔴 P1 | py3Dmol у NB02, NB03, NB04 (3D конформери) | 1 год | ⭐⭐⭐⭐⭐ |
| 🔴 P1 | RDKit molecule grid (NB01, NB04) — generated/toxic molecules | 1 год | ⭐⭐⭐⭐⭐ |
| 🟠 P2 | Plotly interactive parity plot (NB02) | 30 хв | ⭐⭐⭐⭐ |
| 🟠 P2 | Plotly overlay ROC-AUC + PR-AUC (NB03) | 30 хв | ⭐⭐⭐⭐ |
| 🟠 P2 | UMAP latent space Plotly (NB04) | 1 год | ⭐⭐⭐⭐⭐ |
| 🟠 P2 | Radar (toxicity profile, drug-likeness) (NB01, NB02) | 45 хв | ⭐⭐⭐⭐ |
| 🟠 P2 | Plotly 3D UMAP chemical space (NB05) | 30 хв | ⭐⭐⭐⭐ |
| 🟡 P3 | pyvis similarity network (NB05) | 1 год | ⭐⭐⭐⭐ |
| 🟡 P3 | Latent space interpolation GIF (NB04) | 1.5 год | ⭐⭐⭐⭐⭐ |
| 🟡 P3 | Weights & Biases (wandb) інтеграція | 1 год | ⭐⭐⭐⭐ |
| 🟡 P3 | Sunburst scaffold chart (NB05) | 30 хв | ⭐⭐⭐ |
| 🟢 P4 | nbconvert HTML reports для всіх 5 ноутбуків | 30 хв | ⭐⭐⭐⭐ |
| 🟢 P4 | itables інтерактивні таблиці | 20 хв | ⭐⭐⭐ |
| 🟢 P4 | Quarto static site (GitHub Pages) | 2 год | ⭐⭐⭐⭐⭐ |
| 🟢 P4 | SHAP force_plot для окремих молекул (NB03) | 20 хв | ⭐⭐⭐ |
| 🟢 P4 | Annotated confusion matrix Plotly (NB03) | 20 хв | ⭐⭐⭐ |
| 🟢 P4 | RDKit structural alerts highlights (NB01) | 45 хв | ⭐⭐⭐⭐ |

---

## 🗺️ Як це виглядатиме після всіх змін

```
Drug Discovery ML Portfolio/
├── 01_Toxicity_Prediction/
│   ├── Toxicity_Prediction.ipynb          ← + Plotly heatmap, radar, mols2grid
│   ├── tox21_auc_heatmap.html             ← інтерактивний
│   ├── toxicity_radar_aspirin.html        ← приклад профілю
│   └── app.py                             ← FastAPI (вже є)
├── 02_ADMET_Properties/
│   ├── ADMET_Properties_Prediction.ipynb  ← + 3D py3Dmol, parity Plotly, radar
│   ├── parity_interactive.html
│   └── admet_chemical_space_interactive.html
├── 03_Activity_Classification/
│   ├── Activity_Classification.ipynb      ← + ROC overlay, confusion matrix, mols2grid
│   ├── roc_auc_all_models.html
│   └── confusion_matrix_interactive.html
├── 04_Molecule_Generation/
│   ├── Molecule_Generation_VAE.ipynb      ← + grid, UMAP latent, 3D, GIF
│   ├── generated_molecules_grid.png
│   ├── latent_space_umap.html
│   └── latent_interpolation.gif           ← 🔥 WOW-фактор
├── 05_Molecular_Clustering/
│   ├── Molecular_Clustering.ipynb         ← + 3D UMAP, pyvis network, sunburst
│   ├── chemical_space_3d.html
│   ├── similarity_network.html
│   └── scaffold_sunburst.html
├── dashboard/
│   └── app.py                             ← 🔥 Streamlit — ВЕСЬ портфоліо в одному UI
├── reports/
│   ├── Toxicity_Prediction.html           ← nbconvert
│   ├── ADMET_Properties.html
│   ├── Activity_Classification.html
│   ├── Molecule_Generation.html
│   └── Molecular_Clustering.html
└── requirements.txt                       ← + plotly, py3Dmol, pyvis, streamlit
```

---

## 💡 Ключові висновки

1. **Найбільший ROI** — Streamlit dashboard (`dashboard/app.py`) + py3Dmol 3D viewer.  
   Одна демонстрація в браузері > 5 ноутбуків для рекрутера.

2. **Найефектніша «вау-feature»** — latent space interpolation GIF (NB04).  
   Ніхто більше не покаже GIF де молекула поступово трансформується у іншу.

3. **Найпрактичніше для індустрії** — mols2grid скрізь де є молекули + Plotly parity/ROC plots.  
   Кожна велика фармацевтична компанія використовує саме такі підходи.

4. **Для LinkedIn / GitHub README** — додати GIF або відео демо Streamlit dashboard.  
   Це одразу виділить серед інших ML портфоліо.

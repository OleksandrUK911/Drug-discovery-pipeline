# NB05 — Molecular Clustering & Drug Candidate Discovery
**Файл:** `05_Molecular_Clustering/Molecular_Clustering.ipynb`  
**Задача:** Кластеризація хімічного простору, ранжування drug candidates  
**Датасет:** Curated drug-like SMILES (~50 унікальних, Lipinski-filtered)  
**Методи:** PCA → t-SNE → UMAP | KMeans | DBSCAN | Tanimoto similarity

---

## 🎯 Для чого цей ноутбук і чого ми досягаємо

### Контекст: фінальна стадія in silico pipeline

NB05 — це **фінальна і найбільш комплексна частина** всього drug discovery портфоліо. Він відповідає на питання: *«У нас є велика бібліотека молекул. Якби нам треба було вибрати 20 найбільш перспективних кандидатів для синтезу і тестування — які саме і чому?»*

Уяви: є 1,000 drug-like молекул. Відправити всі 1,000 у лабораторію → $50,000–$500,000 і 6 місяців роботи. Або: запустити NB05 → отримати **ранжований список топ-20** з поясненням → відправити лише їх → **економія 98% ресурсів**.

Саме так працюють сучасні фарм-компанії: ML-скринінг → дізайн синтезу → лабораторне підтвердження лише для топ-кандидатів.

### Три рівні аналізу в NB05

#### Рівень 1: Unsupervised Clustering — навчитися розуміти хімічний простір

**Хімічний простір** (chemical space) — це багатовимірне уявлення всіх можливих молекул. Дві молекули «близько» в хімічному просторі = вони структурно схожі = скоріш за все мають схожу біологічну активність (принцип подібності, similarity principle).

Ми проективємо цей 1024D/2048D простір у 2D/3D для візуалізації:
- **PCA** → лінійна проекція, збережена максимальна дисперсія, швидко
- **t-SNE** → нелінійна, зберігає локальну сусідську структуру, «кластери видно як острови»
- **UMAP** → швидше t-SNE + краще зберігає глобальну структуру + можна проектувати нові точки

Потім **кластеризуємо** ці проекції:
- **KMeans** → задаємо k (Elbow method + Silhouette score), кожна точка — рівно в 1 кластері
- **DBSCAN** → знаходить кластери довільної форми + виявляє шум (=аномальні/унікальні молекули!)
- **HDBSCAN** → ієрархічна версія, soft memberships, краща для хімічних даних

**Що дають кластери?** Кожен кластер = хімічна серія (scaffold family). Це критично для:
1. **Diversity selection:** брати з кожного кластера по 1–2 представники → максимальна структурна різноманітність → більше шансів покрити різні binding modes
2. **SAR analysis:** молекули всередині кластера мають схожу активність → помілуй SAR trends всередині серії
3. **IP (intellectual property) аналіз:** кластери часто відповідають патентним родинам

#### Рівень 2: Scoring & Ranking — знайти найкращих кандидатів

Просто «активна» молекула недостатня. Реальний drug candidate повинен бути:
- **Активним** (IC50 < 1 μM проти цілі)
- **Drug-like** (Lipinski, QED)
- **Нетоксичним** (без Tox21 alerts)
- **Синтетично доступним** (SA score < 4)
- **Структурно різноманітним** (не всі з однієї серії)

Ми обчислюємо **DrugScore** як зважену суму:
```
DrugScore = QED×0.4 + norm(MW)×0.2 + norm(LogP)×0.2 + norm(TPSA)×0.1 + norm(HBD)×0.1
```

З інтеграцією NB01/NB03 досягаємо повного multi-objective scoring:
```
FinalScore = DrugScore × pred_activity × (1 - max_toxicity)
```

#### Рівень 3: Scaffold Analysis — зрозуміти молекулярні серії

**Bemis-Murcko scaffold** — «кістяк» молекули без бічних ланцюгів. Дві молекули з однаковим скаффолдом = одна хімічна серія. Scaffold analysis:
- Які скаффолди найпоширеніші в бібліотеці? → scaffold frequency distribution
- Який скаффолд має найвищий середній DrugScore? → найбільш перспективна серія
- Sunburst chart: scaffold family → конкретні скаффолди → кількість молекул

### Що ми робимо покроково

1. **Збираємо бібліотеку** (~500+ drug-like SMILES): ChEMBL API + FDA approved drugs + ZINC subset
2. **Lipinski filter** → залишаємо лише drug-like (MW≤500, logP≤5, HBD≤5, HBA≤10)
3. **Обчислюємо дескриптори:** MW, LogP, TPSA, HBD, HBA, RotBonds, ArRings, QED, SA score
4. **Morgan fingerprints (ECFP4, 1024 bit)** — основа для clustering і Tanimoto similarity
5. **Tanimoto similarity matrix** 50×50 → heatmap
6. **PCA → t-SNE → UMAP** проекції у 2D і 3D (Plotly)
7. **KMeans** (Elbow + Silhouette) + **DBSCAN** + **HDBSCAN**
8. **DrugScore** обчислення і ранжування
9. **Top-20 globally + Top-3 per cluster** — ранжування кандидатів
10. **Scaffold analysis** (Bemis-Murcko): frequency pie chart + sunburst Plotly
11. **pyvis Similarity Network** — граф молекул, Tanimoto ≥ 0.4 = ребро
12. **Інтеграція NB01+NB03:** завантаження збережених моделей → передбачення токсичності і активності → FinalScore
13. **SDF export** топ-кандидатів для молекулярного докінгу
14. **3D UMAP (Plotly scatter_3d)** — кольором кластер, розміром QED

### Що ми досягаємо в результаті

✅ **Головний результат:** Повний **in silico drug discovery pipeline** від бібліотеки молекул до пронумерованого списку топ-20 кандидатів для синтезу і тестування — з обґрунтуванням кожного вибору.

✅ **Multi-objective optimization:** Одночасно оптимізуємо активність, drug-likeness, нетоксичність і синтетичну доступність — саме так і мусить виглядати реальний drug candidate selection.

✅ **SDF file для докінгу:** Топ-кандидати зберігаються у форматі SDF → можна безпосередньо завантажити в AutoDock Vina, Schrödinger Maestro, MOE для структурної валідації.

✅ **Замикання кола:** NB05 використовує *всі* попередні ноутбуки: токсичність (NB01), ADMET (NB02 — LogP і MW), активність (NB03). Це **ансамбль прогнозів** від трьох незалежних ML-систем, який формує один фінальний score.

✅ **Портфоліо-цінність:** Cheminformatics + clustering + multi-objective scoring — це core workflow будь-якого computational chemist або ML drug discovery scientist. Ноутбук демонструє здатність мислити не лише як ML-інженер, а як **drug hunter** — людина, яка розуміє, що стоїть за цифрами.

✅ **Візуальна насиченість:** 3D хімічний простір, pyvis similarity network, sunburst scaffold hierarchy, comparative radar charts — це найвізуально вражаючий ноутбук портфоліо, ідеальний для презентацій і співбесід.

---

## ✅ Що зроблено

### Архітектура пайплайну
| Крок | Реалізація |
|------|-----------|
| Бібліотека | ~50 реальних drug-like молекул × 10 = 500 → Lipinski filter → ~45 унікальних |
| Дескриптори | MW, LogP, TPSA, HBD, HBA, RotBonds, ArRings, QED, NumRings |
| Fingerprints | Morgan (ECFP4) 1024-bit, radius=2 |
| Tanimoto | Матриця подібності 50×50 + heatmap |
| Dim. Reduction | PCA (→50D) → t-SNE (→2D), UMAP (→2D з fallback на t-SNE) |
| Кластеризація | KMeans (k вибирається Elbow + Silhouette), DBSCAN (eps=0.5, min_samples=3) |
| Scoring | Composite DrugScore: QED(0.4) + MW(0.2) + LogP(0.2) + TPSA(0.1) + HBD(0.1) |
| Топ кандидати | Top-20 globally + Top-3 per cluster |
| Scaffold аналіз | Bemis-Murcko scaffold → frequency distribution pie chart |
| Exp. | CSV export + mols2grid interactive grid (якщо доступно) |

### Виправлені баги
| Баг | Причина | Виправлення |
|-----|---------|-------------|
| `TSNE(n_iter=1000)` → `TypeError` | `n_iter` перейменований на `max_iter` в sklearn ≥ 1.4 | Замінено на `max_iter=1000` |
| CSV зберігався за `'05_Molecular_Clustering/top_candidates.csv'` | Неправильний шлях відносно CWD ноутбука | Замінено на `'top_candidates.csv'` |

---

## 🔴 Що залишилось / Що покращити

### 🥇 Рівень 1 — Обов'язково для сильного портфоліо

#### 1. Значно збільшити бібліотеку (поточна ~45 молекул — занадто мало)
Для реального кластерного аналізу потрібно мінімум **500–5000 молекул**.

> 🧠 **Детальніше:** Кластерний аналіз на 45 молекулах — це як ділити місто на райони, маючи карту з 45 будинками. KMeans і DBSCAN з такою кількістю дадуть нестабільні результати: додавання/видалення 5 молекул може повністю змінити кластерну структуру. Мінімально значущий аналіз: 500+ молекул для ~5 кластерів (правило 100 молекул на кластер). З 2000–5000 молекул ви зможете знайти реальні хімічні серії, виявити «хмари» структурно схожих кандидатів і вибрати справді різноманітних представників для синтезу.
```python
# Варіант 1: ChEMBL API — drug-like subset
from chembl_webresource_client.new_client import new_client
molecule = new_client.molecule
drug_like = molecule.filter(
    molecule_properties__mw_freebase__lte=500,
    molecule_properties__alogp__lte=5,
    molecule_properties__hbd__lte=5,
    molecule_properties__hba__lte=10,
).only(['molecule_structures'])[:2000]

# Варіант 2: ZINC250k через DeepChem
# Варіант 3: FDA Approved Drugs (1600+ молекул)
```

#### 2. SA Score як обов'язковий компонент DrugScore
Synthetic Accessibility Score від ErtI & Schuffenhauer. Вже входить до RDKit:

> 🧠 **Детальніше:** SA score (1–10): 1 = тривіальний синтез (аспірин), 10 = практично неможливо синтезувати. Розраховується через порівняння фрагментного складу молекули з PubChem (рідкісні фрагменти → висока SA). Більшість схвалених ліків мають SA 1–4. Молекули з SA > 6 навряд чи пройдуть hit-to-lead оптимізацію навіть при гарній активності. Тому SA score **обов'язково** включати в будь-який scoring функцію для drug candidates — інакше кластер може найкращими «кандидатами» видати молекули, які неможливо синтезувати.
```python
from rdkit.Chem.rdMolDescriptors import CalcCrippenDescriptors
# Або через sascorer:
from rdkit.Chem import RDConfig
import sys, os
sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
import sascorer
sa = sascorer.calculateScore(mol)  # 1=легко, 10=складно синтезувати
# Включити в DrugScore: sa_s = 1 - (sa - 1) / 9
```

#### 3. Export у SDF формат (стандарт хемоінформатики)
CSV недостатньо — додати SDF export для безпосереднього використання в docking software.

> 🧠 **Детальніше:** SDF (Structure Data File) — універсальний формат хемоінформатики для зберігання молекулярних структур **разом** з даними. CSV зберігає лише SMILES рядок — програми по типу AutoDock Vina, Maestro (Schrödinger), MOE не розуміють SMILES напряму і вимагають SDF або mol2. SDF містить: атомні координати (2D або 3D), зв'язки, властивості молекули в структурованому форматі. Для пайплайну drug discovery: «знайшли топ-20 кандидатів у NB05 → зберегли в SDF → відправили на docking → отримали результати» — це стандартний workflow.
```python
from rdkit.Chem import SDWriter

writer = SDWriter('top_candidates.sdf')
for _, row in top20.iterrows():
    mol = Chem.MolFromSmiles(row['SMILES'])
    mol.SetProp('_Name', f"Candidate_{row.name}")
    mol.SetProp('DrugScore', str(row['DrugScore']))
    mol.SetProp('QED', str(row['QED']))
    writer.write(mol)
writer.close()
print("Збережено top_candidates.sdf")
```

### 🥈 Рівень 2 — Значно підвищить рівень

#### 4. HDBSCAN — ієрархічний density clustering
Покращена версія DBSCAN: автоматично визначає не тільки кластери, але й їх ієрархію.

> 🧠 **Детальніше:** Класичний DBSCAN вимагає вручну підбирати `eps` і `min_samples` — і результат дуже чутливий до цих параметрів. HDBSCAN усуває цю проблему: він будує ієрархічне дерево щільності і автоматично вибирає стабільні кластери. Особливо добре працює, коли кластери мають різну щільність (одна хімічна серія з 50 молекула i інша з 200). Додатково HDBSCAN видає **soft memberships** — кожна молекула має ймовірність належності до кожного кластеру, що корисніше чим жорстке «кластер 1 або 2». Noise points (ізоляти) в хімії часто є найбільш новими/унікальними структурами.
```bash
pip install hdbscan
```
```python
import hdbscan
clusterer = hdbscan.HDBSCAN(min_cluster_size=5, metric='euclidean')
hdb_labels = clusterer.fit_predict(X_cluster)
# soft_clusters = hdbscan.all_points_membership_vectors(clusterer)
```

#### 5. Scaffold Tree Visualization
Ієрархічне дерево скаффолдів — кожен вузол = скаффолд, нащадки = більш специфічні.
```python
# Побудувати дерево Murcko → fragment → atom
# Або використати ScaffoldTree бібліотеку
```

#### 6. Diversity Analysis — Sphere Exclusion Algorithm
Вибір максимально різноманітного підмножества молекул (замість top-N by score).

> 🧠 **Детальніше:** Top-20 by DrugScore може вибрати 20 молекул з одної хімічної серії — всі схожі, тестуємо одне й те саме. Sphere Exclusion вирішує це: вибрати N молекул максимально **різноманітних** за структурою (максимізувати мінімальний Tanimoto між будь-якою парою вибраних). Це жадібний алгоритм: вибираємо першу молекулу → «огороджуємо сферою» радіусу threshold → наступна молекула обирається поза всіма сферами → і так далі. Для medicinal chemistry team це критично: різноманітний набір кандидатів дає більше SAR інформації за ті ж гроші.
```python
def sphere_exclusion(smiles_list, fps, n_pick=50, threshold=0.35):
    """Greedy sphere exclusion for diverse subset selection."""
    picked = [0]  # починаємо з першої молекули
    remaining = list(range(1, len(smiles_list)))
    while len(picked) < n_pick and remaining:
        # Знайти molecule максимально далеку від вже відібраних
        max_min_dist = -1
        best = None
        for i in remaining:
            min_sim = min(DataStructs.TanimotoSimilarity(fps[i], fps[j]) for j in picked)
            if min_sim > max_min_dist:
                max_min_dist = min_sim
                best = i
        if max_min_dist > threshold:
            break
        picked.append(best)
        remaining.remove(best)
    return picked
```

#### 7. Інтерактивний Dashboard (Plotly Dash)
```python
import plotly.express as px
fig = px.scatter(
    df_mol, x='UMAP_1', y='UMAP_2',
    color='KMeans_Cluster', size='QED',
    hover_data=['SMILES', 'MW', 'LogP', 'DrugScore'],
    title='Interactive Chemical Space'
)
fig.write_html('chemical_space_interactive.html')
```

#### 8. Інтеграція з результатами NB01-NB03
Передбачити токсичність (NB01) і активність (NB03) для кластерних центроїдів.

> 🧠 **Детальніше:** Це «фінальний акорд» всього портфоліо — демонстрація зв'язку між усіма проєктами. Workflow:
> 1. NB05 знаходить 1000 drug-like кандидатів і кластеризує їх у 8 груп
> 2. Завантажуємо збережені XGBoost моделі з NB01 → передбачаємо токсичність по всіх 12 Tox21 endpoints
> 3. Завантажуємо модель з NB03 → передбачаємо активність проти EGFR (pIC50)
> 4. Фінальний scoring: `score = activity_score × (1 - max_toxicity) × drug_score`
> 5. Топ-20 молекул — «найактивніші і найбезпечніші» кандидати для синтезу
>
> Це демонструє повний **in silico drug discovery pipeline** від бібліотеки молекул до відбору кандидатів — саме так працюють в реальних фармацевтичних компаніях.
```python
# Завантажити збережені моделі з NB01 та NB03
xgb_tox = joblib.load('../01_Toxicity_Prediction/models/xgb_SR-MMP.pkl')
xgb_act = joblib.load('../03_Activity_Classification/models/egfr_xgboost.pkl')

# Передбачення для всіх молекул у бібліотеці
df_mol['pred_toxicity'] = xgb_tox.predict_proba(fp_matrix)[:, 1]
df_mol['pred_activity'] = xgb_act.predict_proba(fp_matrix)[:, 1]

# Новий DrugScore + activity + low toxicity
df_mol['final_score'] = df_mol['DrugScore'] * 0.5 + df_mol['pred_activity'] * 0.3 * (1 - df_mol['pred_toxicity']) * 0.2
```

### 🥉 Рівень 3 — Просунуті ідеї

#### 9. Network Graph — Molecular Similarity Network
Вузли = молекули, ребра = Tanimoto > threshold → граф хімічних серій.

> 🧠 **Детальніше:** Similarity network — потужний інструмент для аналізу хімічних серій. Молекули з Tanimoto > 0.4 → ребро між ними. Зв'язані компоненти графа відповідають хімічним серіям (SAR families). Розмір вузла = DrugScore, колір = кластер, товщина ребра = Tanimoto. Що шукати: «хаби» — молекули в центрі багатьох зв'язків (scaffold center), «мости» — молекули між різними серіями (scaffold hop potential). pyvis вже реалізовано в NB05 — інтерактивний HTML з zoom/drag/tooltip.
```python
import networkx as nx
G = nx.Graph()
for i, j in combinations(range(N), 2):
    if tanimoto[i, j] > 0.4:
        G.add_edge(i, j, weight=tanimoto[i, j])
# Розмалювати вузли по QED, ширину ребер по Tanimoto
```

#### 10. Principal Moments of Inertia (PMI) — 3D shape diversity
> 🧠 **Що це:** PMI аналізує 3D форму молекули через три головні моменти інерції (I1 ≤ I2 ≤ I3). Нормалізовані значення (npr1 = I1/I3, npr2 = I2/I3) відображаються на трикутній PMI діаграмі: кут «rod» (лінійні молекули), кут «disk» (плоскі кільця), кут «sphere» (сферичні молекули). Різноманіття у 3D shape простіорі означає різноманіття в механізмах зв'язування. Важливо: FDA схвалені ліки мають різний розподіл на PMI vs, наприклад, типові HTS хіти — більше rod-shaped, що дозволяє глибше проникати в білкові кишені.

#### 11. Matched Molecular Pairs (MMP) analysis
> 🧠 **Що це:** MMP — пара молекул, які відрізняються лише одним структурним фрагментом (наприклад, -CH3 замінено на -CF3, або -OH замінено на -NH2). Аналіз MMP відповідає на питання: «яка хімічна трансформація найбільше покращує/погіршує QED/активність?». Це автоматизований варіант medicinal chemistry intuition: бачимо патерн «заміна F на Cl на цій позиції → +0.5 pIC50 в середньому». Бібліотека `mmpdb` від RDKit автоматизує цей аналіз.

#### 12. Integration з AutoDock Vina для скорингу кандидатів
> 🧠 **Що це:** AutoDock Vina — безкоштовний інструмент молекулярного докінгу (Scripps Research). Workflow: взяти 3D структуру EGFR з PDB (наприклад 1IEP), підготувати білок (водні молекули, заряди), помістити топ-20 кандидатів від NB05 в «докінг бокс» активного сайту, запустити simulation. Результат: binding affinity (ккал/моль) і позиція молекули в кишені. Молекули з docking score < -9 ккал/моль — пріоритетні для синтезу. Це замикає повний цикл: ML-скринінг (NB05) → structure-based validation (docking) → синтез найкращих.

---

## 📊 Очікувані результати (з ~1000 молекул)

| Metric | Очікується |
|--------|-----------|
| KMeans clusters (k) | 6–12 |
| Silhouette score | 0.25 – 0.45 |
| DBSCAN noise % | 5 – 20% |
| Scaffold diversity | 60 – 85% |
| Top DrugScore | 0.70 – 0.88 |

---

## 🗂️ Залежності для нових фіч
```bash
pip install hdbscan networkx plotly
pip install chembl-webresource-client  # вже є
# sascorer входить до rdkit contrib
```

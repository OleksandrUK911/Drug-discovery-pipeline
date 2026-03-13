# NB03 — Activity Classification (EGFR)
**Файл:** `03_Activity_Classification/Activity_Classification.ipynb`  
**Задача:** Бінарна класифікація — активність молекул проти EGFR (IC50-based)  
**Датасет:** ChEMBL API (EGFR / CHEMBL203) — Active < 1μM, Inactive > 10μM

---

## 🎯 Для чого цей ноутбук і чого ми досягаємо

### Контекст: що таке EGFR і чому він важливий

**EGFR** (Epidermal Growth Factor Receptor, рецептор епідермального фактора росту) — це трансмембранна кіназа, яка регулює клітинний ріст і поділ. При мутаціях або надмірній активації EGFR клітини починають ділитися безконтрольно → **рак**. Мутований EGFR виявлено в ~30% випадків раку легень (NSCLC), рак підшлункової залози, голови і шиї, товстої кишки.

**EGFR-інгібітори** — це клас таргетних протиракових препаратів. FDA-схвалені інгібітори:
- **Erlotinib (Tarceva)** — 1-е покоління, 2004
- **Gefitinib (Iressa)** — 1-е покоління
- **Osimertinib (Tagrisso)** — 3-є покоління, подолає T790M резистентність

Пошук нових EGFR-інгібіторів — активна область досліджень, особливо для подолання резистентних мутацій.

### Яку проблему ми вирішуємо

**Virtual Screening (VS)** — in silico просіювання хімічних бібліотек для знаходження активних молекул. Без ML: лабораторний High-Throughput Screening (HTS) коштує $1–5 мільйонів та займає місяці для перебору ~100,000 молекул. З ML: та сама бібліотека просіюється за хвилини, а до лабораторного тесту доходять лише ~1,000 найперспективніших кандидатів.

**NB03 навчає модель відрізняти активні EGFR-інгібітори від неактивних** на основі хімічної структури:
- IC50 < 1,000 nM (1 μM) → **Active** (1) — молекула ефективно блокує EGFR
- IC50 > 10,000 nM (10 μM) → **Inactive** (0) — молекула практично не впливає
- IC50 між 1–10 μM → **амбігуозна зона** → видаляємо (щоб не вчити модель на «сірих» даних)

### Звідки беруться дані

**ChEMBL** — найбільша відкрита база даних біоактивних молекул (>2 млн сполук, >14,000 цілей). Ми запитуємо ChEMBL API напряму з Python через `chembl_webresource_client`:
```
EGFR (CHEMBL203) → всі IC50 вимірювання → фільтруємо за порогами → SMILES + labels
```
Результат: ~2,000–5,000 молекул з бінарними мітками активності. Це **реальні фармацевтичні дані**, не синтетика.

### Яка технічна складність задачі

1. **Дисбаланс класів:** активних молекул зазвичай менше (~20–35%). Нейтральна модель каже «неактивно» і отримує 75% accuracy — але знаходить ±0 хітів. Тому: `scale_pos_weight`, `class_weight='balanced'`, оцінка за ROC-AUC / PR-AUC / F1 / MCC.

2. **Threshold optimization:** стандартний поріг 0.5 субоптимальний для дисбалансних даних. Ми оптимізуємо threshold для максимального F1 — і пояснюємо, коли краще максимізувати Recall (рання VS), а коли Precision (фінальний відбір).

3. **Scaffold bias:** якщо erlotinib і його хімічна «родичка» потрапляють в train і test — модель серйозно переоцінює свій AUC. Scaffold split обов'язковий.

4. **SAR interpretation:** не лише передбачити, але й зрозуміти ЧОМУ — SHAP waterfall plots для конкретних молекул.

### Що ми робимо покроково

1. **ChEMBL API запит** → IC50 дані для EGFR → бінарна мітка по порогах 1/10 μM
2. **Morgan FP + RDKit дескриптори** → вектор 2056D
3. **Scaffold-aware split** → train/test без data leakage
4. **Тренуємо 4 моделі:** Logistic Regression, Random Forest, XGBoost, DNN
5. **ROC-AUC, PR-AUC, F1, MCC** — повний набір метрик для дисбалансу
6. **ROC curves (Plotly)** — всі 4 моделі на одному canvas, інтерактивно
7. **Confusion matrix (Plotly)** — інтерактивна, з threshold slider
8. **SHAP top-20 features** + SHAP waterfall для конкретної молекули
9. **Threshold optimization** — F1/Recall/Precision curve, оптимальна точка
10. **ChemBERTa fine-tuning** — BERT-like transformer на SMILES (розширення)
11. **MLflow** — всі метрики + параметри кожного run

### Що ми досягаємо в результаті

✅ **Практичний результат:** Модель, яка з ~85–89% ROC-AUC відсіює неактивні молекули і знаходить EGFR-інгібітори у великих хімічних бібліотеках. Це золотий стандарт virtual screening pipeline.

✅ **Explainability:** SHAP показує, які саме фрагменти молекули відповідають за активність — quinazoline scaffold, aniline NH2 група, галогени у певних позиціях. Це дій'но SAR-аналіз.

✅ **Multi-model comparison:** вперше бачимо, коли DNN перевищує gradient boosting і за яких умов. Важливий урок для ML practitioner.

✅ **Зв'язок з NB05:** Збережена best model завантажується в NB05 → передбачення активності для всіх 1,000+ кандидатів із бібліотеки → додається до фінального DrugScore поряд з токсичністю (NB01) і фізхімічними властивостями (NB02). Так реалізується повний *in silico* drug discovery pipeline.

✅ **Портфоліо-цінність:** QSAR / virtual screening — **найзатребуваніша** ML-компетенція у фармацевтичній індустрії (Pfizer, Novartis, GSK, Roche усі мають ML-відділи саме для VS). Ноутбук демонструє повний цикл: завантаження реальних даних з ChEMBL API → препроцесінг → model comparison → explainability → продакшн-ready збереження.

---

## ✅ Що зроблено

### Архітектура пайплайну
| Крок | Реалізація |
|------|-----------|
| Завантаження даних | `chembl_webresource_client` API → IC50 для EGFR; синтетичний fallback |
| Бінарна метка | IC50 < 1000 nM = Active (1); IC50 > 10000 nM = Inactive (0); амбігуозні прибрані |
| Молекулярні ознаки | Morgan fingerprints (2048 bit) + 8 RDKit дескрипторів → 2056D |
| Моделі | Random Forest (`class_weight='balanced'`), XGBoost (`scale_pos_weight`) |
| Deep Learning | DNN: `512→256→64→1` з BatchNorm, Dropout, Sigmoid |
| Дисбаланс класів | `class_weight='balanced'` (RF), `scale_pos_weight` (XGB), `BCELoss` (DNN) |
| Метрики | ROC-AUC, PR-AUC, F1, MCC — повний набір для дисбалансу |
| Візуалізація | ROC curves, PR curves, Metrics comparison, Confusion matrix |
| Explainability | SHAP top-20 features, Decision threshold optimization (F1) |

### Виправлені баги
| Баг | Причина | Виправлення |
|-----|---------|-------------|
| `use_label_encoder=False` у `XGBClassifier` | Видалено в XGBoost ≥ 2.0 | Аргумент прибрано |

---

## 🔴 Що залишилось / Що покращити

### 🥇 Рівень 1 — Обов'язково для сильного портфоліо

#### 1. Замінити бінарну класифікацію на регресію pIC50
Замість порогів active/inactive → передбачати **pIC50 = -log10(IC50_M)** напряму.  
Переваги: менше втрати даних, точніший SAR, реальніша постановка задачі.

> 🧠 **Детальніше:** pIC50 — стандартна мера активності в medicinal chemistry. IC50 = 1 nM → pIC50 = 9 (дуже активний). IC50 = 10 μM → pIC50 = 5 (слабко активний). Бінарна класифікація з порогами 1 μM / 10 μM **викидає** всі «сірі зони» (1–10 μM) і стирає різницю між IC50=0.1 nM і IC50=900 nM (обидва = «active»). Регресія на pIC50 зберігає весь gradient активності → набагато кращий Structure-Activity Relationship (SAR) аналіз. Більшість публікацій в QSAR використовують саме pIC50.
```python
df['pIC50'] = -np.log10(df['IC50_nM'] * 1e-9)  # конвертація nM → M
df = df[(df['pIC50'] > 3) & (df['pIC50'] < 12)]  # видалити аномалії
# Тоді задача — регресія замість класифікації
```

#### 2. Scaffold-aware train/test split
Щоб не допустити data leakage — молекули з однаковим скаффолдом мають йти в один split.
```python
from rdkit.Chem.Scaffolds import MurckoScaffold
# Реалізувати групований split по скаффолду
```

#### 3. Збереження найкращої моделі
```python
import joblib
best_clf_name = max(results, key=lambda x: results[x]['ROC-AUC'])
joblib.dump(models[best_clf_name], f'models/egfr_{best_clf_name.lower().replace(" ", "_")}.pkl')
print(f"Збережено: {best_clf_name}")
```

#### 4. Додати більше цілей (не тільки EGFR)
Перетворити ноутбук на **multi-target QSAR** — EGFR + HER2 + VEGFR2.
```python
TARGETS = {
    'EGFR'  : 'CHEMBL203',
    'HER2'  : 'CHEMBL1824',
    'VEGFR2': 'CHEMBL279',
}
```

### 🥈 Рівень 2 — Значно підвищить рівень

#### 5. SMILES Augmentation (data augmentation для молекул)
Одна молекула = кілька SMILES (різний порядок обходу атомів) → більший датасет.

> 🧠 **Детальніше:** SMILES — це текстове представлення молекули, де атоми перераховуються в певному порядку обходу графа. RDKit може генерувати **різні валідні SMILES** для однієї і тієї ж молекули (randomized SMILES). Аспірин: `CC(=O)Oc1ccccc1C(=O)O` і `OC(=O)c1ccccc1OC(C)=O` — одна і та ж молекула! Якщо передавати ці варіанти SMILES в char-level або subword токенізатор (ChemBERTa), це виконує роль звичайного image augmentation (flip, rotate). Для маленьких датасетів (~500 молекул) може давати +5–15% покращення.
```python
from rdkit.Chem import MolToSmiles, MolFromSmiles
def augment_smiles(smiles, n_aug=5):
    mol = MolFromSmiles(smiles)
    return [MolToSmiles(mol, doRandom=True) for _ in range(n_aug)]
```

#### 6. Порівняння fingerprint types
Додати в benchmark FCFP4, AtomPair, TopologicalTorsion, MACCS keys.

> 🧠 **Детальніше:** Різні fingerprints кодують різні аспекти структури:
> - **ECFP4 (Morgan r=2)** — кругові підструктури радіусом 2 зв'язки (найчастіше в QSAR)
> - **FCFP4** — те ж саме, але атоми замінені на фармакофорні класи (Donor/Acceptor/Aromatic/...)
> - **MACCS** — 166 бінарних ключів: «чи є в молекулі кільце?», «чи є NH2?» тощо — легко інтерпретовані
> - **AtomPair** — кодує пари атомів і відстань між ними по графу — добре вловлює 3D форму
> - **TopologicalTorsion** — чотири послідовні атоми вздовж кута повороту
>
> Для EGFR (кінази) зазвичай FCFP4 або AtomPair дають +2–4% AUC відносно ECFP4.
```python
fp_types = {
    'Morgan (ECFP4)' : lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048),
    'FCFP4'          : lambda mol: AllChem.GetMorganFingerprintAsBitVect(mol, 2, 2048, useFeatures=True),
    'MACCS'          : lambda mol: MACCSkeys.GenMACCSKeys(mol),
    'AtomPair'       : lambda mol: rdMolDescriptors.GetHashedAtomPairFingerprintAsBitVect(mol, 2048),
}
```

#### 7. Threshold optimization для різних завдань
Поточний threshold optimizer максимізує F1. Додати варіанти для різних бізнес-цілей:

> 🧠 **Детальніше:** У drug discovery **помилки не рівнозначні**:
> - **False Negative** (пропустити активну молекулу) = витрачаємо потенційний лід-кандидат. На ранніх стадіях VS — це катастрофа.
> - **False Positive** (взяти неактивну молекулу) = витрачаємо час і гроші на синтез/тестування нікчемної молекули.
>
> Тому threshold вибирається залежно від стадії проєкту. Рання VS (Virtual Screening) з мільйонів молекул → низький threshold (maximize recall, Fbeta з β=2). Фінальний відбір 20 кандидатів для синтезу → високий threshold (maximize precision). ROC curve дозволяє «торгуватися» між цими помилками, вибираючи будь-яку точку кривої.
- `maximize_recall` → для ранньої VS (не пропустити жодного хіту)
- `maximize_precision` → для фінального відбору (мінімум хибних спрацьовувань)
- `F-beta` (β=2 → більш вага до recall)

#### 8. SHAP Waterfall plot для конкретної молекули
```python
# Пояснити передбачення для top-1 active молекули
idx = np.where(y_te == 1)[0][0]
shap.waterfall_plot(shap.Explanation(
    values=shap_values[idx],
    base_values=explainer.expected_value,
    feature_names=FEAT_NAMES
))
```

### 🥉 Рівень 3 — Просунуті ідеї

#### 9. Transfer learning з ChemBERTa
```bash
pip install transformers datasets
```
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
# Fine-tune на EGFR dataset
```
> 🧠 **Що це:** ChemBERTa — BERT-трансформер, pretrained на 77 мільйонах SMILES з ZINC. Fine-tune: заморозити більшість шарів, замінити classification head, навчати ~10 епох на EGFR dataset (~500–2000 молекул). Навіть з малим датасетом fine-tuned ChemBERTa зазвичай перевершує XGBoost на Morgan FP через якість ембедингів. Модель «вже знає хімію» і лише адаптується до нової задачі.

#### 10. Molecular Docking Score як додаткова ознака
Інтегрувати docking score (AutoDock Vina) як feature.
> 🧠 **Що це:** Molecular docking — симуляція зв'язування молекули з білком (EGFR) в 3D. AutoDock Vina рахує «score» (ккал/моль) — чим менший (більш від'ємний), тим краще зв'язування. Додавання цього score як feature в ML модель об'єднує структурний (docking) і статистичний (QSAR) підходи. Ключова проблема: docking потребує 3D структури білка і займає ~1–5 хв на молекулу, тому використовується лише для відфільтрованих кандидатів.

#### 11. 3D Pharmacophore features
Додати 3D дескриптори (USR, USRCAT, PMI) до feature vector.
> 🧠 **Що це:** 2D Morgan fingerprints ігнорують 3D форму молекули. **USR (Ultrafast Shape Recognition)** кодує 3D форму через 12 чисел (моменти відстаней від центру, найближчого/найдальшого атому, центру мас). **PMI (Principal Moments of Inertia)** визначає «форму» молекули на трикутній діаграмі: лінійна (rod) — disk — sphere. Ліки, які зв'язуються з EGFR кишенею, мають специфічну форму — 3D дескриптори допоможуть це виявити.

---

## 📊 Очікувані метрики (реальний ChEMBL EGFR)

| Модель | ROC-AUC | PR-AUC |
|--------|---------|--------|
| Random Forest | 0.82 – 0.87 | 0.60 – 0.72 |
| XGBoost | 0.84 – 0.89 | 0.62 – 0.75 |
| DNN | 0.83 – 0.88 | 0.61 – 0.73 |
| ChemBERTa fine-tuned | 0.88 – 0.93 | 0.70 – 0.82 |

---

## 🗂️ Залежності для нових фіч
```bash
pip install joblib transformers datasets
pip install chembl-webresource-client  # вже є
```

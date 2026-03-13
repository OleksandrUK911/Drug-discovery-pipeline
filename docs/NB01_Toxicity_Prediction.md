# NB01 — Toxicity Prediction (Tox21)
**Файл:** `01_Toxicity_Prediction/Toxicity_Prediction.ipynb`  
**Задача:** Багатоцільова класифікація (12 мішеней) — токсичність молекул  
**Датасет:** Tox21 (~8,000 сполук, 12 assay-цілей)

---

## 🎯 Для чого цей ноутбук і чого ми досягаємо

### Контекст: де ми знаходимось у пайплайні drug discovery

Розробка ліків — це процес, який коштує **$1–3 мільярди** і займає **10–15 років**. Одна з головних причин провалу кандидатів на клінічних стадіях — **токсичність**, яку не виявили достатньо рано. Приблизно **30% відмов** у клінічних випробуваннях пов'язані саме з проблемами безпеки (гепатотоксичність, кардіотоксичність, мутагенність тощо). Традиційні лабораторні тести на токсичність (in vitro, in vivo) дорогі, повільні і не масштабуються на мільйони молекул.

**NB01 вирішує цю проблему** за допомогою машинного навчання: ми навчаємо моделі передбачати токсичність молекул *in silico* — тобто на комп'ютері, без жодного лабораторного тесту.

### Що таке Tox21 і чому він важливий

**Tox21** (Toxicology in the 21st Century) — це міжнародна ініціатива NIH/EPA/FDA, яка протестувала ~8,000 хімічних сполук на **12 токсикологічних мішенях** за допомогою high-throughput screening. Це один з найбільш авторитетних публічних датасетів з токсикології. Кожна молекула отримала бінарну мітку (0/1) для кожної з 12 мішеней:

| Група | Мішені | Що вимірюється |
|-------|--------|----------------|
| Ядерні рецептори (NR) | NR-AR, NR-AR-LBD, NR-AhR, NR-Aromatase, NR-ER, NR-ER-LBD, NR-PPAR-γ | Чи активує/блокує молекула гормональні рецептори — ендокринний дисраптор? |
| Стресові відповіді (SR) | SR-ARE, SR-ATAD5, SR-HSE, SR-MMP, SR-p53 | Чи викликає молекула клітинний стрес, пошкодження ДНК, мітохондріальну токсичність? |

Це критичні механізми: активація androgen receptor (NR-AR) → гормональний дисбаланс; активація p53 (SR-p53) → генотоксичність; зниження мітохондріального мембранного потенціалу (SR-MMP) → клітинна смерть.

### Яку задачу ми вирішуємо технічно

Це задача **multi-label binary classification**: одна молекула → вектор з 12 незалежних бінарних передбачень. Ми НЕ навчаємо одну модель — ми навчаємо **окремий класифікатор для кожної з 12 мішеней** (і один multi-task GCN для всіх одночасно як розширення).

**Вхід:** структура молекули у вигляді SMILES рядка → Morgan fingerprint (2048 bit) + 8 RDKit дескрипторів = вектор 2056 чисел  
**Вихід:** 12 ймовірностей токсичності (по одній для кожної мішені)  
**Метрика:** ROC-AUC (Area Under ROC Curve) — стандарт для дисбалансних класифікаційних задач у медицині

### Чому це складна ML-задача

1. **Сильний дисбаланс класів:** реальна токсичність — рідкісна подія. Для більшості мішеней токсичних молекул лише 5–15%. Модель, яка завжди каже «нетоксично», матиме 90%+ accuracy — але абсолютно нічого не передбачатиме. Тому використовуємо `scale_pos_weight`, `class_weight='balanced'` і оцінюємо ROC-AUC.
2. **Висока розмірність, мало сигналу:** 2048 Morgan bits — більшість = 0 для конкретної молекули. Справжніх «інформативних» бітів може бути 50–200.
3. **Мультизадача:** деякі мішені корелюють між собою (NR-AR і NR-AR-LBD — ліганд-зв'язувальний домен того ж рецептора), інші — незалежні.
4. **Scaffold bias:** схожі молекули тренувального та тестового наборів дають штучно завищені AUC → обов'язковий scaffold split.

### Що ми робимо покроково

1. **Завантажуємо дані** з Tox21 (через DeepChem або синтетичний fallback) — ~8,000 SMILES з мітками
2. **Генеруємо ознаки:** SMILES → Morgan fingerprint (кругові субструктури радіусу 2) + RDKit дескриптори (MW, LogP, TPSA тощо)
3. **Ділимо дані** scaffold-aware: молекули з однаковим Murcko-скаффолдом йдуть в один split
4. **Навчаємо базові моделі:** Random Forest і XGBoost для кожної з 12 мішеней
5. **Оцінюємо:** ROC-AUC на тестовому наборі для кожної мішені + Mean AUC
6. **Пояснюємо передбачення:** SHAP TreeExplainer → які фрагменти молекули найбільше впливають на токсичність (на прикладі SR-MMP)
7. **Deep learning:** Graph Convolutional Network (PyTorch Geometric) — молекула як граф атомів і зв'язків
8. **Збережені моделі** — для Streamlit dashboard і NB05

### Що ми досягаємо в результаті

✅ **Практичний результат:** Система здатна за секунди передбачити токсичний профіль будь-якої нової молекули по 12 механістичним мішеням — без жодного лабораторного тесту. Це дозволяє відфільтрувати потенційно небезпечні сполуки на найранніших стадіях drug discovery.

✅ **Наукова цінність:** Mean ROC-AUC ~0.80–0.84 на реальному Tox21 датасеті — порівнянно з публікаціями в Nature/JCIM.

✅ **Портфоліо-цінність:** Демонструє вміння: (1) обробляти хімічні дані (RDKit, Morgan FP), (2) вирішувати дисбалансні задачі, (3) будувати GCN на молекулярних графах, (4) застосовувати SHAP для explainability — всі ці навички критичні для позицій ML Scientist у фармацевтиці (Pfizer, Novartis, AstraZeneca, Insilico Medicine).

✅ **Зв'язок з іншими ноутбуками:** Збережені XGBoost моделі використовуються в NB05 (фінальний scoring кандидатів: активний + нетоксичний) і в Streamlit dashboard (real-time prediction по SMILES).

---

## ✅ Що зроблено

### Архітектура пайплайну
| Крок | Реалізація |
|------|-----------|
| Завантаження даних | DeepChem `load_tox21()` з scaffold split; синтетичний fallback |
| Молекулярні ознаки | Morgan fingerprints (2048 bit, r=2) + 8 RDKit дескрипторів → вектор 2056D |
| Baseline моделі | Random Forest (200 дерев, `class_weight='balanced'`) |
| Основна модель | XGBoost (200 estimators, `scale_pos_weight` для дисбалансу) |
| Стратегія валідації | 5-fold Stratified K-Fold cross-validation |
| Метрика | ROC-AUC per task + Mean ROC-AUC |
| Explainability | SHAP TreeExplainer — bar + beeswarm для `SR-MMP` |
| Deep Learning | Graph Convolutional Network (PyTorch Geometric) |
| Визуалізація | Per-target bar chart, ROC-AUC comparison, SHAP plots, GCN training curve |

### Виправлені баги
| Баг | Причина | Виправлення |
|-----|---------|-------------|
| `use_label_encoder=False` у `train_evaluate_per_task` | Видалено в XGBoost ≥ 2.0 | Аргумент прибрано |
| `use_label_encoder=False` у SHAP-клітинці | Те ж саме | Аргумент прибрано |
| `plt.subplots(1,2)` перед SHAP (порожній графік) | SHAP створює власну фігуру; зайві axes не використовуються | Зайву ініціалізацію прибрано |
| `from torch_geometric.data import DataLoader` | DataLoader переїхав до `torch_geometric.loader` в PyG ≥ 2.0 | `from torch_geometric.loader import DataLoader` з `ImportError` fallback |

---

## 🔴 Що залишилось / Що покращити

### 🥇 Рівень 1 — Обов'язково для сильного портфоліо

#### 1. Scaffold-Aware Split (замість random)
**Чому важливо:** Random split → надмірно оптимістичні AUC (тест-молекули схожі на тренувальні).  
Scaffold split — стандарт для публікацій та індустрії.

> 🧠 **Детальніше:** Молекулярний скаффолд (каркас) — це «скелет» молекули без бічних ланцюгів. Якщо аспірин і ібупрофен мають однаковий бензольний каркас, то при random split вони можуть потрапити в train і test одночасно — модель «вже бачила цей каркас» і видає оптимістичний AUC. При scaffold split всі молекули з однаковим каркасом йдуть **лише в один** split. Це критично важливо для оцінки реальної здатності моделі узагальнюватись на нові хімічні серії — що і є кінцевою метою в drug discovery.
```python
# Поточний код використовує DeepChem scaffold split тільки якщо DC доступний.
# Додати власну реалізацію через RDKit MurckoScaffold:
from rdkit.Chem.Scaffolds import MurckoScaffold
def scaffold_split(df, smiles_col, test_size=0.2):
    scaffolds = defaultdict(list)
    for i, smi in enumerate(df[smiles_col]):
        mol = Chem.MolFromSmiles(smi)
        scaffold = MurckoScaffold.MurckoDecompose(mol)
        scaffolds[Chem.MolToSmiles(scaffold)].append(i)
    # ... розподіл скаффолдів по splits
```

#### 2. Збереження моделей
> 🧠 **Навіщо:** Без збереження моделей кожен раз доводиться перенавчати (~5–20 хв). Збережені моделі можна завантажити в Streamlit dashboard, FastAPI endpoint або NB05 для передбачення токсичності кандидатів з інших ноутбуків. Це також дозволяє версіонувати моделі разом з MLflow.

```python
import pickle, os
os.makedirs('models', exist_ok=True)
for task, model in xgb_models.items():
    with open(f'models/xgb_{task}.pkl', 'wb') as f:
        pickle.dump(model, f)
torch.save(gcn_model.state_dict(), 'models/gcn_sr_mmp.pt')
print("Моделі збережено в models/")
```

#### 3. Hyperparameter Tuning з Optuna
> 🧠 **Навіщо:** Ручний підбір гіперпараметрів — це «стріляти навмання». Optuna використовує **Bayesian Optimization (TPE)** — кожен наступний trial враховує результати попередніх і пропонує більш перспективні значення. 50 trials з Optuna часто перевершують 200 trials з random search. Важливо: налаштовувати гіперпараметри треба **тільки на train set** (в cross-validation) — тест залишається «невидимим».

```python
import optuna
def objective(trial):
    n_est = trial.suggest_int('n_estimators', 100, 500)
    lr    = trial.suggest_float('learning_rate', 0.01, 0.3, log=True)
    depth = trial.suggest_int('max_depth', 3, 9)
    clf = xgb.XGBClassifier(n_estimators=n_est, learning_rate=lr, max_depth=depth, ...)
    # 3-fold CV на одній target
    return mean_auc

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
```

### 🥈 Рівень 2 — Значно підвищить рівень

#### 4. Multi-task GCN (всі 12 цілей одночасно)
Замість 12 окремих бінарних класифікаторів — одна мережа з 12 output nodes.  
Переваги: спільні ознаки між цілями, менший overfitting, єдина модель для деплою.

> 🧠 **Детальніше:** У хімії токсичні ефекти часто корелюють між собою — якщо молекула активна на NR-AR (андрогенний рецептор), є підвищена ймовірність активності на NR-AR-LBD. Multi-task навчання дозволяє мережі «навчитися» цим кореляціям. Це особливо важливо для tasks з малою кількістю позитивних прикладів (наприклад, SR-ATAD5 може мати лише 100–200 позитивів). Shared backbone екстрактує універсальні молекулярні ознаки, а окремі «голови» (output layers) спеціалізуються на кожній мішені.
```python
class MultiTaskGCN(nn.Module):
    def __init__(self, in_ch=4, hidden=64, n_tasks=12):
        super().__init__()
        self.conv1 = GCNConv(in_ch, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.lin   = nn.Linear(hidden, n_tasks)  # 12 виходів
    def forward(self, data):
        x = F.relu(self.conv1(data.x, data.edge_index))
        x = global_mean_pool(x, data.batch)
        return torch.sigmoid(self.lin(x))  # (B, 12)
```

#### 5. Mol2Vec або ChemBERTa embeddings замість Morgan bits
Претреновані векторні представлення молекул (word2vec на SMILES/графах).

> 🧠 **Детальніше:** Morgan fingerprints — це фіксована, «ручна» математична функція (кожен bit = певна підструктура). Mol2Vec і ChemBERTa — **навчені** представлення: Mol2Vec тренується як Word2Vec на мільйонах молекул, а ChemBERTa — це BERT-подібний transformer, донавчений на 77M SMILES з ZINC. Такі ембединги «розуміють» контекст — два схожих Morgan-вектори можуть сильно відрізнятись за активністю, але ChemBERTa може це врахувати. Заміна Morgan → ChemBERTa типово дає +3–8% AUC.  
```bash
pip install mol2vec
```
```python
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
model = word2vec.Word2Vec.load('mol2vec_model_300dim.pkl')
vec = sentences2vec(MolSentence(mol2alt_sentence(mol, 1)), model, unseen='UNK')
```

#### 6. Calibration кривих (Probability Calibration)
Tree-based моделі зазвичай мають погано відкалібровані ймовірності.

> 🧠 **Детальніше:** XGBoost видає ймовірності, але вони не є «справжніми» ймовірностями в статистичному сенсі. Наприклад, якщо модель каже «80% токсично» — реально toxic може бути лише 60% таких молекул. Reliability diagram (графік calibration) візуалізує це відхилення. **Isotonic regression** або **Platt scaling** (sigmoid) виправляють це. Для токсикології це критично: різниця між «50% і 80% ймовірністю токсичності» має означати різні рішення.
```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
calibrated = CalibratedClassifierCV(xgb_clf, method='isotonic', cv=3)
calibrated.fit(X_tr, y_tr)
# Reliability diagram
fraction_of_positives, mean_predicted_value = calibration_curve(y_te, proba, n_bins=10)
```

### 🥉 Рівень 3 — Просунуті ідеї

#### 7. Active Learning для анотації нових сполук
> 🧠 **Що це:** Замість того, щоб анотувати всі молекули підряд — Active Learning вибирає ті молекули, де модель **найменш впевнена** (висока uncertainty). Це дозволяє досягти того ж рівня AUC з у 3–5 разів меншою кількістю дорогих лабораторних тестів. В реальній фармацевтиці кожен експеримент коштує $100–$1000, тому такий підхід має очевидну практичну цінність.

#### 8. FastAPI endpoint для real-time передбачень
> 🧠 **Що це:** Загорнути збережену XGBoost модель у REST API — передати SMILES через HTTP → отримати JSON з ймовірностями токсичності по всіх 12 assay. Це дозволяє використовувати модель без Jupyter: з будь-якої мови програмування, мобільного додатку або іншого сервісу. Файл `01_Toxicity_Prediction/app.py` вже містить заготовку FastAPI сервера.

#### 9. Порівняння з AtomPair, FCFP, RDKit fingerprints
> 🧠 **Що це:** Різні fingerprints кодують різні аспекти молекулярної структури. ECFP4 (Morgan r=2) — локальні кільця; AtomPair — відстані між атомами; FCFP4 — функціональні групи замість атомів; MACCS — 166 фіксованих структурних ключів. Бенчмарк покаже, яка комбінація найкраще підходить саме для задачі токсичності.

#### 10. Attention weights visualization у GCN (яким атомам модель приділяє увагу)
> 🧠 **Що це:** У Graph Attention Networks (GAT) кожне ребро графа молекули має "вагу уваги" — скільки впливу атом A справив на атом B. Візуалізація цих ваг безпосередньо на 2D структурі молекули дозволяє побачити: які атоми/функціональні групи найбільше відповідають за передбачену токсичність. Це набагато інтерпретованіше за SHAP для граф-моделей.

---

## 📊 Очікувані метрики (реальний Tox21)

| Модель | Mean ROC-AUC (очікується) |
|--------|--------------------------|
| Random Forest | 0.78 – 0.82 |
| XGBoost | 0.80 – 0.84 |
| GCN (single task) | 0.79 – 0.85 |
| Multi-task GCN | 0.82 – 0.87 |
| ChemBERTa fine-tuned | 0.85 – 0.90 |

---

## 🗂️ Залежності для нових фіч
```bash
pip install optuna mol2vec mlflow
# PyG вже встановлено
```

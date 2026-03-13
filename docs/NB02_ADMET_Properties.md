# NB02 — ADMET Properties Prediction
**Файл:** `02_ADMET_Properties/ADMET_Properties_Prediction.ipynb`  
**Задача:** Регресія — передбачення logS (розчинність), logP (ліпофільність)  
**Датасети:** ESOL (1,128 молекул), Lipophilicity (4,200 молекул)

---

## 🎯 Для чого цей ноутбук і чого ми досягаємо

### Контекст: ADMET — ключ до виживання ліків

Активна молекула, яка вбиває ракову клітину в пробірці — це лише початок. Щоб стати ліками, ця молекула повинна пройти крізь **ADMET фільтр**: потрапити в організм, дістатися до органу-мішені, не зруйнуватися по дорозі, вийти з організму і при цьому не вбити пацієнта. Статистика невблаганна: **~40% кандидатів** провалюють клінічні випробування саме через погані ADMET властивості — не через брак активності.

**ADMET** розшифровується як:
- **A**bsorption — всмоктування: чи потрапить ліки з ШКТ у кров? → залежить від розчинності (logS) і ліпофільності (logP)
- **D**istribution — розподіл: чи дістанеться до мозку? до пухлини? → залежить від BBB проникності, зв'язування з білками
- **M**etabolism — метаболізм: як швидко печінка розкладе молекулу? → CYP3A4 ферменти
- **E**xcretion — виведення: чи накопичуватиметься в нирках?
- **T**oxicity — токсичність: hERG кардіотоксичність, гепатотоксичність (це NB01)

**NB02 фокусується на A і D:** передбачаємо **logS** (водна розчинність) і **logP** (ліпофільність) — дві фізико-хімічні властивості, які стоять на «вхідних воротах» будь-якого орального ліки.

### Чому logS і logP — це найважливіші відправні точки

**logS (логарифм водної розчинності)** — одиниця log(моль/л). Погана розчинність = ліки не розчиняться в шлунковому соку → не всмокчеться → не подіє. Більшість фармацевтичних компаній відсіюють кандидатів з logS < -5 вже на старті. Вимірювати розчинність лабораторно — дорого (~$100/молекула). Предиктивна модель за ~$0.001/молекула.

**logP (коефіцієнт розподілу октанол/вода)** — міра балансу між гідрофільністю та ліпофільністю. Золоте правило: logP 1–3 = оптимум для перорального ліки (досить ліпофільний для проникнення через мембрани, але не настільки, щоб не розчинитися). LogP входить до всіх п'яти правил Ліпінського.

**Правила Ліпінського (Rule of Five)** — емпіричні фільтри ораль­ної біодоступності:
- MW ≤ 500 Da
- LogP ≤ 5
- HBD ≤ 5 (донори водневих зв'язків)
- HBA ≤ 10 (акцептори водневих зв'язків)

Молекула, яка порушує 2+ правила, має <10% шансів стати оральним ліками.

### Яку задачу ми вирішуємо технічно

Це **задача регресії**: на вході SMILES → на виході неперервне число (logS або logP). На відміну від класифікації немає порогу «добре/погано» — модель має навчитися передбачати точне значення.

**Вхід:** SMILES → Morgan fingerprint 2048 bit + 9 RDKit дескрипторів = вектор 2057D  
**Вихід:** logS (mol/L, від −10 до +2) або logP (від −3 до +7)  
**Метрики:** RMSE (Root Mean Square Error), R² (коефіцієнт детермінації), MAE  
**Датасети:**
- **ESOL (Delaney, 2004)** — 1,128 молекул з виміряним logS, golden standard для QSAR регресії. Мета: RMSE < 1.0 log(mol/L) — це рівень публікацій.
- **Lipophilicity** — 4,200 молекул з виміряним logP з ChEMBL.

### Яка архітектура моделей і чому саме так

Ми порівнюємо **три підходи**, від простого до складного:

1. **Ridge Regression** (linear baseline) — показує, чи є взагалі лінійний зв'язок між fingerprint і logS. Зазвичай RMSE ~1.1–1.3. Якщо наша складна модель не краща за Ridge — щось пішло не так.

2. **Random Forest / XGBoost** (ensemble tree methods) — ловлять нелінійні залежності, стійкі до шуму. XGBoost зазвичай дає RMSE ~0.85–1.0 на ESOL.

3. **Deep Neural Network** (DNN) — `2057 → 512 → 256 → 128 → 64 → 1` з BatchNorm, Dropout, Adam optimizer. Вчиться ієрархічним представленням. При достатній кількості даних і правильній регуляризації — конкурує з XGBoost.

Додатково: **AttentiveFP** (graph attention network) — SOTA для ESOL з RMSE ~0.59.

### Що ми робимо покроково

1. **Завантажуємо ESOL і Lipophilicity** датасети (через DeepChem або CSV)
2. **Генеруємо дескриптори:** Morgan FP + RDKit descriptors для кожної молекули
3. **5-fold cross-validation** на тренувальному наборі — оцінюємо RMSE/R²/MAE
4. **Тренуємо найкращу модель** (XGBoost) на повному train set
5. **Parity plot** — scatter plot «реальне vs передбачене» з кольоровими residuals
6. **DNN з Learning Rate Scheduler** — ReduceLROnPlateau, training curves
7. **Heatmap кореляцій** дескрипторів
8. **Хімічний простір** (MW vs LogP), UMAP/PCA проекція
9. **py3Dmol 3D конформер** для найкраще передбаченої молекули
10. **MLflow** — логування RMSE/R²/MAE для кожного run

### Що ми досягаємо в результаті

✅ **Практичний результат:** Система за мілісекунди передбачає logS і logP для будь-якої молекули. Це дозволяє фільтрувати бібліотеки (тисячі/мільйони сполук) на ADMET-відповідність ще до синтезу.

✅ **RMSE < 1.0 log(mol/L)** на ESOL — це рівень публікацій (benchmark literature: AttentiveFP ~0.59, XGBoost ~0.85–1.0).

✅ **Drug-likeness radar charts** — візуальний інструмент для швидкої оцінки Lipinski профілю будь-якої молекули: 6 нормалізованих осей у Plotly radar chart.

✅ **Підставка для NB05:** Передбачений logS і logP входять до DrugScore формули кластерного ранжування. Молекули з поганим ADMET автоматично отримують нижчий score.

✅ **Streamlit dashboard:** Сторінка NB02 в dashboard дозволяє введення SMILES → миттєве передбачення ADMET + Lipinski assessment + radar chart + drug-likeness оцінка.

✅ **Портфоліо-цінність:** Демонструє вміння: (1) вирішувати регресійні задачі в хімії, (2) порівнювати лінійні/ensemble/DNN підходи, (3) правильно оцінювати через CV, (4) інтерпретувати результати через parity plots. ADMET prediction — один з найзатребуваніших напрямків у computational drug discovery.

---

## ✅ Що зроблено

### Архітектура пайплайну
| Крок | Реалізація |
|------|-----------|
| Завантаження даних | DeepChem `load_delaney()` + `load_lipo()` з scaffold split; синтетичний fallback |
| Молекулярні ознаки | Morgan fingerprints (2048 bit) + 9 RDKit дескрипторів → 2057D вектор |
| Моделі | Ridge Regression (baseline), Random Forest Regressor, XGBoost Regressor |
| Deep Learning | DNN: `Linear(2057→512→256→128→64→1)` з BatchNorm + Dropout |
| Валідація | 5-fold K-Fold CV; RMSE / R² / MAE |
| Візуалізація | Parity plot, Residual plot, Metrics bar chart, DNN learning curve |
| Explainability | Descriptor correlation heatmap, Хімічний простір (MW vs LogP, TPSA vs HBA) |
| Регуляризація DNN | BatchNorm + Dropout (0.3, 0.2, 0.1) + Adam weight_decay=1e-4 |
| LR Scheduler | ReduceLROnPlateau (patience=5, factor=0.5) |

### Баги
> ✅ Жодних багів не знайдено — ноутбук чистий.

---

## 🔴 Що залишилось / Що покращити

### 🥇 Рівень 1 — Обов'язково для сильного портфоліо

#### 1. Розширити до повного ADMET (не тільки logS + logP)
Поточний ноутбук покриває лише 2 з 5 ADMET цілей. Додати:

> 🧠 **Детальніше про ADMET:** ADMET = **A**bsorption (всмоктування) · **D**istribution (розподіл) · **M**etabolism (метаболізм) · **E**xcretion (виведення) · **T**oxicity (токсичність). Більшість ліків-кандидатів провалюються не через недостатню активність, а саме через погані ADMET профілі (~40% відмов на клінічних стадіях). Тому: logS (розчинність) → всмоктування; BBB → розподіл у мозок (важливо для CNS drugs); CYP3A4 → метаболізм; hERG → серцева токсичність (QT prolongation). Модель, яка покриває всі ці endpoints, є набагато кориснішою ніж одна, яка передбачає лише logS.

| Endpoint | Датасет | Тип задачі |
|----------|---------|-----------|
| BBB Permeability | BBBP (MoleculeNet) | Бінарна класифікація |
| CYP3A4 Inhibition | CYP (DeepChem) | Бінарна класифікація |
| hERG Cardiotoxicity | hERG (ChEMBL) | Регресія / Класифікація |
| Plasma Protein Binding | PPBDB | Регресія |

```python
# Завантаження BBBP
tasks_bbbp, ds_bbbp, _ = dc.molnet.load_bbbp(featurizer='Raw', splitter='scaffold')
```

#### 2. Multi-task DNN — всі ADMET endpoints одночасно
Поточний DNN → 1 endpoint. Multi-task → 1 модель для всіх одночасно (shared backbone).

> 🧠 **Детальніше:** Shared backbone — це частина нейронної мережі, яка навчається спільним ознакам для всіх задач. Фізико-хімічна інтуїція підказує, що це правильно: висока ліпофільність (logP) одночасно пов'язана з низькою розчинністю і кращою проникністю через мембрани (BBB). Мережа може «відкрити» ці кореляції самостійно. Технічно: замість `nn.Linear(256, 1)` — `nn.ModuleList([nn.Linear(256, 1) for _ in range(n_tasks)])`, а loss = сума MSE по всіх наявних endpoint-ах (NaN-masking для відсутніх значень).
```python
class MultiTaskADMETDNN(nn.Module):
    def __init__(self, in_dim, n_tasks):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 256),   nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.2),
        )
        # Окрема голова для кожного endpoint
        self.heads = nn.ModuleList([nn.Linear(256, 1) for _ in range(n_tasks)])
    def forward(self, x):
        h = self.shared(x)
        return torch.cat([head(h) for head in self.heads], dim=1)
```

#### 3. Збереження моделей і scaler
```python
import joblib
joblib.dump(best_model, 'models/xgb_esol.pkl')
joblib.dump(scaler, 'models/scaler_esol.pkl')
torch.save(dnn.state_dict(), 'models/dnn_esol.pt')
```

### 🥈 Рівень 2 — Значно підвищить рівень

#### 4. Attentive FP (AttentiveFP) — sota модель для молекулярних властивостей
State-of-the-art графова модель від Xiong et al. (2020). Вже доступна в DeepChem.

> 🧠 **Детальніше:** AttentiveFP — це Graph Attention Network спеціально розроблена для молекул. Ключова відмінність від звичайного GCN: attention mechanism дозволяє кожному атому «вирішувати», яких сусідів слухати і наскільки. Наприклад, карбонільна група може активно «слухати» сусідній гідроксил, але ігнорувати далекі алкільні ланцюги. Це дуже природньо моделює хімічну інтуїцію. На ESOL benchmark AttentiveFP досягає RMSE ~0.59 vs ~0.85 для XGBoost — покращення на ~30%.
```python
model = dc.models.AttentiveFPModel(n_tasks=1, mode='regression', batch_size=32, learning_rate=0.001)
model.fit(train_ds, nb_epoch=50)
```

#### 5. Applicability Domain (AD) — коли модель "не впевнена"
Визначення, чи нова молекула потрапляє в хімічний простір Training Set.

> 🧠 **Детальніше:** Будь-яка ML модель екстраполює за межами training data — і екстраполяція в хімії буває вкрай ненадійною. AD визначає «зону довіри»: якщо нова молекула близька до training molecules (Tanimoto similarity > 0.3, або LOF score > threshold) — передбачення надійне. Якщо ні — треба попередити користувача. LOF (Local Outlier Factor) вимірює, наскільки молекула «самотня» у feature space відносно своїх K найближчих сусідів. Це особливо важливо для регуляторних застосувань (REACH, FDA) де модель повинна знати межі своєї застосовності.
```python
from sklearn.neighbors import LocalOutlierFactor
lof = LocalOutlierFactor(n_neighbors=20, novelty=True)
lof.fit(X_train)
ad_scores = lof.decision_function(X_test)  # від'ємні → поза доменом
# Попереджувати, якщо молекула поза AD
```

#### 6. Attentive FP vs DNN vs RF — повне порівняння на бенчмарці
Порівняти з опублікованими результатами MoleculeNet leaderboard.

#### 7. Uncertainty Quantification з MC Dropout
> 🧠 **Детальніше:** MC (Monte Carlo) Dropout — простий але ефективний спосіб отримати розподіл передбачень замість однієї точкової оцінки. Фокус: під час inference залишати dropout **увімкненим** і робити 100 forward passes. Кожен раз мережа ігнорує різні нейрони → різні передбачення. Стандартне відхилення цих 100 значень = **uncertainty**. Молекули з великою uncertainty — найкраще кандидати для дорогих лабораторних тестів (Active Learning). Технічно: `model.train()` замість `model.eval()` під час inference.

```python
def mc_predict(model, X, n_samples=100):
    model.train()  # Увімкнути dropout під час inference!
    preds = [model(X).detach().cpu().numpy() for _ in range(n_samples)]
    return np.mean(preds, axis=0), np.std(preds, axis=0)

mean_pred, uncertainty = mc_predict(dnn, X_test_tensor)
# Плоти: точки з великою uncertainty червоні
```

### 🥉 Рівень 3 — Просунуті ідеї

#### 8. Порівняння з RDKit's own logP calculator (Crippen) — baseline
> 🧠 **Що це:** RDKit має вбудований детерміністичний розрахунок logP за правилами Crippen (фрагментний підхід, 68 атомних внесків). Це «нульовий рівень» — якщо ваша ML модель не перевищує Crippen на logP, значить щось пішло не так. А для logS (розчинність) — порівняйте з формулою Yalkowsky. Це особливо важливо для презентації результатів: «наша XGBoost модель краща ніж класичний Crippen на 15% RMSE».

#### 9. Transfer learning: GROVER або MolBERT fine-tuned на ESOL
> 🧠 **Що це:** GROVER (Graph-level Molecular pre-training via variational objectives) і MolBERT — self-supervised pretrained графові моделі, навчені на 10M+ молекул без будь-яких лейблів (лише структура). Потім їх fine-tune на ваших ~1000 labeled molecules. Аналогія: замість навчання читати з нуля — взяти людину, яка вже вміє читати, і навчити лише читати хімічні формули. Fine-tuning на 1000 прикладів з таким pretraining часто б'є XGBoost на 2000 прикладів.

#### 10. Ensemble (stacking) RF + XGBoost + DNN → meta-learner (Ridge)
> 🧠 **Що це:** Stacking — двохрівнева ансамблева техніка. Рівень 1: Ridge, RF, XGBoost роблять out-of-fold передбачення. Рівень 2: Meta-learner (ще одна Ridge або LogReg) навчається **на передбаченнях** рівня 1. Ключова перевага: meta-learner «дізнається», коли довіряти RF, а коли XGBoost. На практиці stacking дає +1–3% RMSE відносно найкращої одиночної моделі. Важливо: обов'язково використовувати out-of-fold predictions для тренування meta-learner, щоб уникнути data leakage.
```python
from sklearn.ensemble import StackingRegressor
estimators = [('rf', rf_model), ('xgb', xgb_model)]
stacking = StackingRegressor(estimators=estimators, final_estimator=Ridge())
stacking.fit(X_train, y_train)
```

---

## 📊 Очікувані метрики (реальний ESOL)

| Модель | RMSE (log mol/L) | R² |
|--------|-----------------|-----|
| Ridge Regression | 1.1 – 1.3 | 0.65 – 0.75 |
| Random Forest | 0.9 – 1.1 | 0.77 – 0.83 |
| XGBoost | 0.85 – 1.05 | 0.79 – 0.85 |
| DNN (ours) | 0.85 – 1.0 | 0.80 – 0.86 |
| AttentiveFP (SOTA) | ~0.59 | ~0.92 |

> Ціль: **RMSE < 1.0** на ESOL — це вважається "хорошим" результатом для публікації.

---

## 🗂️ Залежності для нових фіч
```bash
pip install joblib deepchem  # deepchem вже є
# AttentiveFP включено в deepchem>=2.7
```

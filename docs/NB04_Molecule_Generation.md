# NB04 — Molecule Generation (LSTM + VAE)
**Файл:** `04_Molecule_Generation/Molecule_Generation_VAE.ipynb`  
**Задача:** De novo генерація drug-like молекул  
**Архітектура:** LSTM Language Model → Variational Autoencoder (VAE)  
**Датасет:** Curated FDA/ChEMBL SMILES (30 сполук × 100 = 3000 train sequences)

---

## 🎯 Для чого цей ноутбук і чого ми досягаємо

### Контекст: навіщо генерувати молекули

Всі попередні ноутбуки (NB01–NB03) вирішують задачу **оцінки існуючих молекул**: дати їй SMILES — отримати передбачення токсичності, розчинності, активності. Але що якби ми хотіли не оцінювати, а **створювати** нові молекули з потрібними властивостями?

Традиційна medicinal chemistry так і працює: хімік бере відому активну молекулу (hit) і вручну модифікує її (замінює групи, додає кільця) — це займає **місяці** і вимагає глибокої експертизи. ML-генеративні моделі роблять це **автоматично й у масштабі**: генерують тисячі нових молекул, яких ще не існує в жодній базі даних, і при цьому спрямовуються до заданих властивостей.

Це найбільш «сексуальна» і футуристична частина AI drug discovery — саме тут знаходяться Insilico Medicine, Recursion Pharmaceuticals, Exscientia, які вже вивели AI-генеровані молекули в клінічні випробування.

### Два підходи у NB04

#### Підхід 1: LSTM Language Model (SMILES як «мова»)

Ключова ідея: **SMILES — це текст**. `CC(=O)Oc1ccccc1C(=O)O` — це рядок символів зі своїм алфавітом і граматикою. Якщо GPT навчили «писати» тексти, навчимо LSTM «писати» SMILES.

Модель навчається на корпусі drug-like SMILES і вчиться передбачати наступний токен (`C`, `(`, `=`, `O`, цифри кілець тощо) з урахуванням попереднього контексту. При генерації запускаємо autoregressive sampling: подаємо `<SOS>` → модель генерує токен за токеном до `<EOS>`. Temperature scaling контролює «випадковість»:
- T=1.0 → більша різноманітність, менша якість
- T=0.7 → консервативніші, якісніші молекули

**Метрики VUN (Validity, Uniqueness, Novelty):**
- Validity: % молекул, які RDKit може розпарсити
- Uniqueness: % унікальних серед valid
- Novelty: % тих, яких немає в training set

#### Підхід 2: VAE (Variational Autoencoder)

LSTM просто «наслідує» training distribution — не можна сказати «дай молекулу з logP=2.5». VAE вирішує це: він кодує кожну молекулу в **точку в latent space** (64-вимірний вектор), і цей простір має красиву властивість — **безперервність** і **змістовність**.

Архітектура:
- **Encoder:** Bidirectional GRU → μ и log σ² (параметри нормального розподілу)
- **Latent space:** z ~ N(μ, σ²) — «молекулярна ДНК» у 64D
- **Decoder:** GRU → послідовність токенів
- **Loss:** ELBO = Reconstruction Loss (CrossEntropy) + β·KL Divergence

ВAE дозволяє:
1. **Interpolation:** Паracetamol → Caffeine: переміщаємося між двома точками в latent space і декодуємо проміжні точки → «молекулярний морфінг»
2. **Targeted generation:** знайти в latent space напрямок «збільшення QED» і рухатися в цьому напрямку
3. **Latent space visualization:** UMAP проекція latent vectors → бачимо, як VAE організував хімічний простір

### Що означає «drug-like» і як ми це вимірюємо

**QED (Quantitative Estimate of Drug-likeness)** —综合метрика від 0 до 1, яка зважено враховує MW, logP, HBD, HBA, PSA, RotBonds, ArRings, Alerts. FDA-схвалені ліки мають медіану QED ~0.67. Ціль генерації: більшість (>50%) згенерованих молекул повинні мати QED > 0.5.

**SA Score (Synthetic Accessibility)** — наскільки реально синтезувати молекулу (1=легко, 10=practically impossible). Drug-like ціль: SA < 4.

### Що ми робимо покроково

1. **Корпус SMILES:** 30 реальних FDA/ChEMBL drug SMILES × 100 копій = 3,000 sequences (для демо; ідеально — 250k ZINC)
2. **Tokenization:** Regex-парсер → SMILES токени (підтримує Cl, Br, [NH+], @, / тощо) → vocab ~35–45 токенів
3. **LSTM тренування:** 2-layer LSTM з teacher forcing, CrossEntropyLoss, Adam
4. **Sampling з temperature** → генерація ~1,000 SMILES → VUN метрики
5. **VAE тренування:** Bidirectional GRU encoder, GRU decoder, ELBO loss з KL annealing
6. **Latent interpolation:** Paracetamol ↔ Caffeine — 8 проміжних молекул
7. **Property analysis:** QED, MW, LogP розподіли train vs generated (violin plots)
8. **UMAP latent space:** scatter Training vs Generated, розмір = QED
9. **py3Dmol 3D conformer** для top-3 by QED молекул
10. **MLflow:** validity, uniqueness, novelty, mean_QED, mean_SA на кожен sampling run

### Що ми досягаємо в результаті

✅ **Практичний результат:** Система, яка генерує нові молекули «з нуля» — тисячі хімічних структур, яких не існує у жодній базі даних, але які структурно схожі на відомі ліки.

✅ **De novo drug design pipeline:** LSTM/VAE → фільтр Lipinski + QED + SA → передбачення токсичності (NB01) → передбачення активності (NB03) → топ-кандидати → NB05 clustering.

✅ **Latent space navigation:** VAE дозволяє «подорожувати» хімічним простором — від однієї відомої молекули до іншої крізь невідомий хімічний ландшафт.

✅ **Навчальна цінність:** Цей ноутбук вимагає розуміння найскладніших концепцій: Language Models на SMILES, reparameterization trick у VAE, posterior collapse і KL annealing, SELFIES як альтернатива. Це рівень ML Research Scientist.

✅ **Портфоліо-цінність:** Generative AI for drug discovery — це **найгарячіший** напрямок у pharmatech (Insilico Medicine, Recursion, Exscientia залучили сотні мільйонів $ саме на це). Реалізація LSTM + VAE + latent interpolation + VUN metrics на реальних SMILES — це конкурентоспроможне портфоліо.

---

## ✅ Що зроблено

### Архітектура пайплайну
| Крок | Реалізація |
|------|-----------|
| Корпус | 30 real drug SMILES × 100 = 3000 SMILES; Lipinski-compliant |
| Токенізація | Regex-based (підтримка Cl, Br, [NH], @ тощо); vocab ~35-45 токенів |
| LSTM | 2 шари, hidden=256, embed=128, teacher forcing, CrossEntropyLoss |
| Sampling | Autoregressive з temperature scaling (T=1.0, T=0.7) |
| Метрики gen. | Validity / Uniqueness / Novelty (VUN) |
| VAE Encoder | Bidirectional GRU → μ / log σ² (latent_dim=64) |
| VAE Decoder | GRU з teacher forcing → token logits |
| VAE Loss | ELBO = Reconstruction (CE) + β·KL divergence (β=0.5) |
| Latent интерполяція | Paracetamol → Caffeine (8 кроків) |
| Аналіз властивостей | MW, logP, QED: train vs generated distributions |
| Візуалізація | Training curves (LSTM loss, VAE ELBO/Recon/KL), VUN bars, property histograms |

### Баги
> ✅ Жодних багів не знайдено.

---

## 🔴 Що залишилось / Що покращити

### 🥇 Рівень 1 — Обов'язково для сильного портфоліо

#### 1. Збільшити Training Corpus (критично важливо!)
Поточний корпус: 30 SMILES × 100 = 3000 — **занадто мало** для реальної генерації.  
Мінімум для якісних результатів: **50,000–250,000 SMILES** з ChEMBL або ZINC.

> 🧠 **Детальніше:** Мовна модель на SMILES — це буквально «мова хімії». Щоб модель навчилась «писати правильні речення» (валідні SMILES), їй потрібно побачити достатньо різних структур. З 3000 SMILES модель вивчить лише ~30 унікальних молекули (100 копій кожної) — це як навчати GPT на одній книзі. З 250k SMILES (ZINC drug-like) модель вивчає різноманітність реальних drug-like молекул: різні кільця, замісники, гетероатоми. Validity з 3k corpus: ~40–60%. З 250k corpus: ~85–96%. Це **найважливіше** покращення для цього ноутбука.

```python
# Варіант 1: завантажити ZINC drug-like subset (250k SMILES)
import urllib.request
url = "https://raw.githubusercontent.com/aspuru-guzik-group/chemical_vae/master/models/zinc_properties/250k_rndm_zinc_drugs_clean_3.csv"
urllib.request.urlretrieve(url, "zinc250k.csv")
df_zinc = pd.read_csv("zinc250k.csv")
ALL_SMILES = df_zinc['smiles'].dropna().tolist()

# Варіант 2: через DeepChem MoleculeNet
tasks, ds, _ = dc.molnet.load_zinc15(featurizer='Raw', splitter=None)
ALL_SMILES = [str(x) for x in ds[0].ids]
```

#### 2. KL Annealing (запобігти posterior collapse)
> 🧠 **Детальніше:** Posterior collapse — поширена проблема у VAE для послідовностей: декодер стає настільки потужним, що ігнорує latent space (z) і все генерує лише з текстового контексту. Ознака: KL loss → 0 дуже швидко, але reconstruction loss залишається поганим. **Рішення:** починати з β=0 (чистий автоенкодер), поступово збільшувати β до 1 протягом перших 50% епох. Тоді декодер «звикає» спочатку реконструювати, а потім поступово вчиться використовувати latent code. Без annealing ~60% запусків VAE на SMILES колапсують.

```python
def kl_weight(epoch, n_epochs, max_weight=1.0, warmup_fraction=0.5):
    """Лінійний warmup: 0 → max_weight протягом перших warmup_fraction епох."""
    warmup_epochs = int(n_epochs * warmup_fraction)
    return min(max_weight, max_weight * epoch / warmup_epochs)

for epoch in range(1, VAE_EPOCHS + 1):
    beta = kl_weight(epoch, VAE_EPOCHS)
    for xb in vae_dl:
        recon, mu, logvar = vae(xb)
        loss, rl, kl = vae_loss(recon, xb, mu, logvar, beta=beta)
```

#### 3. Валідація + QED scoring для generated molecules
```python
from rdkit.Chem import QED
valid_gen = [s for s in lstm_samples if Chem.MolFromSmiles(s) is not None]
qed_scores = [QED.qed(Chem.MolFromSmiles(s)) for s in valid_gen]
print(f"Mean QED: {np.mean(qed_scores):.3f} (> 0.5 = drug-like)")
```

### 🥈 Рівень 2 — Значно підвищить рівень

#### 4. SELFIES Encoding замість SMILES (100% валідність!)
**SELFIES** (Self-Referencing Embedded Strings) — кожна можлива рядкова послідовність є валідною молекулою. Вирішує проблему невалідних SMILES при генерації.

> 🧠 **Детальніше:** SMILES має жорсткий синтаксис — більшість випадково згенерованих SMILES-рядків є **невалідними** (незакриті дужки, неправильна валентність, тощо). VAE/LSTM можуть генерувати 20–50% невалідних SMILES навіть після навчання. SELFIES — нова система нотації (Krenn et al., 2020) спеціально розроблена як «foolproof»: кожен граматично правильний рядок SELFIES відповідає хімічно валідній молекулі за визначенням. Це як замінити вільний текст на структуровану форму — ніколи не отримаєш безглуздий результат. Ціна: трохи більший словник (~30–60 токенів), але validity → 100%.

```bash
pip install selfies
```
```python
import selfies as sf

def smiles_to_selfies_tokens(smiles):
    sel = sf.encoder(smiles)  # SMILES → SELFIES
    return list(sf.split_selfies(sel))

def selfies_to_smiles(selfies_str):
    return sf.decoder(selfies_str)  # SELFIES → SMILES (завжди валідний!)

# Замінити tokenize_smiles → smiles_to_selfies_tokens у всьому пайплайні
```

#### 5. Latent Space Visualization з t-SNE / UMAP
Закодувати training + generated molecules → порівняти розподіли:

> 🧠 **Детальніше:** Latent space VAE — це «карта хімічного простору» в 64D. UMAP спроєктує цю карту в 2D зберігаючи кластерну структуру. Ідеальна картина: training і generated molecules **перекриваються** (модель вивчила той самий хімічний простір), але generated містять і нові точки (новизна). Якщо generated далеко від training — модель генерує «хімічні нісенітниці». Якщо generated повністю всередині training — модель не навчилась нічого нового, лише копіює. Колір = QED: кластери з high-QED молекул у latent space підкажуть, в яких напрямках «рухатись» при optimized sampling.
```python
# Encode training molecules
with torch.no_grad():
    enc_data = torch.tensor([encode_smiles(s) for s in ALL_SMILES[:500]], ...)
    mu_train, _ = vae.encode(enc_data)

# Encode generated molecules
# ...

# UMAP 2D plot
from umap import UMAP
reducer = UMAP(n_components=2, random_state=42)
coords = reducer.fit_transform(np.vstack([mu_train.cpu(), mu_gen.cpu()]))
# Кольором розрізнити train vs generated
```

#### 6. Property-Conditioned Generation (CVAE)
Додати target property як додатковий вхід до decoder → генерувати молекули з заданим logP або QED.

> 🧠 **Детальніше:** Звичайний VAE генерує молекули «random» з latent space. CVAE (Conditional VAE) дозволяє сказати «хочу молекулу з LogP=3, MW=350». Це реалізується конкатенацією нормалізованого property vector до latent vector z перед декодуванням. Для drug design це надзвичайно корисно: можна генерувати молекули з конкретним ADMET профілем. Наприклад: LogP 1–3 (оптимальна розчинність + мембранна проникність), MW < 400 (краще клітинне проникнення), QED > 0.7 (drug-like).
```python
class CondVAE(MolVAE):
    def __init__(self, vocab_size, n_props=2, ...):
        # Конкатенувати z + property vector перед decoder
        self.latent2hid = nn.Linear(latent_dim + n_props, hidden_dim)

# Навчання: передавати [MW, logP] нормалізовані як умову
# Генерація: z ~ N(0,I), condition=[target_mw, target_logp]
```

### 🥉 Рівень 3 — Просунуті ідеї

#### 7. REINVENT-style Reinforcement Learning
Використовувати навчений LSTM як Prior, потім fine-tune з RL щоб максимізувати reward (QED).

> 🧠 **Детальніше:** REINVENT (AstraZeneca) — один з найвпливовіших практичних підходів до молекулярної генерації. Алгоритм:
> 1. **Prior** = навчений LSTM (знає як генерувати drug-like молекули)
> 2. **Agent** = копія Prior, яку будемо fine-tune
> 3. На кожному кроці Agent генерує SMILES → рахуємо reward (QED, predicted activity, SA score)
> 4. Loss = -log P_agent(SMILES) · (reward + KL(agent || prior))
> 5. KL член не дозволяє agent занадто сильно відхилятись від prior (уникаємо mode collapse)
>
> Результат: за 100–500 epochs agent навчається генерувати молекули з цілеспрямованими властивостями, залишаючись в «хімічно розумному» просторі.
```python
# Формула REINVENT:
# loss = -log P_agent(SMILES) · reward(SMILES)
# де reward = QED(SMILES) або predicted activity
reward = QED.qed(Chem.MolFromSmiles(generated_smiles))
agent_loss = -log_prob * reward
```

#### 8. JT-VAE (Junction Tree VAE) — найточніший метод
Оперує деревами фрагментів молекул замість SMILES рядків → набагато вища валідність.
> 🧠 **Що це:** JT-VAE (Jin et al., 2018) декомпонує молекулу на хімічно значущі фрагменти (кільця, функціональні групи) і будує «дерево монтажу». Генерація відбувається як складання Lego: спочатку вибирається каркас, потім приєднуються фрагменти. Validity → ~100% за визначенням (тільки хімічно правильні приєднання). Ціна: складніша реалізація (~1000 рядків коду) і повільніше тренування.

#### 9. Synthesis Planning Integration
Після генерації — AiZynthFinder або ASKCOS для перевірки синтетичної доступності.
> 🧠 **Що це:** Навіть якщо молекула «drug-like» за Lipinski і має гарний QED, вона може бути **нездійсненною в синтезі** (занадто багато стадій, недоступні реагенти, нестабільні інтермедіати). AiZynthFinder (AstraZeneca) і ASKCOS (MIT) — програми ретросинтетичного планування: вони «розкладають» молекулу назад на доступні комерційні реагенти через відомі хімічні реакції. Інтеграція: після генерації фільтрувати молекули з SA_score < 4 і successfull retrosynthesis.

#### 10. Порівняння з GuacaMol benchmark
Стандартний benchmark для generative models (validity, KL divergence, FCD, scaffold similarity).
```bash
pip install guacamol
```
> 🧠 **Що це:** GuacaMol (Brown et al., 2019) — стандартний benchmark для generative models від BenevolentAI. Включає: validity, uniqueness, novelty, KL divergence між розподілами властивостей, FCD (Fréchet ChemNet Distance — аналог FID для молекул), і 20 goal-directed optimization задач. Порівняти свою VAE/LSTM з лідерборд результатами — переконливий аргумент для резюме чи портфоліо.

---

## 📊 Очікувані метрики VUN (з 50k+ training SMILES)

| Модель | Validity | Uniqueness | Novelty |
|--------|----------|-----------|---------|
| LSTM (T=1.0) | 90–96% | 95–99% | 80–95% |
| LSTM (T=0.7) | 95–99% | 80–90% | 60–75% |
| VAE | 70–85% | 98–99% | 90–99% |
| CVAE | 68–82% | 97–99% | 88–98% |

> З поточним корпусом 3000 SMILES очікувати ~40–70% validity — для демо прийнятно.

---

## 🗂️ Залежності для нових фіч
```bash
pip install selfies guacamol
# deepchem для ZINC15 вже встановлено
```

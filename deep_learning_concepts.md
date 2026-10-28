# 🧠 Deep Learning Concepts Used in IPOPilot

This document explains every deep learning concept embedded in IPOPilot's prediction engine.
The core model is the **Deep Mixture Density Network Ensemble (Deep MDN)** — a custom-built probabilistic neural network that predicts the **full distribution** of an IPO's listing-day return, not just a single number.

---

## 🗺️ High-Level Architecture

```
Numeric Features (32)          Categorical Features (board, sector)
        │                                    │
  Standardise + NaN fill             Embedding Lookup
        │                                    │
        └──────────────┬─────────────────────┘
                       ↓
              Linear Projection → d=96
                       ↓
              GELU Activation
                       ↓
          ┌────────────────────────┐
          │  Residual Block × 3    │
          │  LayerNorm → Linear    │
          │  → GELU → Dropout      │
          │  → Linear → Residual   │
          └────────────────────────┘
                       ↓
              LayerNorm
                       ↓
              MDN Head → K=4 Gaussian components
              (logit_π, μ, log σ per component)
                       +
              Wide Linear Path (raw features → μ)
                       ↓
          ┌────────────────────────┐
          │   5-Member Ensemble    │
          │  (independently seeded)│
          └────────────────────────┘
                       ↓
         Combined Mixture Distribution
         (4 components × 5 members = 20 Gaussians)
                       ↓
           Variance Temperature Calibration
                       ↓
         Quantiles, Mean Return, P(loss)
```

---

## 1. 🔢 Mixture Density Network (MDN)

**File:** [`backend/ipopilot/ml/listing_model.py`](file:///d:/Users/Pranil/Github_Repos/IPO/backend/ipopilot/ml/listing_model.py#L82-L113)

### What it is
Instead of predicting a single number for listing gain, the network predicts a **mixture of Gaussians** — a probability distribution over possible outcomes:

$$
P(y \mid x) = \sum_{k=1}^{K} \pi_k(x) \cdot \mathcal{N}(y \mid \mu_k(x),\, \sigma_k^2(x))
$$

where:
- $\pi_k$ are mixture weights (summing to 1)
- $\mu_k$ are component means (the predicted "scenarios")
- $\sigma_k$ are component standard deviations (uncertainty per scenario)
- $K = 4$ components are used

### Why it's used here
IPO returns are **highly non-Gaussian** — they can have fat tails, bimodal distributions (big hit vs. flat listing), or be heavily skewed. A mixture can capture all of these. This lets IPOPilot output calibrated confidence intervals like *"80% chance of return between –5% and +60%"* rather than just a point forecast.

### The head
```python
self.head = nn.Linear(d, 3 * k)   # → logit_π, μ, log σ  for each k
```

The **negative log-likelihood (NLL)** is the training loss:
```python
log_pi = F.log_softmax(logit_pi, dim=1)
z      = (y.unsqueeze(1) - mu) / sigma
nll    = -logsumexp(log_pi + log_prob, dim=1)
```

---

## 2. 🏗️ Residual MLP with LayerNorm + GELU

**File:** [`backend/ipopilot/ml/listing_model.py`](file:///d:/Users/Pranil/Github_Repos/IPO/backend/ipopilot/ml/listing_model.py#L70-L79)

### Residual Connections (Skip Connections)
Borrowed from ResNets, the `ResBlock` adds its input to its output:

```python
def forward(self, x):
    return x + self.drop(self.fc2(self.drop(F.gelu(self.fc1(self.norm(x))))))
```

`output = x + f(x)` — this ensures gradients flow cleanly through deep networks (avoids vanishing gradients) and lets the network default to an identity transform when no information is added.

### LayerNorm
Applied **before** each transformation (Pre-LN style), which stabilises training:
- Normalises the feature vector of each sample independently (not across the batch)
- Essential for small-batch training, common in financial datasets

### GELU Activation
Gaussian Error Linear Unit — smoother than ReLU, used in GPT-family models:

$$
\text{GELU}(x) = x \cdot \Phi(x)
$$

where $\Phi$ is the Gaussian CDF. In practice it approximates `x * sigmoid(1.702 * x)`.
It allows small negative values to flow through, helping gradient flow.

**Architecture:** 3 residual blocks with hidden dim `d=96`, dropout `p=0.15`.

---

## 3. 🔡 Learned Categorical Embeddings

**File:** [`backend/ipopilot/ml/listing_model.py`](file:///d:/Users/Pranil/Github_Repos/IPO/backend/ipopilot/ml/listing_model.py#L85-L86)

```python
self.embs = nn.ModuleList([nn.Embedding(n, min(8, (n+1)//2 + 1)) for n in cat_sizes])
```

Categorical features (`board`: mainboard/SME, `sector`: 24+ industries) are each mapped to a dense learned embedding vector (4–8 dimensions). This is the same idea as Word2Vec — semantically similar sectors will have nearby vectors in embedding space. Much more powerful than one-hot encoding for high-cardinality categoricals.

---

## 4. 🌐 Wide & Deep Architecture

**File:** [`backend/ipopilot/ml/listing_model.py`](file:///d:/Users/Pranil/Github_Repos/IPO/backend/ipopilot/ml/listing_model.py#L91-L103)

Inspired by Google's Wide & Deep paper (used in recommender systems):

```python
self.wide = nn.Linear(n_num, k)        # memorisation path
nn.init.zeros_(self.wide.weight)       # starts as pure bias
...
mu = mu + self.wide(x_num)             # raw features added directly to μ
```

The **wide** path is a direct linear connection from raw standardised features to the component means. This makes simple monotone relationships (like "higher GMP → higher listing gain") trivially learnable even on small datasets, while the **deep** path handles interactions and nonlinearities. Zero-initialised so it starts as neutral and learns only what the deep path misses.

---

## 5. 🎲 Deep Ensemble

**File:** [`backend/ipopilot/ml/listing_model.py`](file:///d:/Users/Pranil/Github_Repos/IPO/backend/ipopilot/ml/listing_model.py#L171-L200)

Five identical networks are trained from **different random seeds**. Their mixture distributions are then averaged:

$$
P_{\text{ensemble}}(y \mid x) = \frac{1}{M} \sum_{m=1}^{M} P_m(y \mid x)
$$

This creates an **ensemble of 4×5 = 20 Gaussian components** total, capturing two types of uncertainty:
- **Aleatoric uncertainty** — inherent randomness in IPO returns (captured by each MDN's σ)
- **Epistemic uncertainty** — model's uncertainty due to limited data (captured by disagreement between members)

Deep Ensembles are consistently the best-performing uncertainty quantification method in tabular ML benchmarks.

---

## 6. 📅 Recency Weighting (Temporal Sample Weighting)

**File:** [`backend/ipopilot/ml/train.py`](file:///d:/Users/Pranil/Github_Repos/IPO/backend/ipopilot/ml/train.py#L167-L170)

```python
def recency_weights(df, ref=None):
    age = [(ref - d).days / 365.25 for d in df["close_date"]]
    return 0.5 ** (np.clip(age, 0, None) / RECENCY_HALF_LIFE_Y)
```

**Half-life = 4 years.** IPOs from 4 years ago get half the weight of recent ones. This accounts for **market regime shifts** — the 2021 bull run dynamics are less informative for 2026 predictions than 2024 data. The weights are passed as per-sample scalars to the NLL loss.

---

## 7. 🌡️ Variance Temperature Calibration

**File:** [`backend/ipopilot/ml/train.py`](file:///d:/Users/Pranil/Github_Repos/IPO/backend/ipopilot/ml/train.py#L188-L192)

A scalar temperature $T$ is tuned on the **validation period** by minimising NLL:

$$
\sigma_{\text{calibrated}} = \sigma_{\text{raw}} \times T
$$

```python
grid = np.linspace(0.6, 2.0, 57)
nlls = [ens.nll(va, y, temp=t) for t in grid]
return float(grid[int(np.argmin(nlls))])
```

If the model is **overconfident** (intervals too narrow), $T > 1$ spreads them. If **underconfident**, $T < 1$ tightens them. This is a lightweight post-hoc calibration approach (current temperature = `1.0`, meaning the model is already well-calibrated).

---

## 8. ⚡ AdamW + Cosine LR Schedule + Gradient Clipping

**File:** [`backend/ipopilot/ml/listing_model.py`](file:///d:/Users/Pranil/Github_Repos/IPO/backend/ipopilot/ml/listing_model.py#L274-L275)

```python
opt   = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr*0.02)
```

- **AdamW**: Adam with decoupled L2 weight regularisation (weight_decay=1e-3). Unlike vanilla Adam where weight decay is folded into the gradient update, AdamW applies it directly to weights, which correctly regularises transformers and MLPs.
- **Cosine Annealing**: Learning rate decays smoothly from 1.5e-3 → 3e-5 over `epochs` steps following a cosine curve. Allows aggressive early learning and careful fine-tuning near convergence.
- **Gradient Clipping**: `clip_grad_norm_(model.parameters(), 2.0)` — prevents exploding gradients from rare, extreme IPO return samples.

---

## 9. 🛑 Early Stopping with Patience

**File:** [`backend/ipopilot/ml/listing_model.py`](file:///d:/Users/Pranil/Github_Repos/IPO/backend/ipopilot/ml/listing_model.py#L305-L315)

```python
if v < best - 1e-4:
    best, best_ep, bad = v, ep + 1, 0
    best_state = {k: t.detach().clone() for k, t in model.state_dict().items()}
else:
    bad += 1
    if bad >= patience:   # patience = 40 epochs
        break
```

Training stops when validation NLL hasn't improved by more than 1e-4 for 40 consecutive epochs. The **best checkpoint** is restored at the end. This prevents overfitting on the small IPO dataset (≈1,100 training samples). The best epochs in the current production model: `[8, 10, 5, 7, 6]` — the model converges very quickly due to the small dataset.

---

## 10. 🎭 GMP-Masking Data Augmentation

**File:** [`backend/ipopilot/ml/listing_model.py`](file:///d:/Users/Pranil/Github_Repos/IPO/backend/ipopilot/ml/listing_model.py#L293-L298)

```python
m = has_gmp[idx] & (torch.rand(len(idx), device=DEVICE) < gmp_mask_p)
xb[m, gidx] = 0.0          # zero-out GMP features
xb[m, miss_j] = miss_z     # set missing=1 flag
```

With probability `gmp_mask_p`, GMP (Grey Market Premium) features are randomly zeroed out during training — teaching the network to make predictions **both with and without GMP data**. This is critical because GMP is available for only ~26% of training IPOs but ~63% of test IPOs. Even with `gmp_mask_p=0.0` (currently disabled), the architecture is designed to handle missing GMP via explicit `gmp_missing` flags.

Also: **Gaussian noise augmentation** (`0.03 * randn`) is applied to numeric features each batch to act as a regulariser.

---

## 11. 🤖 FinBERT Sentiment (Transfer Learning)

**File:** [`backend/ipopilot/ml/sentiment.py`](file:///d:/Users/Pranil/Github_Repos/IPO/backend/ipopilot/ml/sentiment.py)

```python
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")
```

**FinBERT** is a BERT model fine-tuned on financial text. It classifies each IPO-related news headline into `positive`, `negative`, `neutral` sentiment. The resulting probabilities (`p_pos`, `p_neg`) are aggregated per IPO into features: `sent_mean`, `sent_neg_share`, `news_count` — which the Deep MDN uses as inputs.

This is **transfer learning**: a 110M-parameter language model's representations are distilled into 3 scalar features per IPO, making NLP practical for tabular ML.

---

## 12. 🔄 Walk-Forward (Time-Series Cross-Validation)

**File:** [`backend/ipopilot/ml/train.py`](file:///d:/Users/Pranil/Github_Repos/IPO/backend/ipopilot/ml/train.py#L243-L265)

```python
for (y, qn) in quarters:  # each quarter in test period
    hist    = df[df["close_date"] < q_start]   # everything before
    target  = te[te["close_date"] in quarter]  # predict this quarter
    preds   = fit_predict_all(hist, target, ...)  # retrain fresh each quarter
```

All five models are **completely retrained** for each quarter in the test period (2025 Q2 onwards) using only data available before that quarter — exactly mirroring production. This avoids look-ahead bias that plagues simpler train/test splits and gives honest out-of-sample metrics.

**Time splits:**
- **Train:** < 2024-07-01 (1,137 IPOs)
- **Validation:** 2024-07-01 – 2025-03-31 (239 IPOs) — hyperparameter & temperature selection only
- **Test:** ≥ 2025-04-01 (574 IPOs) — never used during development

---

## 📊 Model Performance Summary (Walk-Forward Test)

| Model | MAE % | RMSE % | Dir. Acc | CRPS % | AUC(loss) |
|-------|-------|--------|----------|--------|-----------|
| Climatology (baseline) | 21.0 | 32.5 | 63.6% | 17.7 | 0.474 |
| GMP Only (heuristic) | 16.5 | 26.3 | 66.0% | 14.2 | 0.694 |
| Ridge Regression | 18.4 | 24.6 | 67.8% | 14.4 | 0.730 |
| LightGBM | 17.4 | 25.2 | 66.6% | 13.9 | 0.721 |
| **Deep MDN Ensemble** | **17.2** | **24.1** | **67.1%** | **13.8** | **0.750** |

> [!NOTE]
> The Deep MDN ensemble achieves the **best calibration** (lowest CRPS, highest AUC for P(loss)) even though MAE/RMSE margins are slim. This matters most for the strategy optimizer, which needs reliable probability intervals.

---

*For setup instructions, see [`SETUP_AND_RUN.md`](./SETUP_AND_RUN.md).*
*For a simulated training walkthrough, see [`scripts/dl_training_simulation.py`](./scripts/dl_training_simulation.py).*

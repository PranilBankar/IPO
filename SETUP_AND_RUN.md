# 🚀 IPOPilot — Setup & Run Guide

A step-by-step walkthrough to get IPOPilot running locally on **Windows (PowerShell)**.

---

## 📋 Prerequisites

Before you begin, make sure the following are installed:

| Tool | Version | Where to get it |
|------|---------|----------------|
| **Python** | 3.10 or 3.11 | https://python.org |
| **Node.js** | 18+ (LTS) | https://nodejs.org |
| **Git** | any recent | https://git-scm.com |
| **PyTorch** | 2.3+ | See Step 3 below |

> [!IMPORTANT]
> PyTorch must be installed **separately** before the rest of the Python dependencies.
> A GPU (CUDA 12.1) is optional but will make model training significantly faster.

---

## 🗂️ Project Structure (Quick Overview)

```
IPO/
├── backend/              ← Python FastAPI + ML engine
│   ├── ipopilot/
│   │   ├── api/          ← REST API routes
│   │   ├── engine/       ← Prediction, backtest, optimizer
│   │   ├── ingestion/    ← Data scraping pipeline
│   │   └── ml/           ← Deep learning models (see deep_learning_concepts.md)
│   └── requirements.txt
├── frontend/             ← Next.js web dashboard
├── data/
│   ├── ipopilot.db       ← SQLite database (auto-created)
│   ├── models/           ← Trained model files (see below)
│   └── raw/              ← HTTP cache / scraped data
├── scripts/
│   ├── setup.ps1         ← One-time full setup
│   └── start.ps1         ← Start both servers
└── .env.example          ← Environment variable template
```

---

## ⚙️ Step 1 — Clone the Repository

```powershell
git clone https://github.com/PranilBankar/IPO.git
cd IPO
```

---

## ⚙️ Step 2 — Create a `.env` File

Copy the example and edit as needed:

```powershell
Copy-Item .env.example .env
```

Open `.env` and set at minimum:

```env
# Required if you want admin endpoints
IPOPILOT_ADMIN_KEY=your-secret-key-here

# Optional: set a custom data directory (default: ./data)
# IPOPILOT_DATA_DIR=D:/my_data

# Optional: PostgreSQL (defaults to SQLite)
# DATABASE_URL=postgresql+psycopg2://user:pass@localhost/ipopilot
```

> [!NOTE]
> If you skip setting `IPOPILOT_ADMIN_KEY`, admin API endpoints will be unprotected. Fine for local dev.

---

## ⚙️ Step 3 — Install PyTorch (GPU or CPU)

Open a PowerShell terminal in the repo root.

**For NVIDIA GPU (CUDA 12.1) — recommended:**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Verify it works:
```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

---

## ⚙️ Step 4 — Install All Python Dependencies

```powershell
cd backend
pip install -r requirements.txt
cd ..
```

> [!TIP]
> Consider creating a virtual environment first:
> ```powershell
> python -m venv .venv
> .venv\Scripts\Activate.ps1
> ```

---

## ⚙️ Step 5 — Install Frontend Dependencies

```powershell
cd frontend
npm install
cd ..
```

---

## ⚙️ Step 6 — Bootstrap the Database & Ingest Data

This is the **one-time setup step** that initialises the SQLite DB, scrapes the IPO universe, market data, subscriptions, GMP history, and news headlines.

```powershell
cd backend

# Full bootstrap (includes long historical backfills — takes ~20–40 min)
python -m ipopilot.cli bootstrap

# --- OR --- Quick bootstrap (skip backfills, good for testing)
python -m ipopilot.cli bootstrap --quick
```

After bootstrap completes, run the individual backfill commands if needed:

```powershell
# Backfill per-category subscription history
python -m ipopilot.cli backfill-subs

# Backfill IPOWatch day-wise GMP history (since 2019)
python -m ipopilot.cli backfill-gmp --since 2019

# Fetch news headlines + FinBERT sentiment scores
python -m ipopilot.cli news

# Also fetch historical headlines for listed IPOs since 2023
python -m ipopilot.cli news --history
```

> [!NOTE]
> All data is stored in `data/ipopilot.db` (SQLite by default).
> Scraped HTML is cached in `data/raw/http_cache/` to avoid repeat requests.

---

## ⚙️ Step 7 — Train the Models

```powershell
# From the backend/ directory:
python -m ipopilot.cli train

# Custom epochs and ensemble size:
python -m ipopilot.cli train --epochs 200 --ensemble 5
```

This runs the full walk-forward evaluation pipeline and saves:

| File | Description |
|------|-------------|
| `data/models/listing_current/members.pt` | PyTorch ensemble weights (Deep MDN) |
| `data/models/listing_current/config.json` | Preprocessor vocab, arch config, temperature |
| `data/models/metrics.json` | Full evaluation report (MAE, RMSE, CRPS, AUC, etc.) |
| `data/models/test_predictions.pkl` | Walk-forward predictions for all test IPOs |

Expected training time: **~15–30 minutes** on CPU, **~5 minutes** on GPU.

See [`deep_learning_concepts.md`](./deep_learning_concepts.md) for what the model is doing.

---

## ⚙️ Step 8 — Run the Strategy Backtest (Optional)

```powershell
python -m ipopilot.cli backtest
```

Results are printed as JSON and the output is also saved to `data/models/backtest.json`.

---

## ⚙️ Step 9 — Start the Application

### Option A — One Command (Recommended)

From the repo root:

```powershell
.\scripts\start.ps1
```

This opens two PowerShell windows (backend + frontend) and automatically opens `http://localhost:3000` in your browser after 6 seconds.

### Option B — Start Each Server Manually

**Backend (FastAPI on port 8000):**
```powershell
cd backend
python -m ipopilot.cli serve --host 127.0.0.1 --port 8000
```

**Frontend (Next.js dev server on port 3000):**
```powershell
cd frontend
npm run dev
```

Then open: [http://localhost:3000](http://localhost:3000)

### Option C — Production Build

```powershell
.\scripts\start.ps1 -Prod
```

---

## ⚙️ Step 10 — One-Shot Automated Setup (Alternative to Steps 3–8)

The `setup.ps1` script runs steps 3–8 in one go:

```powershell
# From repo root (install PyTorch first manually, then):
.\scripts\setup.ps1

# Quick version (skips long backfills):
.\scripts\setup.ps1 -Quick
```

---

## 🐳 Docker (Optional)

A `docker-compose.yml` is provided for containerised deployment:

```powershell
docker-compose up --build
```

- Backend API: `http://localhost:8000`
- Frontend: `http://localhost:3000`

---

## 🔍 Where Are the Current Model Files?

The currently trained production model is stored at:

```
data/
└── models/
    ├── listing_current/
    │   ├── members.pt          ← PyTorch ensemble weights (~2.4 MB, 5 members)
    │   └── config.json         ← Arch config + preprocessor vocab + temperature
    ├── metrics.json            ← Latest evaluation report (version: deep_mdn_20261018)
    ├── backtest.json           ← Strategy backtest results
    └── test_predictions.pkl    ← Walk-forward test set predictions (~170 KB)
```

**Model version:** `deep_mdn_20261018` (trained 2026-10-18, on 1,950 IPOs total)

**Key metrics (walk-forward test, 574 IPOs):**
- MAE: **17.2%** | RMSE: **24.1%** | Directional Accuracy: **67.1%**
- 80% Coverage: **83.4%** | CRPS: **13.75%** | AUC(loss): **0.750**

---

## 🔧 Useful CLI Commands Reference

```powershell
# From backend/ directory

python -m ipopilot.cli bootstrap          # Full first-run ingestion
python -m ipopilot.cli bootstrap --quick  # Quick bootstrap (skip backfills)
python -m ipopilot.cli sync               # Refresh live IPOs + GMP
python -m ipopilot.cli backfill-subs      # Category subscription history
python -m ipopilot.cli backfill-gmp       # Day-wise GMP history
python -m ipopilot.cli news               # Headlines + FinBERT sentiment
python -m ipopilot.cli news --history     # + Historical headlines since 2023
python -m ipopilot.cli train              # Train + evaluate all models
python -m ipopilot.cli train --epochs 300 --ensemble 7  # Custom run
python -m ipopilot.cli backtest           # Strategy backtest
python -m ipopilot.cli predict            # Refresh live predictions
python -m ipopilot.cli serve              # Start API server (:8000)
```

---

## ❓ Common Issues

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError: torch` | Run Step 3 again — PyTorch must be installed before `requirements.txt` |
| `CORS error` in browser | Check `IPOPILOT_CORS_ORIGINS` in `.env` includes your frontend URL |
| Training crashes (OOM) | Reduce `--ensemble` size or use CPU-only PyTorch |
| SQLite database locked | Close any other process holding `data/ipopilot.db` |
| FinBERT download hangs | Set `IPOPILOT_SENTIMENT_MODEL` to a local Hugging Face cache path |
| `npm install` fails | Ensure Node.js ≥18 is installed; try `npm install --legacy-peer-deps` |

---

*For deep learning concepts used in this project, see [`deep_learning_concepts.md`](./deep_learning_concepts.md).*

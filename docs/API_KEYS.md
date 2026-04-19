# API Keys & Model Hosting Guide

> **Goal:** Get every external dependency wired up using only **free tiers**.
> **Audience:** Solo dev (you) running ProTrader AI locally and on Streamlit Cloud.
> **Scope:** Every key the codebase touches + step-by-step upload of the v2 model
> from `sentiment_analysis_v2.py` to HuggingFace Hub so production can pull it.

---

## 0. TL;DR — what you actually need

| Tier               | Key / Account                       | Required? | Free?            |
|--------------------|-------------------------------------|-----------|------------------|
| **Must-have**      | None — yfinance + RSS work keyless  | -         | -                |
| **Strongly rec.**  | NewsAPI                             | optional  | yes (100 req/day)|
| **Strongly rec.**  | HuggingFace Hub (host v2 model)     | for A2    | yes              |
| **Nice-to-have**   | Reddit (PRAW)                       | optional  | yes              |
| **Nice-to-have**   | Google Gemini                       | optional  | yes (60 RPM)     |
| **Nice-to-have**   | Roboflow (chart pattern vision)     | optional  | yes (1k inf/mo)  |
| **For intraday**   | Dhan API **or** Upstox API          | optional  | yes (KYC only)   |

The app **runs with zero keys** — every key just unlocks a richer feature.
Keep this file ungitignored; it's instructions, not secrets. Real secrets go
in `.env` (local) and Streamlit Cloud → **Settings → Secrets** (cloud).

---

## 1. Storage convention

The codebase uses [config/settings.py](../config/settings.py) `_get_secret()`
which falls back through:

1. `st.secrets["KEY"]` (Streamlit Cloud / local `.streamlit/secrets.toml`)
2. `.env` file via `python-dotenv`
3. OS environment variable
4. Default empty string

So you have three legal places to put a key:

**Local dev — `.env`** (gitignored):
```
NEWS_API_KEY=abc123
GEMINI_API_KEY=xyz789
```

**Local dev with Streamlit secrets — `.streamlit/secrets.toml`** (gitignored):
```toml
NEWS_API_KEY = "abc123"
GEMINI_API_KEY = "xyz789"
```

**Streamlit Cloud** → app dashboard → **Settings → Secrets**, paste the same TOML.

---

## 2. NewsAPI (financial headlines)

Used by [data/multi_sentiment.py](../data/multi_sentiment.py) `fetch_newsapi()`.
Without this key, the app falls back to RSS only (still works, just thinner).

**Steps:**
1. Go to https://newsapi.org/register
2. Sign up with email — no card needed.
3. Verify email → you'll see your key on the dashboard.
4. Add to `.env` / secrets:
   ```
   NEWS_API_KEY=your_key_here
   ```

**Free tier limits:** 100 requests/day, headlines only (no historical >24h).
The app caches aggressively so 100/day is plenty for ~10 ticker queries.

---

## 3. Reddit / PRAW (social sentiment)

Used by [data/multi_sentiment.py](../data/multi_sentiment.py) `fetch_reddit()`.
Pulls posts from r/wallstreetbets, r/stocks, r/IndianStockMarket.

**Steps:**
1. Log in to Reddit → https://www.reddit.com/prefs/apps
2. Scroll to bottom → **"create another app..."**
3. Fill the form:
   - **name:** `protrader-ai-local`
   - **type:** select **`script`**
   - **redirect uri:** `http://localhost:8080` (required but unused)
4. Click **create app** → you'll see:
   - `client_id` — the string under "personal use script"
   - `client_secret` — the secret field
5. Add to `.env`:
   ```
   REDDIT_CLIENT_ID=your_client_id
   REDDIT_CLIENT_SECRET=your_client_secret
   REDDIT_USER_AGENT=protrader-ai/1.0 by your_reddit_username
   ```

**Free tier:** 60 requests/minute (more than enough).

---

## 4. Google Gemini (optional AI commentary)

Used in the explanation/summary panels. Pure UX — model still predicts without it.

**Steps:**
1. Go to https://aistudio.google.com/app/apikey
2. Sign in with your Google account.
3. Click **"Create API key"** → choose a Cloud project (or let it create one).
4. Copy the key, add to `.env`:
   ```
   GEMINI_API_KEY=your_key_here
   ```

**Free tier:** 60 requests/minute on `gemini-1.5-flash`. No card needed.

---

## 5. Roboflow (optional chart-pattern vision)

Only needed if you turn on the chart-pattern detection card.

**Steps:**
1. Sign up: https://app.roboflow.com/login
2. Settings → **Workspace** → copy **API key**.
3. Pick a public chart-pattern model (e.g. `chart-patterns/2`) and copy the
   workspace + project + version → that becomes your `ROBOFLOW_MODEL_ID`.
4. Add to `.env`:
   ```
   ROBOFLOW_API_KEY=your_key
   ROBOFLOW_MODEL_ID=workspace/project/version
   ```

**Free tier:** 1,000 inferences/month. Cache results per chart.

---

## 6. Free intraday data (no broker fees, no card)

The app is moving from end-of-day → 1-min bars (TASKS A3). The free ladder:

### 6.1 yfinance (zero setup, zero key)
- **Granularity:** 1-min for last 7 days, 5-min for last 60 days, daily forever.
- **Latency:** ~15 min delayed for NSE/BSE; live for US.
- **Use it as default** — no signup needed.

### 6.2 Yahoo unofficial WebSocket (~30 s latency, no key)
- Endpoint: `wss://streamer.finance.yahoo.com/`
- Used by `services/intraday_stream.py` (TASKS A3.2).
- No auth — but rate-limit yourself; Yahoo can soft-ban an IP.

### 6.3 Dhan API (true real-time, free, KYC required)
**Steps:**
1. Open a free Dhan demat account (PAN + Aadhaar e-KYC, no minimum balance):
   https://dhan.co/
2. Once logged in: **My Profile → DhanHQ Trading APIs**.
3. Generate **Access Token** (valid 24 h) and **Client ID**.
4. Add to `.env`:
   ```
   DHAN_CLIENT_ID=your_client_id
   DHAN_ACCESS_TOKEN=your_token
   ```
5. Token rotation: TASKS B5.4 will add a small daily refresh script.

**Cost:** ₹0 for market data. Order placement also free under their plan.

### 6.4 Upstox API (alternative to Dhan)
**Steps:**
1. Open free Upstox account (similar e-KYC).
2. https://account.upstox.com/developer/apps → **New App**.
3. Get `API Key`, `API Secret`, generate access token via OAuth flow.
4. Add to `.env`:
   ```
   UPSTOX_API_KEY=...
   UPSTOX_API_SECRET=...
   UPSTOX_ACCESS_TOKEN=...
   ```

Pick **one** of Dhan / Upstox — don't wire both, you'll just split your testing time.

### 6.5 NSE option chain (no key)
- Endpoint: `https://www.nseindia.com/api/option-chain-indices?symbol=NIFTY`
- Requires browser-like headers + a session cookie warmup. The implementation
  in TASKS A4.1 (`services/option_chain_service.py`) handles this.
- **No signup, no key**, but throttle to ≤ 1 request / 3 s to avoid 429s.

### 6.6 Macro data (no key)
All via yfinance tickers (TASKS A5.1):
| Macro feature           | yfinance ticker |
|-------------------------|-----------------|
| USD/INR                 | `INR=X`         |
| Crude (WTI)             | `CL=F`          |
| US 10Y yield            | `^TNX`          |
| Gold                    | `GC=F`          |
| S&P 500                 | `^GSPC`         |
| VIX                     | `^VIX`          |
| Nifty                   | `^NSEI`         |

---

## 7. HuggingFace Hub — host the v2 trained model

This is the big one. The Colab script
[sentiment_analysis_v2.py](../sentiment_analysis_v2.py) saves these artifacts
to `SAVE_DIR = /content/saved_models`:

```
saved_models/
├── sentiment_model/          ← FinBERT fine-tuned (HF format)
│   ├── config.json
│   ├── pytorch_model.bin     (or model.safetensors)
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── special_tokens_map.json
├── master_lr.pkl             ← logistic-regression base learner
├── master_rf.pkl             ← random-forest base learner
├── master_xgb.pkl            ← XGBoost base learner
├── master_lgb.pkl            ← LightGBM base learner
├── master_stacker.pkl        ← logistic-regression meta learner
└── config.json               ← feature columns + sentiment_model_name
```

We'll push the **whole folder** to a private HuggingFace repo and load it
in production via `huggingface_hub.snapshot_download()`.

### 7.1 Create the HuggingFace account + token

1. Sign up: https://huggingface.co/join (free, no card).
2. Verify email.
3. https://huggingface.co/settings/tokens → **New token**.
   - **Name:** `protrader-ai`
   - **Role:** **`write`** (you need write to push the model)
4. Copy the token (starts with `hf_...`). You'll only see it once.
5. Add to `.env`:
   ```
   HF_TOKEN=hf_your_token_here
   HF_REPO_ID=YOUR_HF_USERNAME/protrader-sentiment-v2
   ```

### 7.2 Create the model repo

1. https://huggingface.co/new
2. **Owner:** your username
3. **Model name:** `protrader-sentiment-v2`
4. **Visibility:** **Private** (recommended — your fine-tuned weights)
5. **License:** `apache-2.0` (matches FinBERT base) → **Create model**

### 7.3 Upload from Colab (run at the END of your training notebook)

Add this cell to the bottom of [sentiment_analysis_v2.py](../sentiment_analysis_v2.py)
(or run it separately) **after** the existing save block:

```python
# ── Push trained artifacts to HuggingFace Hub ──────────────────────
from huggingface_hub import HfApi, login

HF_TOKEN   = "hf_..."                        # paste in Colab — don't commit
HF_REPO_ID = "your_username/protrader-sentiment-v2"

login(token=HF_TOKEN)
api = HfApi()

api.upload_folder(
    folder_path=SAVE_DIR,              # /content/saved_models
    repo_id=HF_REPO_ID,
    repo_type="model",
    commit_message="v2 ensemble: finbert + 4 base learners + stacker",
    ignore_patterns=["*.tmp", "checkpoint-*"],
)
print(f"✅ Pushed to https://huggingface.co/{HF_REPO_ID}")
```

That's a single API call — uploads the whole 500 MB-ish folder in one go.

### 7.4 Upload from your laptop instead (if you have the folder locally)

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli login                    # paste your hf_... token
huggingface-cli upload your_username/protrader-sentiment-v2 \
    ./saved_models . \
    --repo-type=model \
    --commit-message="v2 ensemble bundle"
```

### 7.5 Load the model in production (the integration code)

This is what `services/v2_ensemble_service.py` (TASKS A2.3) will look like.
**Don't** add this yet if you haven't built A2; just keep the file uploaded
and ready.

```python
# services/v2_ensemble_service.py
from functools import lru_cache
from pathlib import Path
import joblib
from huggingface_hub import snapshot_download
from transformers import pipeline
from config.settings import HF_TOKEN, HF_REPO_ID

@lru_cache(maxsize=1)
def _v2_bundle():
    """Download once, cache forever (per process)."""
    local_dir = snapshot_download(
        repo_id=HF_REPO_ID,
        token=HF_TOKEN,
        cache_dir=".hf_cache",        # persistent on Streamlit Cloud disk
    )
    p = Path(local_dir)
    return {
        "sentiment": pipeline(
            "sentiment-analysis",
            model=str(p / "sentiment_model"),
            framework="pt",
        ),
        "lr":      joblib.load(p / "master_lr.pkl"),
        "rf":      joblib.load(p / "master_rf.pkl"),
        "xgb":     joblib.load(p / "master_xgb.pkl"),
        "lgb":     joblib.load(p / "master_lgb.pkl"),
        "stacker": joblib.load(p / "master_stacker.pkl"),
    }

def v2_bullish_probability(headline: str) -> float:
    """Return bullish prob ∈ [0,1] — used as ONE FEATURE in the meta-stacker."""
    b = _v2_bundle()
    # ... (feature pipeline matching what was trained in Colab)
    return float(b["stacker"].predict_proba(features)[0, 1])
```

The output of `v2_bullish_probability()` becomes one extra column appended
to the existing meta-stacker feature matrix — exactly the integration
strategy from TASKS A2.

### 7.6 Streamlit Cloud secrets for HF

Paste into Streamlit Cloud → Settings → Secrets:
```toml
HF_TOKEN   = "hf_your_token_here"
HF_REPO_ID = "your_username/protrader-sentiment-v2"
```

### 7.7 Updating the model later

Re-train in Colab → re-run cell 7.3 → it just creates a new commit on the
HF repo. Production picks it up next time the cache expires (or on restart).
Pin to a specific revision if you want determinism:

```python
snapshot_download(repo_id=HF_REPO_ID, revision="v1.2.0", token=HF_TOKEN)
```

Tag releases on the HF web UI: model page → **"Files and versions"** →
**"Add tag"**.

### 7.8 Sanity check after upload

```bash
python -c "
from huggingface_hub import HfApi
api = HfApi()
files = api.list_repo_files('your_username/protrader-sentiment-v2', token='hf_...')
for f in files: print(f)
"
```

Expected output should list `sentiment_model/config.json`, `master_*.pkl`, etc.

---

## 8. Quick-start checklist

Local-dev minimum-viable setup (~10 minutes):

- [ ] `pip install -r requirements.txt`
- [ ] Create `.env` from `.env.example`
- [ ] (optional) Add `NEWS_API_KEY` from §2
- [ ] (optional) Add `GEMINI_API_KEY` from §4
- [ ] `streamlit run app.py` → app loads, predictions work

Full-feature setup (~1 hour incl. broker KYC wait):

- [ ] All §2 / §3 / §4 keys
- [ ] HuggingFace token + uploaded v2 model (§7)
- [ ] Dhan **or** Upstox account + token (§6.3 / §6.4)
- [ ] Roboflow key if using chart vision (§5)

---

## 9. Cost summary

| Service           | Monthly cost at our usage |
|-------------------|---------------------------|
| yfinance          | ₹0 (no auth)              |
| RSS feeds         | ₹0                        |
| NewsAPI free      | ₹0                        |
| Reddit API        | ₹0                        |
| Gemini Flash      | ₹0                        |
| Roboflow free     | ₹0 (1k inf cap)           |
| Dhan / Upstox     | ₹0 for market data        |
| HuggingFace Hub   | ₹0 (private repos free)   |
| Streamlit Cloud   | ₹0 (community tier)       |
| **TOTAL**         | **₹0**                    |

The only things you ever pay for in this project are (a) a domain name when
you migrate to Next.js (TASKS B4) and (b) optionally GPU hours if you decide
to fine-tune the FinBERT base. Everything in the live app stays free-tier.

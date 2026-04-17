# ProTrader AI — Deployment Guide

## 1. Python version

**Required: Python 3.11** (3.10 also works; avoid 3.12 / 3.13 because
`tensorflow-cpu` < 2.16 does not have wheels for them yet).

Streamlit Community Cloud lets you pick the runtime when you create the
app — under **Advanced settings → Python version** select **3.11**.
The repo also ships `runtime.txt` (`python-3.11`) as a fallback signal
for any host that honours it.

## 2. Files added for cloud deploy

| File                              | Purpose                                              |
|-----------------------------------|------------------------------------------------------|
| `runtime.txt`                     | Tells the host to use Python 3.11                    |
| `packages.txt`                    | apt packages installed before pip (libgomp for LGBM) |
| `.streamlit/config.toml`          | Theme + server config                                |
| `.streamlit/secrets.toml.example` | Template for secret keys (copy to `secrets.toml`)    |
| Updated `requirements.txt`        | Pinned versions tested for Python 3.11               |
| Updated `config/settings.py`      | Reads `st.secrets` first, then `.env`                |

## 3. Deploy to Streamlit Community Cloud

1. Push the repo to GitHub (private or public).
2. Go to <https://share.streamlit.io> → **New app**.
3. Repository: your fork. Branch: `main`. Main file path: `app.py`.
4. **Advanced settings → Python version: 3.11**.
5. Click **Deploy**. First build takes ~6 min (downloads tensorflow-cpu,
   torch, transformers, finbert weights).
6. Once green, open **Settings → Secrets** and paste:

   ```toml
   GEMINI_API_KEY = "..."
   NEWS_API_KEY = "..."
   REDDIT_CLIENT_ID = "..."
   REDDIT_CLIENT_SECRET = "..."
   REDDIT_USER_AGENT = "ProTraderAI/1.0"
   ROBOFLOW_API_KEY = "..."
   ```

   The app reloads automatically.

## 4. Run locally

```bash
# 1. Create a venv on Python 3.11
py -3.11 -m venv venv
source venv/Scripts/activate    # Windows: venv\Scripts\activate

# 2. Install
pip install -r requirements.txt

# 3. Set keys (one of):
#    a) put them in .env (already supported)
#    b) copy .streamlit/secrets.toml.example -> .streamlit/secrets.toml

# 4. Run
streamlit run app.py
```

## 5. Memory & cold-start notes

- Streamlit Cloud free tier is **1 GB RAM**. The app at idle uses ~600 MB
  (tensorflow + finbert).
- First request loads FinBERT (~440 MB download, cached after);
  subsequent requests are sub-second for sentiment.
- Reddit and Google Trends are optional — leaving their keys blank
  silently disables those sources and the multi-source weights
  re-normalise automatically.

## 6. Verification checklist after deploy

- [ ] App loads landing page without error.
- [ ] Sidebar dropdown lists Indian stocks.
- [ ] Click **Launch Analysis** on RELIANCE → completes in ≤ 90 s.
- [ ] **Multi-Source Sentiment** tab shows ≥ 1 source available.
- [ ] **Dynamic Fusion** tab shows model weights.
- [ ] **Backtest** tab runs without TensorFlow import errors.
- [ ] Checking the Streamlit Cloud logs shows no "module not found".

If any check fails, the most common cause is the Python version — make
sure it is 3.11 in **Advanced settings**, not 3.12.

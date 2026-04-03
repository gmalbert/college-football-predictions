# Deployment & Operations Roadmap

This document covers deploying the app to Streamlit Community Cloud, CI/CD,
monitoring, and ongoing operations.

---

## 1. Streamlit Cloud Deployment

### Prerequisites

1. **GitHub repo** — public or private, connected to Streamlit Cloud.
2. **`requirements.txt`** in the repo root.
3. **App entrypoint** — `predictions.py`.
4. **Secrets** configured in Streamlit Cloud dashboard.

### Step-by-Step

1. Push code to `main` branch.
2. Go to <https://share.streamlit.io/>.
3. Click **New app** → select repo, branch (`main`), and file (`predictions.py`).
4. Under **Advanced settings → Secrets**, paste your `.streamlit/secrets.toml`
   contents.
5. Click **Deploy**.

### `.streamlit/secrets.toml` (local only — gitignored)

```toml
[cfbd]
api_key = "your-cfbd-bearer-token"

[odds]
api_key = "your-odds-api-key"
```

### `.gitignore` additions

```
.streamlit/secrets.toml
data_files/raw/
data_files/processed/
data_files/features/
data_files/models/
__pycache__/
*.pyc
.env
```

---

## 2. Secrets Management in Production

```python
# utils/config.py
import streamlit as st
import os

def get_secret(section: str, key: str) -> str:
    """
    Fetch a secret from Streamlit secrets (Cloud) or environment variables
    (local fallback).
    """
    try:
        return st.secrets[section][key]
    except (KeyError, FileNotFoundError):
        env_key = f"{section.upper()}_{key.upper()}"
        value = os.environ.get(env_key)
        if value is None:
            raise ValueError(
                f"Secret '{section}.{key}' not found in Streamlit secrets "
                f"or environment variable '{env_key}'."
            )
        return value
```

---

## 3. Data Refresh Strategy

Since Streamlit Cloud doesn't support cron jobs natively, use one of these
approaches:

### Option A: GitHub Actions Scheduler (Recommended)

```yaml
# .github/workflows/refresh_data.yml
name: Refresh Data

on:
  schedule:
    - cron: "0 8 * * *"       # daily at 8 AM UTC
    - cron: "0 */2 * * 6"     # every 2 hours on Saturdays (game day)
  workflow_dispatch:            # manual trigger

jobs:
  refresh:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Fetch latest data
        env:
          CFBD_API_KEY: ${{ secrets.CFBD_API_KEY }}
        run: python scripts/refresh_data.py

      - name: Commit updated data
        run: |
          git config user.name "github-actions[bot]"
          git config user.email "github-actions[bot]@users.noreply.github.com"
          git add data_files/processed/ data_files/features/
          git diff --cached --quiet || git commit -m "chore: refresh data $(date -u +%Y-%m-%d)"
          git push
```

### Option B: Streamlit `@st.cache_data` with TTL

For lightweight refreshes, cache API calls with a time-to-live:

```python
import streamlit as st
import datetime

@st.cache_data(ttl=datetime.timedelta(hours=6))
def fetch_weekly_games(year: int, week: int):
    """Cached API call — re-fetches every 6 hours."""
    # ... call CFBD API ...
    pass
```

---

## 4. Performance Optimization

| Technique | Where | Impact |
|-----------|-------|--------|
| `@st.cache_data` | API calls, data transforms | Eliminate redundant API calls |
| `@st.cache_resource` | Model loading | Load model once, share across sessions |
| Parquet files | Data storage | 10× faster reads than CSV |
| `st.fragment` | Live score widget | Partial page refresh without full rerun |
| Lazy imports | Heavy libraries (xgboost) | Faster cold starts |

```python
@st.cache_resource
def load_spread_model():
    """Load the trained XGBoost model once."""
    import joblib
    return joblib.load("data_files/models/spread_xgb.joblib")
```

---

## 5. Monitoring & Logging

### Application Logging

```python
# utils/logger.py
import logging
import sys

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

### Model Accuracy Tracking

After each week, log model performance to a CSV so you can track drift:

```python
# scripts/log_performance.py
import csv
from pathlib import Path
from datetime import date

PERF_LOG = Path("data_files/performance_log.csv")

def log_weekly_performance(
    season: int, week: int, brier: float, ats_pct: float, roi: float
) -> None:
    write_header = not PERF_LOG.exists()
    with open(PERF_LOG, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["date", "season", "week", "brier", "ats_pct", "roi"])
        writer.writerow([date.today().isoformat(), season, week, brier, ats_pct, roi])
```

---

## 6. Error Handling in Production

```python
# utils/error_handling.py
import streamlit as st
import traceback
from utils.logger import get_logger

logger = get_logger(__name__)

def safe_api_call(func, *args, **kwargs):
    """Wrap API calls with user-friendly error handling."""
    try:
        return func(*args, **kwargs)
    except ConnectionError:
        st.error("🔌 Could not connect to the data source. Please try again later.")
        logger.error(f"Connection error in {func.__name__}")
        return None
    except Exception as e:
        st.error(f"⚠️ Something went wrong: {e}")
        logger.error(f"Error in {func.__name__}: {traceback.format_exc()}")
        return None
```

---

## 7. Resource Limits (Streamlit Cloud Free Tier)

| Resource | Limit |
|----------|-------|
| RAM | 1 GB |
| Storage | Ephemeral (wiped on reboot) |
| Concurrent viewers | ~20-50 |
| App sleep | After 7 days of inactivity |
| CPU | Shared |

**Implications:**
- Pre-compute features and predictions offline (GitHub Actions) rather than
  on-the-fly.
- Store processed data in the repo (Parquet, < 100 MB) so it's available
  on every deploy.
- Use `@st.cache_data` aggressively to reduce RAM churn.
- Keep model files small (< 50 MB).

---

## 8. Security Checklist

- [ ] `.streamlit/secrets.toml` is in `.gitignore`
- [ ] No API keys in source code
- [ ] CFBD key uses least-privilege scope
- [ ] No user-input-driven SQL or shell commands
- [ ] Dependencies pinned in `requirements.txt`
- [ ] Dependabot or Renovate enabled for dependency updates
- [ ] `.env` file in `.gitignore`

---

## 9. Domain & Custom URL (Optional)

Streamlit Cloud apps get a URL like
`https://your-app.streamlit.app`. For a custom domain:

1. Purchase a domain (Namecheap, Cloudflare, etc.).
2. In Streamlit Cloud → App Settings → Custom domain.
3. Add a CNAME record pointing to `cname.streamlit.app`.
4. Wait for DNS propagation + SSL provisioning.

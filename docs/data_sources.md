# Data Sources Roadmap

This document catalogues every data source the platform will consume, the
specific endpoints / tables of interest, refresh cadence, and implementation
notes.

---

## 1. College Football Data API (CFBD)

**Base URL:** `https://api.collegefootballdata.com/`  
**Auth:** Bearer token (free tier available, register at <https://collegefootballdata.com/key>)  
**Python wrapper:** `cfbd` PyPI package

### Key Endpoints

| Endpoint | Purpose | Refresh |
|----------|---------|---------|
| `/games` | Scores, results, home/away, venue, weather | Daily during season |
| `/games/teams` | Team-level box-score stats per game | Daily during season |
| `/games/players` | Player-level box-score stats per game | Daily during season |
| `/games/media` | TV network & streaming info | Weekly |
| `/lines` | Sportsbook betting lines (spread, O/U, moneyline) | Every 6 hours game weeks |
| `/stats/season` | Season-level team stats (off / def / special teams) | Daily |
| `/stats/season/advanced` | EPA, success rate, havoc, PPA, etc. | Daily |
| `/rankings` | AP, Coaches, CFP, FPI polls | Weekly |
| `/recruiting/players` | Recruit ratings, stars, positions | Yearly |
| `/recruiting/teams` | Team-level recruiting aggregates | Yearly |
| `/talent` | Blue-chip ratio / composite talent | Yearly |
| `/records` | Historical W-L records | Yearly |
| `/conferences` | Conference membership over time | Yearly |
| `/teams` | Logos, colors, venue, location | Yearly |
| `/plays` | Play-by-play data | Post-game |
| `/metrics/wp` | Win probability by play | Post-game |
| `/ppa/games` | Predicted Points Added per game | Post-game |
| `/ratings/sp` | S&P+ (Bill Connelly) ratings | Weekly |
| `/ratings/elo` | Elo ratings | Weekly |
| `/ratings/fpi` | FPI ratings | Weekly |

### Example: Fetching Game Data

```python
import cfbd
from cfbd.rest import ApiException

configuration = cfbd.Configuration()
configuration.api_key["Authorization"] = "YOUR_CFBD_API_KEY"
configuration.api_key_prefix["Authorization"] = "Bearer"

api = cfbd.GamesApi(cfbd.ApiClient(configuration))

try:
    games = api.get_games(year=2025, season_type="regular")
    for g in games:
        print(f"{g.away_team} @ {g.home_team}: {g.away_points}-{g.home_points}")
except ApiException as e:
    print(f"CFBD API error: {e}")
```

### Example: Fetching Betting Lines

```python
api = cfbd.BettingApi(cfbd.ApiClient(configuration))
lines = api.get_lines(year=2025, week=1)
for game in lines:
    for line in game.lines:
        print(
            f"{game.away_team} @ {game.home_team} | "
            f"Provider: {line.provider} | Spread: {line.spread} | O/U: {line.over_under}"
        )
```

### Example: Fetching Advanced Stats

```python
api = cfbd.StatsApi(cfbd.ApiClient(configuration))
advanced = api.get_advanced_team_season_stats(year=2025)
for team in advanced:
    off = team.offense
    defe = team.defense
    print(
        f"{team.team}: Off EPA/play={off.total_ppa:.3f}, "
        f"Def EPA/play={defe.total_ppa:.3f}, "
        f"Off Success Rate={off.success_rate:.3f}"
    )
```

---

## 2. ESPN API (Unofficial / Public Endpoints)

**Base URL:** `https://site.api.espn.com/apis/site/v2/sports/football/college-football/`  
**Auth:** None (public, no key required)  
**Rate limit:** Unofficial — keep requests ≤ 1/sec to be respectful

### Key Endpoints

| Endpoint | Purpose | Refresh |
|----------|---------|---------|
| `/scoreboard` | Live & recent scores, game status, odds | Real-time on game days |
| `/teams` | Full team directory | Yearly |
| `/teams/{id}` | Team details, record, roster link | Weekly |
| `/teams/{id}/roster` | Current roster with jersey, position, class | Weekly |
| `/teams/{id}/schedule` | Team-specific schedule | Weekly |
| `/summary?event={id}` | Deep game summary (drives, plays, leaders) | Post-game |
| `/news` | Headlines and articles | Hourly |
| `/rankings` | Current poll rankings | Weekly |

### Example: Live Scoreboard

```python
import requests

SCOREBOARD_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/football/"
    "college-football/scoreboard"
)

def get_live_scores(limit: int = 25, groups: int = 80) -> list[dict]:
    """Fetch current/recent college football scores from ESPN."""
    params = {"limit": limit, "groups": groups}
    resp = requests.get(SCOREBOARD_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    games = []
    for event in data.get("events", []):
        competition = event["competitions"][0]
        home = competition["competitors"][0]
        away = competition["competitors"][1]
        games.append({
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "home_score": home.get("score", "0"),
            "away_score": away.get("score", "0"),
            "status": event["status"]["type"]["shortDetail"],
            "odds": competition.get("odds", []),
        })
    return games
```

### Example: Fetch Team Roster

```python
def get_team_roster(team_id: int) -> list[dict]:
    """Return the current roster for a given ESPN team ID."""
    url = (
        f"https://site.api.espn.com/apis/site/v2/sports/football/"
        f"college-football/teams/{team_id}/roster"
    )
    resp = requests.get(url, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    roster = []
    for group in data.get("athletes", []):
        for player in group.get("items", []):
            roster.append({
                "name": player["displayName"],
                "position": player.get("position", {}).get("abbreviation"),
                "jersey": player.get("jersey"),
                "year": player.get("experience", {}).get("displayValue"),
            })
    return roster
```

---

## 3. Additional / Future Sources

| Source | Data | Status |
|--------|------|--------|
| **Odds API** (`the-odds-api.com`) | Real-time lines from 20+ sportsbooks | Future |
| **Weather API** (Open-Meteo / NWS) | Game-day weather for outdoor venues | Future |
| **Pro Football Focus (PFF)** | Grades & advanced player metrics | Paid — evaluate ROI |
| **247Sports / Rivals** | Recruiting rankings & portal data | Scrape or manual |
| **Vegas Insider / DonBest** | Historical line movement | Future |
| **Sagarin / Massey Ratings** | Third-party power ratings | Future |

---

## 4. Data Storage Strategy

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Raw | `data_files/raw/` JSON/CSV | Immutable landing zone for API responses |
| Processed | `data_files/processed/` Parquet | Cleaned, typed, deduplicated tables |
| Features | `data_files/features/` Parquet | Model-ready feature matrices |
| Cache | Streamlit `@st.cache_data` | In-memory hot cache for dashboard queries |

For a lightweight MVP, flat Parquet files are sufficient. As the project
scales, consider migrating to **DuckDB** (embedded, fast analytics) or a
hosted **PostgreSQL** database.

---

## 5. API Key Management

```
# .streamlit/secrets.toml  (gitignored)
[cfbd]
api_key = "your-cfbd-api-key"

[odds]
api_key = "your-odds-api-key"  # future
```

Access in Streamlit:
```python
import streamlit as st
CFBD_KEY = st.secrets["cfbd"]["api_key"]
```

> **Never commit API keys to the repo.** Use Streamlit Cloud's Secrets
> management (<https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management>)
> for production.

# College Football Predictions (Tailgate Edge) — 12-Month Feature Roadmap

> Generated: 2026-07-31 | Horizon: August 2026 – July 2027

---

## Q1 (Aug–Oct 2026) — Data Foundation

### Feature 1 — PFF College Stats Integration

Ingest PFF college grades for offensive/defensive lines, quarterbacks,
and coverage. These correlate strongly with game outcomes.

```python
# utils/cfbd_client.py (add PFF grades)
import requests, pandas as pd, os

def fetch_pff_team_grades(season: int) -> pd.DataFrame:
    """Fetch PFF college team grades per week."""
    resp = requests.get(
        "https://api.collegefootballdata.com/ppa/games",
        params={"year": season, "unit": "offense"},
        headers={"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"},
        timeout=15,
    )
    return pd.DataFrame(resp.json())
```

### Feature 2 — SP+ / ESPN FPI Integration

Incorporate SP+ ratings (Bill Connelly) and ESPN FPI as model features.
These provide forward-looking team strength estimates.

```python
# utils/sp_plus.py
import requests, pandas as pd

def fetch_sp_ratings(season: int) -> pd.DataFrame:
    resp = requests.get(
        "https://api.collegefootballdata.com/ratings/sp",
        params={"year": season},
        headers={"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"},
        timeout=15,
    )
    data = resp.json()
    return pd.DataFrame([
        {"team": r["team"], "sp_offense": r.get("offense", {}).get("rating", 0),
         "sp_defense": r.get("defense", {}).get("rating", 0),
         "sp_overall": r.get("rating", 0), "season": season}
        for r in data
    ])
```

### Feature 3 — Recruiting Class Rank Feature

Recruiting ranking (247Sports Composite) predicts team trajectory.
Teams with top-10 recruiting classes over 3 years are typically top-25 teams.

```python
# utils/recruiting.py
import requests, pandas as pd

def fetch_recruiting_rankings(years: list[int]) -> pd.DataFrame:
    dfs = []
    for year in years:
        resp = requests.get(
            "https://api.collegefootballdata.com/recruiting/teams",
            params={"year": year},
            headers={"Authorization": f"Bearer {os.environ['CFBD_API_KEY']}"},
            timeout=15,
        )
        df = pd.DataFrame(resp.json())
        df["year"] = year
        dfs.append(df)
    combined = pd.concat(dfs)
    return combined.groupby("team")["points"].mean().reset_index().rename(
        columns={"points": "avg_recruiting_score_3yr"}
    )
```

### Feature 4 — Conference Strength Adjusted Elo

Separate Elo systems per conference + inter-conference adjustment.
A 10-win MAC team has much lower Elo than a 10-win SEC team.

```python
# utils/elo.py (extend)
CONFERENCE_BASELINES = {
    "SEC": 1600, "Big Ten": 1590, "Big 12": 1570, "ACC": 1550,
    "Pac-12": 1540, "AAC": 1480, "MAC": 1460, "MWC": 1470,
    "Sun Belt": 1450, "CUSA": 1440, "Ind": 1500,
}

def initialize_team_elo(team: str, conference: str) -> float:
    return CONFERENCE_BASELINES.get(conference, 1500)
```

### Feature 5 — Home Field Advantage by Stadium Size

Larger stadiums create more home advantage. The Big House (107K) has
demonstrably higher home advantage than a 30K FBS stadium.

```python
# utils/home_field.py
STADIUM_CAPACITIES = {
    "Michigan Wolverines": 107_601, "Penn State Nittany Lions": 106_572,
    "Ohio State Buckeyes": 102_780, "Alabama Crimson Tide": 100_077,
    "LSU Tigers": 102_321,
}

def get_home_advantage(team: str) -> float:
    """Estimate home advantage (points) based on stadium capacity."""
    capacity = STADIUM_CAPACITIES.get(team, 50_000)
    base = 3.0
    size_bonus = max(0, (capacity - 50_000) / 50_000 * 2.0)
    return base + size_bonus
```

---

## Q2 (Nov 2026 – Jan 2027) — Model Enhancement

### Feature 6 — Quarterback Injury Impact

QBs in NCAAF are more critical than in NFL due to smaller roster depth.
Track backup QB performance and adjust when starter is questionable.

```python
# utils/qb_impact.py
import pandas as pd

QB_PERFORMANCE_TIERS = {
    "starter": 1.0, "backup_experienced": 0.80,
    "backup_inexperienced": 0.65, "walk_on": 0.50,
}

def get_qb_impact_multiplier(team: str, injury_report: pd.DataFrame) -> float:
    starter_report = injury_report[
        (injury_report["team"] == team) & (injury_report["position"] == "QB")
    ]
    if starter_report.empty or starter_report.iloc[0]["status"] == "Active":
        return QB_PERFORMANCE_TIERS["starter"]
    return QB_PERFORMANCE_TIERS.get("backup_experienced", 0.80)
```

### Feature 7 — Temperature/Weather Impact on Scoring

Cold-weather games at outdoor stadiums suppress scoring significantly.
Northern programs have home advantage in cold-weather games in November.

```python
# utils/weather_scoring.py
def cold_weather_team_advantage(
    home_team: str, away_team: str, temp_f: float
) -> float:
    """Return adj to home team win prob for cold-weather games."""
    COLD_WEATHER_PROGRAMS = {
        "Wisconsin", "Michigan", "Minnesota", "Iowa", "Penn State",
        "Ohio State", "Northwestern", "Illinois",
    }
    if temp_f > 45:
        return 0.0
    home_is_cold = any(kw in home_team for kw in COLD_WEATHER_PROGRAMS)
    away_is_warm = not any(kw in away_team for kw in COLD_WEATHER_PROGRAMS)
    if home_is_cold and away_is_warm:
        cold_factor = max(0, (45 - temp_f) / 45 * 0.06)
        return cold_factor
    return 0.0
```

### Feature 8 — Turnover Regression Model

Teams with extreme turnover margins (+ or -) tend to regress significantly.
Flag teams due for turnover regression as betting value targets.

```python
# analytics/turnover_regression.py
import pandas as pd

def flag_turnover_outliers(season_stats: pd.DataFrame) -> pd.DataFrame:
    league_avg = season_stats["turnover_margin"].mean()
    season_stats["turnover_deviation"] = season_stats["turnover_margin"] - league_avg
    season_stats["regression_flag"] = season_stats["turnover_deviation"].apply(
        lambda d: "Due for negative regression" if d > 2.5
        else ("Due for positive regression" if d < -2.5 else "Normal")
    )
    return season_stats
```

### Feature 9 — CFP Playoff Probability Simulator

Daily Monte Carlo simulation of remaining games + CFP selection.
Output each Power 4 team's probability of making the 12-team CFP.

```python
# analytics/cfp_simulator.py
import numpy as np, pandas as pd
from collections import defaultdict

def simulate_cfp_field(
    remaining_games: pd.DataFrame, current_rankings: pd.DataFrame,
    team_elos: dict, n_sim: int = 5_000
) -> pd.DataFrame:
    cfp_counts = defaultdict(int)
    for _ in range(n_sim):
        final_records = current_rankings.set_index("team")["wins"].to_dict()
        for _, game in remaining_games.iterrows():
            h, a = game["home_team"], game["away_team"]
            home_wp = 1 / (1 + 10 ** ((team_elos.get(a, 1500) - team_elos.get(h, 1500)) / 400))
            if np.random.random() < home_wp:
                final_records[h] = final_records.get(h, 0) + 1
            else:
                final_records[a] = final_records.get(a, 0) + 1

        # Simple top-12 by wins selection (actual CFP uses committee)
        top12 = sorted(final_records, key=final_records.get, reverse=True)[:12]
        for t in top12:
            cfp_counts[t] += 1

    return pd.DataFrame([
        {"team": t, "cfp_prob": cfp_counts[t] / n_sim}
        for t in current_rankings["team"]
    ]).sort_values("cfp_prob", ascending=False)
```

### Feature 10 — Transfer Portal Impact Tracker

Track transfer portal entries and additions. Quantify projected roster
strength change per team after each transfer window.

---

## Q3 (Feb–Apr 2027) — Dashboard

### Feature 11 — Power Rankings with Explanations

Weekly power rankings with SHAP-style feature importance showing what
drives each team's current ranking.

### Feature 12 — Bowl Game Prediction Suite

Special bowl game model. Extended rest periods, player opt-outs, and
coaches interviewing elsewhere create unique prediction dynamics.

```python
# analytics/bowl_game.py
import pandas as pd

def bowl_game_adjustments(team: str, bowl_info: dict) -> dict:
    """Apply bowl-specific adjustments to model inputs."""
    opt_outs = bowl_info.get("opt_out_players", [])
    coach_lame_duck = bowl_info.get("coach_is_outgoing", False)
    days_since_last_game = bowl_info.get("days_rest", 35)

    motivation_factor = 1.0
    if coach_lame_duck:
        motivation_factor *= 0.95  # slight demotivation
    if len(opt_outs) >= 3:
        motivation_factor *= 0.88  # significant opt-outs
    # Rust factor for very long layoffs
    rust_factor = max(0.95, 1.0 - max(0, days_since_last_game - 21) * 0.005)
    return {"motivation": motivation_factor, "rust": rust_factor}
```

### Feature 13 — Rivalry Game Special Analysis

Historic rivalry games (Michigan-Ohio State, Alabama-Auburn, Army-Navy)
have unique dynamics. Build rivalry-specific adjustment factors.

### Feature 14 — Coaching Tenure Effect

New coaches typically underperform in year 1, overperform in years 3-5
as their recruiting materializes. Encode coaching tenure as a feature.

### Feature 15 — Stadium Noise Impact

Night games in loud stadiums (Death Valley, The Swamp) create measurable
home advantage beyond standard. Quantify acoustic home advantage.

---

## Q4 (May–Jul 2027) — Automation & Intelligence

### Feature 16 — Automated Bowl Season Pipeline

GitHub Action runs bowl game predictions for all 40+ bowl games starting
in late December. Export best_bets.json with picks ranked by edge.

### Feature 17 — Early Season Uncertainty Model

First 2-3 weeks of the season have high uncertainty (new rosters, new
schemes). Add uncertainty bands to early-season predictions.

### Feature 18 — Spring Game Analytics

Ingest spring game data as early signals of depth chart changes and
scheme evolution for the upcoming season.

### Feature 19 — Conference Championship Game Predictor

Dedicated model for conference championship games. Divisional tiebreakers,
neutral site adjustments, and 2-game familiarity effects.

### Feature 20 — Injury Bowl Watch (December Opt-Outs)

Monitor December opt-out announcements by key players ahead of bowl games.
Auto-adjust predictions when major skill positions opt out.

### Feature 21 — AI Pre-Game Analysis

GPT-4o-mini generates a 200-word pre-game analysis for any matchup:
key storylines, model edge, injury report, and weather factors.

```python
# utils/ai_preview.py
import os, json
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def generate_ncaaf_preview(game_data: dict) -> str:
    prompt = f"""
    Write a 150-word college football game preview for:
    {game_data['home_team']} vs {game_data['away_team']}

    Home team SP+: {game_data.get('home_sp', 'N/A')}
    Away team SP+: {game_data.get('away_sp', 'N/A')}
    Model: {game_data['home_team']} {game_data['model_spread']:+.1f} (Line: {game_data.get('line', 'N/A')})
    Weather: {game_data.get('weather', 'Clear')}
    Key injury: {game_data.get('key_injury', 'None')}

    Include: key matchup factors, model value bet if applicable, betting recommendation.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=250, temperature=0.5,
    )
    return resp.choices[0].message.content
```

### Feature 22 — Heisman Tracker

Track Heisman Trophy frontrunners. Model probability of winning Heisman
based on team's media market, conference, and statistical production.

---

## Timeline Summary

| Quarter | Focus | Key Deliverables |
|---------|-------|-----------------|
| Q1 Aug–Oct 2026 | Data foundation | PFF grades, SP+/FPI, recruiting rankings, conference Elo, home field |
| Q2 Nov 2026–Jan 2027 | Model enhancement | QB injury impact, weather scoring, turnover regression, CFP simulator, transfer portal |
| Q3 Feb–Apr 2027 | Dashboard | Power rankings, bowl game model, rivalry analysis, coaching tenure, stadium noise |
| Q4 May–Jul 2027 | Automation | Bowl pipeline, uncertainty model, spring game data, conference championship, opt-out tracker, AI preview, Heisman tracker |

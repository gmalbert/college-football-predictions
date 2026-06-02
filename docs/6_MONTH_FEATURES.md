# Tailgate Edge — 6-Month Feature Roadmap

## Month 1: Game Week Experience

- **Saturday schedule widget** — Today's games sorted by kickoff time with prediction cards (model win%, spread pick, edge tier).
- **Top 25 rankings integration** — Pull AP/Coaches poll rankings from CFBD API; display next to team names.
- **Conference rivalry badge** — Highlight rivalry games (Iron Bowl, Red River, etc.) with custom badge.
- **Live score refresh** — Refresh game scores every 60 seconds on game days using ESPN API.

## Month 2: Team Analytics

- **Team profile page** — Season stats, Elo trend chart, schedule with prediction results.
- **Recruiting integration** — 247Sports recruiting class rank from CFBD API; show as sidebar stat.
- **Strength of schedule bar chart** — Visual SOS ranking for each conference.
- **Injury report widget** — Surface any announced starters listed as questionable via ESPN API.

## Month 3: Betting Tools

- **Value bets page** — Filter by edge > 3%, sorted by confidence. Show spread and moneyline.
- **Parlay calculator** — Select 2–4 games; compute combined probability and implied parlay odds.
- **Line movement tracker** — Show opening vs. current spread for games this week.

## Month 4: Historical Analysis

- **Season results log** — All model predictions vs. actual results in a filterable table.
- **Conference accuracy breakdown** — Model accuracy by conference (SEC, Big Ten, etc.).
- **Upset tracker** — Highlight games where a ≥14-point underdog won.

## Month 5: Advanced Features

- **Playoff simulator** — Monte Carlo College Football Playoff bracket simulator (12-team format).
- **Bowl game predictor** — Special model for bowl game matchups (cross-conference, long prep time).
- **Weather impact map** — For outdoor games, show forecast and historically lower-scoring conditions.

## Month 6: Automation

- **Friday email digest** — Top picks for the upcoming Saturday with game times and spread.
- **GitHub Actions automation** — Nightly data refresh from CFBD and Odds API during season.
- **Discord game-day alerts** — Post top pick before Saturday noon kickoffs.

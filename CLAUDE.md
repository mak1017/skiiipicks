# skiii picks

Sports betting model and phone dashboard for NFL, college football, NBA and MLB. Python builds ratings,
projections, props, injuries, odds and a P&L log, then injects everything into one self-contained HTML page.
GitHub Actions rebuilds it every hour and publishes it to GitHub Pages.

## Layout

- `skiiipicks/build.py` - entry point. Runs every sport, writes `dashboard_data.json`. `--data-dir` uses local files.
- `skiiipicks/core.py` - ridge-regression team ratings (offense/defense), walk-forward backtests, ATS scoring.
- `skiiipicks/nfl.py`, `cfb.py`, `nba.py`, `mlb.py` - per-sport data loading, profiles, upcoming games.
- `skiiipicks/stack.py` - stage-2 game-context model (rest, QB changes, Elo, talent, travel), pick tiers.
- `skiiipicks/props.py` - player props (NFL, CFB, NBA, MLB batters) with fitted spreads.
- `skiiipicks/injuries.py` - ESPN live injury page with the sportsdataverse daily feed as fallback.
- `skiiipicks/odds.py` - The Odds API (key in env var `ODDS_API_KEY`), cached in `odds_cache.json`.
- `skiiipicks/pnl.py` - ratings (green/yellow/red), picks log, grading, CLV, daily top picks.
- `skiiipicks/dashboard_template.html` - the whole front end (HTML + CSS + JS in one file).
  `dashboard.py` replaces `/*DATA*/null` in it with the JSON.
- `picks_log.csv`, `odds_cache.json` - state the GitHub workflow commits back after each run. Don't delete.
- `.github/workflows/rebuild.yml` - hourly build + Pages deploy.

## Commands

```bash
pip install -r requirements.txt
python -m skiiipicks.build --out dashboard_data.json --log picks_log.csv --odds-cache odds_cache.json   # full build, 5-10 min
python -m skiiipicks.dashboard dashboard_data.json dashboard.html                                     # rebuild page only, instant
python -m skiiipicks.predict nfl KC BUF                                                               # away team first
```

For front-end work, only run the second command against the existing `dashboard_data.json`, then open
`dashboard.html` in a browser. There is no test suite; check the page at phone width (390px) in light and dark.

## Front-end rules

- One file, no frameworks or build step. Google Fonts (Barlow, Barlow Condensed) is the only external load.
- Colors are CSS tokens on `:root`, redefined for dark mode in both the `prefers-color-scheme` block and
  `[data-theme="dark"]`. Never use a literal color in a component.
- Keep `[hidden]{display:none!important}`; views, the slip sheet and tabs toggle with the `hidden` attribute.
- Five views switched by the bottom tab bar: Today, Games, Matchup, P&L, Model (`go(view)` in the script).
- Rating thresholds: winner picks green >= 70%, yellow >= 58%. Spread leans green >= 56%, yellow >= 52.4%.
  Props green >= 58%, yellow >= 52.4%. EV green >= +5%. Keep these in sync with `pnl.py`.
- The parlay slip lives in `localStorage` (`skp_slip`) and is per device.
- Phone first: nothing may scroll sideways at 390px except tables inside `.scroll`.

## Model rules

- Every reported accuracy number must be walk-forward (no peeking at the game being predicted).
- Don't claim an edge the backtest doesn't show; the dashboard copy states break-even (52.4% at -110) plainly.

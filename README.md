# skiiipicks: the skiii picks model for NFL, college football, NBA and MLB

Opponent-adjusted team ratings, game projections, and an honest backtest for four sports.
Everything downloads from free public data (nflverse, sportsdataverse, cfbfastR, baseballr).

## Quick start

```bash
pip install -r requirements.txt
python -m skiiipicks.build                  # downloads ~250 MB of data, fits, backtests (~5-10 min)
python -m skiiipicks.dashboard              # writes dashboard.html (open on any phone or browser)
python -m skiiipicks.predict nfl KC BUF     # away team first, then home
```

Re-run those first two commands any time to refresh the ratings with the latest games.

## How the model works

**Core rating engine (all sports, `core.py`).** For every team-game:

    points scored = league average + team offense − opponent defense + home edge

fit with ridge regression, so a team's offense is judged against the defenses it actually
faced. Recent games count more (exponential decay) and last season's games are
down-weighted, so early-season ratings lean on the prior year and fade it out.

**Sport-specific layers**

| Sport | Projection | Extra profile stats |
|---|---|---|
| NFL | 60% points ratings + 40% opponent-adjusted EPA/play (converted at ~36 pts per EPA/play) | Success rate, explosive plays, pass/rush EPA, red zone TD%, points per drive, points by quarter, points per drive by starting field position |
| College football | Points ratings (all divisions, so FCS games connect to FBS) | Success rate (adjusted), yards/play, explosive plays, red zone TD%, points per drive, quarter and field-position splits |
| NBA | Adjusted efficiency (points per 100 possessions) × projected pace | Four factors for and against, scoring mix (paint / 3s / FTs), fast break, quarter scoring |
| MLB | Runs ratings | Runs by inning group, first-inning scoring rate, first-5-innings runs, K/BB/HR rates, home/away splits |

**Stage 2: game context (`stack.py`).** A second model learns from several seasons of
walk-forward predictions how much to trust the ratings and how to adjust for things ratings can't see:

- NFL: rest advantage, starting-QB changes (listed starter vs. the team's usual starter), divisional games
- College football: Elo, recruiting talent, returning production, returning starting QB, travel distance

It's always trained only on seasons before the one it predicts. It also learns a cover-probability
curve from how past model-vs-Vegas disagreements actually turned out, which powers the spread leans.

Win probability = normal distribution of the projected margin, with spread set from backtest error.
Picks are tiered by win chance: Strong 75%+, Solid 65%+, Lean 57%+, otherwise Toss-up.

## What the backtests say (walk-forward, model never sees the game it predicts)

- **NFL 2023-2026 (917 games):** 63% winners. QB changes are worth about 2 points and rest about 0.2 per day, but Vegas prices them in: 47.8% ATS. Weather didn't help totals.
- **College football 2023-2026 (2,669 games):** 73.7% winners with the stage-2 factors vs 71.5% without, every season better. ATS 50.9%.
- **NBA 2025-26:** 69% winners over 1,213 games. No historical lines available to test ATS.
- **MLB 2026:** 55% winners. Doesn't know starting pitchers, which is the biggest single-game factor.

Break-even at −110 odds is 52.4%. None of these clear it reliably. Use the model to research
matchups and sanity-check lines, not as a betting system.

## Props (`props.py`)

- **Game and team props (all sports):** totals, margins, 1st half and 1st quarter (from each team's
  quarter-by-quarter scoring), team totals; MLB 1st-inning run and first-5-innings totals.
- **NFL player props:** passing yards/TDs, rushing yards, receptions, receiving yards, anytime TD.
- **NBA player props:** points, rebounds, assists, threes, PRA (switch on once the season's games are in the data).
- **College football player props:** same stats as NFL, from cfbfastR play-by-play names (QB, top 2 RBs, top 4 receivers).
- **MLB batter props:** hits, total bases, home run, strikeouts, RBIs for each team's 9 most-used recent batters,
  named via the Chadwick Bureau register. Uses the latest MLB play-by-play available (it lags a few weeks).
  No pitcher props: probable starters aren't in the free data.

Count stats (receptions, TDs thrown, rebounds, hits, total bases...) use Poisson / negative binomial
distributions fitted from the backtest; yardage and points use a normal distribution.

Projection = recent weighted form x how much the opponent allows of that stat. The over/under spread
for each stat is fitted from a walk-forward backtest on last season. Type a sportsbook line into the
dashboard to get the over/under chance. No free source has historical prop lines, so the model was
scored against season-average lines, which are softer than real ones.

## Ratings and P&L (`pnl.py`)

- Picks: green = 70%+ to win, yellow = 58-70%, red = under 58%.
  Spread leans: green = 56%+ to cover, yellow = 52.4-56%, red = below break-even.
- Each build saves the day's picks to `picks_log.csv`, then grades them once games are final.
  Picks for finished games are frozen; picks for upcoming games refresh each build.
- 1 unit per pick. NFL moneyline uses real odds; spreads use -110. NBA/MLB track win-loss only (no free odds).
- "This season, backtested" is rebuilt every time and kept separate from live results.
- On GitHub, use `workflow-for-github.yml` so the log is saved between runs (it needs `contents: write`).

## Top 10 picks

The 10 most confident winner picks across all sports for today only (Eastern time; fewer on light days),
moved down a little when the picked team has a key player Out or Doubtful. The list is saved in the
picks log, so the Top 10 has its own live record, plus a backtested record (the daily top 10 by win chance).

## Injuries (`injuries.py`)

NFL and NBA injury reports, from ESPN's live injury page when it responds, otherwise the daily
sportsdataverse feed. Players ruled out (Out, Doubtful, IR, suspended) are removed from player props;
questionable / day-to-day players are tagged. Key players (anyone with a player prop, or NBA players
averaging 20+ minutes) are called out on the pick. Injuries do not change projected scores.

The GitHub workflow runs at 8 AM Eastern and then hourly from 10 AM to midnight Eastern so reports
stay fresh into game time. Final inactives (NFL: 90 minutes before kickoff) can still land between runs.

## Sportsbook odds (`odds.py`)

Uses The Odds API (free key at the-odds-api.com; 500 credits/month). Add it on GitHub under
Settings > Secrets and variables > Actions > New repository secret, named `ODDS_API_KEY`.
Odds are fetched once a day (8 AM Eastern, `ODDS_HOURS: "12"` in UTC) for sports with games in the
next 3 days, about 3 credits per sport, and cached in `odds_cache.json` for the hourly runs.

Adds: best moneyline price and book per pick, the market's no-vig win chance, EV of the pick at the best
price, spreads/totals for NBA and MLB, real moneyline units for every sport, and closing line value
(first logged price vs. the last price before the game). Player prop lines need a paid plan.

## Ideas to improve it

- MLB: add starting pitcher ratings (pitcher_id is in the play-by-play).
- NFL/CFB: QB injuries and rest days; weather for totals.
- NBA: player availability (the biggest NBA factor). Back-to-backs are worth about 2.5 points but didn't improve accuracy in testing.
- Compare against closing lines over several seasons before trusting any edge.

## Files

    skiiipicks/core.py                 rating engine, win prob, backtest + ATS scoring
    skiiipicks/nfl.py | cfb.py | nba.py | mlb.py   sport modules
    skiiipicks/stack.py                stage-2 context model, multi-season testing, cover curve, pick tiers
    skiiipicks/build.py                runs everything, writes dashboard_data.json
    skiiipicks/dashboard.py            injects data into dashboard_template.html
    skiiipicks/predict.py              command-line projections

Use `--data-dir` with `build` to run from local copies (file names listed in `build.py::sources`).

If gambling stops being fun: 1-800-GAMBLER.

## Automatic daily updates (GitHub, free)

The file `.github/workflows/rebuild.yml` rebuilds everything every day at 8 AM Eastern and publishes
the dashboard to `https://YOUR-USERNAME.github.io/skiiipicks/`.

1. Make a free account at github.com.
2. Click **+** (top right), then **New repository**. Name it `skiiipicks`, choose **Public**, and click **Create repository**.
3. On the new repo page, click **uploading an existing file**. Drag in everything *inside* the unzipped
   `skiiipicks` folder (not the folder itself), then click **Commit changes**.
   - The `.github` folder is hidden on Mac (press Cmd+Shift+. in Finder to show it). If it doesn't upload,
     click **Add file > Create new file**, type `.github/workflows/rebuild.yml` as the name, paste the
     contents of that file, and commit.
4. Go to **Settings > Pages**. Under **Source**, pick **GitHub Actions**.
5. Go to the **Actions** tab, click **Rebuild skiiipicks**, then **Run workflow**. The first run takes about 10 minutes.
6. Open `https://YOUR-USERNAME.github.io/skiiipicks/` on your phone and add it to your home screen.

Good to know:
- Free GitHub Pages needs a public repo, so anyone with the address can see the dashboard and code.
- GitHub pauses scheduled runs after 60 days with no changes to the repo. It emails you first;
  click the link (or press **Run workflow**) to keep it going.
- Seasons switch automatically by date. If a new season's data isn't out yet, that sport uses last season.
- To change the time, edit the `cron` line in `rebuild.yml` (it's in UTC).

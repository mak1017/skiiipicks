"""MLB: opponent-adjusted runs model + inning-by-inning scoring profile.

Limitation: starting pitchers drive a large share of single-game outcomes and
this model rates teams, not that day's starter. Check probables before using it.
"""
import numpy as np
import pandas as pd

from .core import fit_ratings, team_game_rows, walk_forward, backtest_summary, win_prob

SCHED_URL = "https://raw.githubusercontent.com/sportsdataverse/baseballr-data/main/mlb/schedule/{}.parquet"
PBP_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/mlb_pbp/mlb_pbp_{}.parquet"

PARAMS = dict(alpha=25.0, half_life_days=120, prior_weight=0.3)
SIGMA = 4.3
TEAM_ABBR = {
    "Arizona Diamondbacks": "ARI", "Athletics": "ATH", "Atlanta Braves": "ATL", "Baltimore Orioles": "BAL",
    "Boston Red Sox": "BOS", "Chicago Cubs": "CHC", "Chicago White Sox": "CWS", "Cincinnati Reds": "CIN",
    "Cleveland Guardians": "CLE", "Colorado Rockies": "COL", "Detroit Tigers": "DET", "Houston Astros": "HOU",
    "Kansas City Royals": "KC", "Los Angeles Angels": "LAA", "Los Angeles Dodgers": "LAD", "Miami Marlins": "MIA",
    "Milwaukee Brewers": "MIL", "Minnesota Twins": "MIN", "New York Mets": "NYM", "New York Yankees": "NYY",
    "Philadelphia Phillies": "PHI", "Pittsburgh Pirates": "PIT", "San Diego Padres": "SD", "San Francisco Giants": "SF",
    "Seattle Mariners": "SEA", "St. Louis Cardinals": "STL", "Tampa Bay Rays": "TB", "Texas Rangers": "TEX",
    "Toronto Blue Jays": "TOR", "Washington Nationals": "WSH",
}


def load_games(path):
    s = pd.read_parquet(path)
    s = s[s["game_type"].isin(["R", "F", "D", "L", "W"])]
    s = s[s["home_team_name"].isin(TEAM_ABBR) & s["away_team_name"].isin(TEAM_ABBR)]
    done = s["abstract_state"].eq("Final") & s["detailed_state"].isin(["Final", "Game Over", "Completed Early"])
    g = pd.DataFrame({
        "game_pk": s["game_pk"], "season": s["season"].astype(int), "game_type": s["game_type"],
        "date": pd.to_datetime(s["game_date"]), "home": s["home_team_name"].map(TEAM_ABBR),
        "away": s["away_team_name"].map(TEAM_ABBR),
        "home_pts": s["home_score"].where(done), "away_pts": s["away_score"].where(done),
        "series": s.get("series_description"), "status": s["detailed_state"], "neutral": False,
    })
    return g.drop_duplicates("game_pk").sort_values("date").reset_index(drop=True)


def make_fit(games, cur_season):
    rows = team_game_rows(games, "home_pts", "away_pts")
    return lambda as_of: fit_ratings(rows, as_of, cur_season, **PARAMS)


def backtest(games, season):
    start = games.loc[games["season"] == season, "date"].min() + pd.Timedelta(days=21)
    bt = walk_forward(games, make_fit(games, season), start)
    return bt, backtest_summary(bt, SIGMA)


def inning_profile(pbp, games):
    p = pbp.merge(games[["game_pk", "home", "away"]], on="game_pk")
    p = p.sort_values(["game_pk", "at_bat_index"])
    p["bat"] = np.where(p["half_inning"] == "top", p["away"], p["home"])
    p["fld"] = np.where(p["half_inning"] == "top", p["home"], p["away"])
    p["score"] = np.where(p["half_inning"] == "top", p["away_score"], p["home_score"])
    half = p.groupby(["game_pk", "inning", "half_inning"]).agg(
        bat=("bat", "first"), fld=("fld", "first"), end=("score", "max")).reset_index()
    half = half.sort_values(["game_pk", "half_inning", "inning"])
    half["runs"] = half.groupby(["game_pk", "half_inning"])["end"].diff().fillna(half["end"]).clip(lower=0)
    half["bucket"] = pd.cut(half["inning"], [0, 3, 6, 9, 99], labels=["1-3", "4-6", "7-9", "Extras"])

    n_games = pd.concat([half.groupby("bat")["game_pk"].nunique()], axis=1)["game_pk"]
    off_b = half.groupby(["bat", "bucket"], observed=False)["runs"].sum().unstack().div(n_games, axis=0)
    def_b = half.groupby(["fld", "bucket"], observed=False)["runs"].sum().unstack().div(
        half.groupby("fld")["game_pk"].nunique(), axis=0)
    first = half[half["inning"] == 1]
    yrfi_o = first.groupby("bat")["runs"].apply(lambda r: (r > 0).mean())
    yrfi_d = first.groupby("fld")["runs"].apply(lambda r: (r > 0).mean())
    f5 = half[half["inning"] <= 5]
    f5_o = f5.groupby(["bat"])["runs"].sum() / n_games
    f5_d = f5.groupby(["fld"])["runs"].sum() / half.groupby("fld")["game_pk"].nunique()

    # plate-appearance outcomes
    ev = p["event_type"].fillna("")
    p["k"] = ev.str.contains("strikeout")
    p["bb"] = ev.isin(["walk", "intent_walk", "hit_by_pitch"])
    p["hr"] = ev.eq("home_run")
    pa_o = p.groupby("bat")[["k", "bb", "hr"]].mean()
    pa_d = p.groupby("fld")[["k", "bb", "hr"]].mean()
    return dict(off_b=off_b, def_b=def_b, yrfi_o=yrfi_o, yrfi_d=yrfi_d, f5_o=f5_o, f5_d=f5_d,
                pa_o=pa_o, pa_d=pa_d, n=n_games)


def team_profiles(games, ratings, prof, season):
    reg = games[(games["season"] == season) & (games["game_type"] == "R") & games["home_pts"].notna()]
    out = []
    for t in sorted(TEAM_ABBR.values()):
        h, a = reg[reg["home"] == t], reg[reg["away"] == t]
        rf = pd.concat([h["home_pts"], a["away_pts"]]); ra = pd.concat([h["away_pts"], a["home_pts"]])
        w = int((h["home_pts"] > h["away_pts"]).sum() + (a["away_pts"] > a["home_pts"]).sum())
        out.append({
            "team": t, "gp": int(len(rf)), "record": f"{w}-{len(rf) - w}",
            "pf": float(rf.mean()), "pa": float(ra.mean()),
            "off_rating": ratings.off.get(t, 0), "def_rating": ratings.dfn.get(t, 0),
            "net_rating": ratings.off.get(t, 0) + ratings.dfn.get(t, 0),
            "home_rf": float(h["home_pts"].mean()), "away_rf": float(a["away_pts"].mean()),
            "home_ra": float(h["away_pts"].mean()), "away_ra": float(a["home_pts"].mean()),
            "inn_for": {k: float(v) for k, v in prof["off_b"].loc[t].items()} if t in prof["off_b"].index else {},
            "inn_against": {k: float(v) for k, v in prof["def_b"].loc[t].items()} if t in prof["def_b"].index else {},
            "yrfi_off": float(prof["yrfi_o"].get(t, np.nan)), "yrfi_def": float(prof["yrfi_d"].get(t, np.nan)),
            "f5_for": float(prof["f5_o"].get(t, np.nan)), "f5_against": float(prof["f5_d"].get(t, np.nan)),
            "off_k": float(prof["pa_o"].loc[t, "k"]), "off_bb": float(prof["pa_o"].loc[t, "bb"]),
            "off_hr": float(prof["pa_o"].loc[t, "hr"]), "pit_k": float(prof["pa_d"].loc[t, "k"]),
            "pit_bb": float(prof["pa_d"].loc[t, "bb"]), "pit_hr": float(prof["pa_d"].loc[t, "hr"]),
            "pbp_games": int(prof["n"].get(t, 0)),
        })
    return out


def upcoming(games, ratings):
    fut = games[games["home_pts"].isna() & games["home"].notna() & games["away"].notna()]
    fut = fut[~fut["status"].isin(["Postponed", "Cancelled"])].sort_values("date").head(12)
    res = []
    for _, g in fut.iterrows():
        ph, pa = ratings.predict(g["home"], g["away"])
        res.append({"date": g["date"].strftime("%Y-%m-%d"), "home": g["home"], "away": g["away"],
                    "series": g.get("series"), "pred_home": ph, "pred_away": pa,
                    "win_home": win_prob(ph - pa, SIGMA), "line_home": None, "total_line": None})
    return res

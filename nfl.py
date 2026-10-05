"""NFL: points model + EPA efficiency + where points are scored/allowed."""
import numpy as np
import pandas as pd

from .core import fit_ratings, team_game_rows, walk_forward, backtest_summary, win_prob, rank_dict, Blend, epa_rows

GAMES_URL = "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
PBP_URL = "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.parquet"

PARAMS = dict(alpha=6.0, half_life_days=150, prior_weight=0.45)
EPA_PARAMS = dict(alpha=1.5, half_life_days=150, prior_weight=0.4)
EPA_TO_PTS = 36.0   # fitted on 2025: 1 EPA/play of margin is about 36 points
BLEND_W = 0.6
SIGMA = 13.2


def load_pbp(seasons, path=PBP_URL):
    cols = None
    return pd.concat([pd.read_parquet(path.format(y)) for y in seasons], ignore_index=True)


def game_epa(pbp):
    pl = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()]
    ge = pl.groupby(["game_id", "posteam", "defteam"]).agg(
        epa=("epa", "mean"), date=("game_date", "first"), home=("home_team", "first"),
        season=("season", "first")).reset_index()
    return epa_rows(pd.DataFrame({"date": ge["date"], "team": ge["posteam"], "opp": ge["defteam"],
                                  "epa": ge["epa"], "is_home": ge["posteam"] == ge["home"],
                                  "season": ge["season"], "neutral": False}))


def load_games(path=GAMES_URL, seasons=(2024, 2025, 2026)):
    g = pd.read_csv(path)
    g = g[g["season"].isin(seasons)].copy()
    return pd.DataFrame({
        "game_id": g["game_id"], "season": g["season"], "week": g["week"],
        "date": pd.to_datetime(g["gameday"]), "home": g["home_team"], "away": g["away_team"],
        "home_pts": g["home_score"], "away_pts": g["away_score"],
        "neutral": g["location"].eq("Neutral"),
        # nflverse spread_line is the home team's expected margin
        "line_home": g["spread_line"], "total_line": g["total_line"],
    }).reset_index(drop=True)


def make_fit(games, cur_season, erows=None):
    rows = team_game_rows(games, "home_pts", "away_pts")

    def fit(as_of):
        pts = fit_ratings(rows, as_of, cur_season, **PARAMS)
        eff = fit_ratings(erows, as_of, cur_season, **EPA_PARAMS) if erows is not None else None
        return Blend(pts, eff, EPA_TO_PTS, BLEND_W)
    return fit


def backtest(games, season, erows=None):
    start = games.loc[games["season"] == season, "date"].min()
    prev = games[games["season"] >= season - 1]
    bt = walk_forward(prev, make_fit(prev, season, erows), start)
    bt = bt[bt["season"] == season]
    return bt, backtest_summary(bt, SIGMA)


def _quarter_points(pbp):
    q = pbp[pbp["qtr"].between(1, 5)].groupby(["game_id", "qtr"]).agg(
        h=("total_home_score", "max"), a=("total_away_score", "max"),
        home=("home_team", "first"), away=("away_team", "first")).reset_index()
    q["qtr"] = q["qtr"].clip(upper=4)  # fold OT into Q4
    q = q.groupby(["game_id", "qtr"]).agg(h=("h", "max"), a=("a", "max"),
                                          home=("home", "first"), away=("away", "first")).reset_index()
    q = q.sort_values(["game_id", "qtr"])
    q["h_pts"] = q.groupby("game_id")["h"].diff().fillna(q["h"])
    q["a_pts"] = q.groupby("game_id")["a"].diff().fillna(q["a"])
    rows = pd.concat([
        q.rename(columns={"home": "team", "h_pts": "pf", "a_pts": "pa"})[["game_id", "team", "qtr", "pf", "pa"]],
        q.rename(columns={"away": "team", "a_pts": "pf", "h_pts": "pa"})[["game_id", "team", "qtr", "pf", "pa"]],
    ])
    return rows.groupby(["team", "qtr"])[["pf", "pa"]].mean()


def _drive_table(pbp):
    p = pbp[pbp["posteam"].notna() & pbp["fixed_drive"].notna()]
    d = p.groupby(["game_id", "posteam", "fixed_drive"]).agg(
        defteam=("defteam", "first"), start=("yardline_100", "first"),
        best=("yardline_100", "min"), result=("fixed_drive_result", "first")).reset_index()
    d["pts"] = d["result"].map({"Touchdown": 6.95, "Field goal": 3.0}).fillna(0.0)
    d["zone"] = pd.cut(d["start"], [0, 49, 74, 100], labels=["Opp territory", "Own 26-50", "Own 1-25"])
    d["rz"] = d["best"] <= 20
    return d


def team_profiles(pbp, ratings, cur_season):
    plays = pbp[pbp["play_type"].isin(["pass", "run"]) & pbp["epa"].notna()].copy()
    plays["explosive"] = ((plays["pass"] == 1) & (plays["yards_gained"] >= 20)) | \
                         ((plays["rush"] == 1) & (plays["yards_gained"] >= 10))

    # Opponent-adjusted EPA/play
    ge = plays.groupby(["game_id", "posteam", "defteam"]).agg(epa=("epa", "mean"), date=("game_date", "first"),
                                                               home=("home_team", "first")).reset_index()
    rows = pd.DataFrame({"date": pd.to_datetime(ge["date"]), "team": ge["posteam"], "opp": ge["defteam"],
                         "y": ge["epa"], "loc": np.where(ge["posteam"] == ge["home"], 0.5, -0.5),
                         "season": cur_season})
    epa_r = fit_ratings(rows, pd.Timestamp.today() + pd.Timedelta(days=2), cur_season, alpha=2.0, half_life_days=9999, prior_weight=1)

    off = plays.groupby("posteam")
    dfn = plays.groupby("defteam")

    def split(gb, kind):
        s = gb.apply(lambda x: x.loc[x[kind] == 1, "epa"].mean(), include_groups=False)
        return s

    drives = _drive_table(pbp)
    qp = _quarter_points(pbp)
    games_played = pbp.groupby("home_team")["game_id"].nunique().add(
        pbp.groupby("away_team")["game_id"].nunique(), fill_value=0)

    out = []
    teams = sorted(plays["posteam"].unique())
    off_pass, off_rush = split(off, "pass"), split(off, "rush")
    def_pass, def_rush = split(dfn, "pass"), split(dfn, "rush")
    for t in teams:
        od, dd = drives[drives["posteam"] == t], drives[drives["defteam"] == t]
        zone_o = od.groupby("zone", observed=False)["pts"].mean()
        zone_d = dd.groupby("zone", observed=False)["pts"].mean()
        rz_o, rz_d = od[od["rz"]], dd[dd["rz"]]
        q = qp.loc[t] if t in qp.index.get_level_values(0) else None
        out.append({
            "team": t, "gp": int(games_played.get(t, 0)),
            "off_rating": ratings.off.get(t, 0), "def_rating": ratings.dfn.get(t, 0),
            "net_rating": ratings.off.get(t, 0) + ratings.dfn.get(t, 0),
            "off_epa": float(off.get_group(t)["epa"].mean()), "def_epa": float(dfn.get_group(t)["epa"].mean()),
            "adj_off_epa": epa_r.off.get(t, 0), "adj_def_epa": epa_r.dfn.get(t, 0),
            "off_sr": float(off.get_group(t)["success"].mean()), "def_sr": float(dfn.get_group(t)["success"].mean()),
            "off_pass_epa": float(off_pass.get(t, np.nan)), "off_rush_epa": float(off_rush.get(t, np.nan)),
            "def_pass_epa": float(def_pass.get(t, np.nan)), "def_rush_epa": float(def_rush.get(t, np.nan)),
            "off_explosive": float(off.get_group(t)["explosive"].mean()),
            "def_explosive": float(dfn.get_group(t)["explosive"].mean()),
            "plays_pg": len(off.get_group(t)) / max(games_played.get(t, 1), 1),
            "off_ppd": float(od["pts"].mean()), "def_ppd": float(dd["pts"].mean()),
            "off_rz_td": float((rz_o["result"] == "Touchdown").mean()) if len(rz_o) else None,
            "def_rz_td": float((rz_d["result"] == "Touchdown").mean()) if len(rz_d) else None,
            "zones_off": {str(k): (None if pd.isna(v) else float(v)) for k, v in zone_o.items()},
            "zones_def": {str(k): (None if pd.isna(v) else float(v)) for k, v in zone_d.items()},
            "q_for": [float(q.loc[i, "pf"]) if q is not None and i in q.index else 0 for i in (1, 2, 3, 4)],
            "q_against": [float(q.loc[i, "pa"]) if q is not None and i in q.index else 0 for i in (1, 2, 3, 4)],
        })
    return out


def upcoming(games, ratings, n_weeks=1):
    fut = games[games["home_pts"].isna()].sort_values("date")
    if fut.empty:
        return []
    weeks = sorted(fut["week"].unique())[:n_weeks]
    fut = fut[fut["week"].isin(weeks)]
    res = []
    for _, g in fut.iterrows():
        ph, pa = ratings.predict(g["home"], g["away"], bool(g["neutral"]))
        res.append({"date": g["date"].strftime("%Y-%m-%d"), "week": int(g["week"]), "home": g["home"],
                    "away": g["away"], "pred_home": ph, "pred_away": pa,
                    "win_home": win_prob(ph - pa, SIGMA),
                    "line_home": None if pd.isna(g["line_home"]) else float(g["line_home"]),
                    "total_line": None if pd.isna(g["total_line"]) else float(g["total_line"])})
    return res

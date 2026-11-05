"""MLB: opponent-adjusted runs model + starting pitchers + inning-by-inning scoring profile.

Stage 1 rates teams on runs. Stage 2 (stack.WinStack) adds both starting pitchers, rated from every
plate appearance they've faced before that day (wOBA allowed and FIP per batter, shrunk toward league
average, scaled by how deep they usually go). Upcoming starters come from MLB's probable pitchers.
"""
import numpy as np
import pandas as pd

from .core import fit_ratings, team_game_rows, walk_forward, backtest_summary, win_prob

SCHED_URL = "https://raw.githubusercontent.com/sportsdataverse/baseballr-data/main/mlb/schedule/{}.parquet"
PBP_URL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download/mlb_pbp/mlb_pbp_{}.parquet"

PARAMS = dict(alpha=25.0, half_life_days=365, prior_weight=0.6)  # longer memory tested better (2025-26)
PROBABLES_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={}&endDate={}&hydrate=probablePitcher"
SP_SHRINK = 250   # batters faced of league-average performance added to every pitcher
LW = {"walk": .69, "intent_walk": .69, "hit_by_pitch": .72, "single": .89, "double": 1.27, "triple": 1.62, "home_run": 2.1}
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
        res.append({"game_pk": int(g["game_pk"]), "date": g["date"].strftime("%Y-%m-%d"), "home": g["home"], "away": g["away"],
                    "series": g.get("series"), "pred_home": ph, "pred_away": pa,
                    "win_home": win_prob(ph - pa, SIGMA), "line_home": None, "total_line": None})
    return res


# ---------------- starting pitchers ----------------

def _pa(pbp, games):
    p = pbp.merge(games[["game_pk", "date"]], on="game_pk")
    ev = p["event_type"].fillna("")
    p = p[ev != ""].copy(); ev = p["event_type"]
    p["w"] = ev.map(LW).fillna(0.0)
    p["k"] = ev.str.contains("strikeout").astype(float)
    p["bb"] = ev.isin(["walk", "intent_walk", "hit_by_pitch"]).astype(float)
    p["hr"] = ev.eq("home_run").astype(float)
    p["pa"] = 1.0
    return p


def _rate(c, lg, n0=SP_SHRINK):
    fip_lg = 13 * lg["hr"] + 3 * lg["bb"] - 2 * lg["k"]
    woba = (c["w"] + n0 * lg["w"]) / (c["pa"] + n0) - lg["w"]
    fip = ((13 * c["hr"] + 3 * c["bb"] - 2 * c["k"]) + n0 * fip_lg) / (c["pa"] + n0) - fip_lg
    depth = (c["spa"] + 22 * 3) / (c["starts"] + 3)   # batters faced per start, shrunk toward 22
    return woba, fip, depth


def starters(pbp, games):
    """Returns (per-game starter table with pre-game ratings, current rating per pitcher)."""
    p = _pa(pbp, games)
    first = p[p["inning"] == 1].sort_values("at_bat_index").groupby(["game_pk", "half_inning"])["pitcher_id"].first().unstack()
    st = pd.DataFrame({"game_pk": first.index, "home_sp": first.get("top"), "away_sp": first.get("bottom")}).reset_index(drop=True)
    pg = p.groupby(["pitcher_id", "game_pk", "date"])[["w", "k", "bb", "hr", "pa"]].sum().reset_index()
    starts = pd.concat([st[["game_pk", "home_sp"]].rename(columns={"home_sp": "pitcher_id"}),
                        st[["game_pk", "away_sp"]].rename(columns={"away_sp": "pitcher_id"})]).assign(start=1.0)
    pg = pg.merge(starts, on=["game_pk", "pitcher_id"], how="left").fillna({"start": 0.0})
    pg["spa"] = pg["pa"] * pg["start"]
    pg = pg.sort_values(["pitcher_id", "date", "game_pk"])
    lg = p[["w", "k", "bb", "hr"]].mean()
    cum = pg.groupby("pitcher_id")[["w", "k", "bb", "hr", "pa", "start", "spa"]].cumsum()
    pre = (cum - pg[["w", "k", "bb", "hr", "pa", "start", "spa"]]).rename(columns={"start": "starts"})
    pg["woba"], pg["fip"], pg["depth"] = _rate(pre, lg)
    tot = pg.groupby("pitcher_id")[["w", "k", "bb", "hr", "pa", "start", "spa"]].sum().rename(columns={"start": "starts"})
    cur = pd.DataFrame(dict(zip(["woba", "fip", "depth"], _rate(tot, lg))))
    cur["bf"] = tot["pa"]; cur["starts"] = tot["starts"]
    key = pg[["pitcher_id", "game_pk", "woba", "fip", "depth"]]
    st = st.merge(key.rename(columns={"pitcher_id": "home_sp", **{c: "h_" + c for c in ("woba", "fip", "depth")}}),
                  on=["game_pk", "home_sp"], how="left")
    st = st.merge(key.rename(columns={"pitcher_id": "away_sp", **{c: "a_" + c for c in ("woba", "fip", "depth")}}),
                  on=["game_pk", "away_sp"], how="left")
    return st, cur


def sp_features(df):
    """df has h_/a_ woba, fip, depth. Positive = home starter better, weighted by expected innings."""
    out = df.copy()
    for c in ("woba", "fip"):
        out["sp_" + c] = (out["a_" + c].fillna(0) * out["a_depth"].fillna(22)
                          - out["h_" + c].fillna(0) * out["h_depth"].fillna(22)) / 22
    return out


def fetch_probables(start, end, timeout=15):
    """{game_pk: {"home": (id, name), "away": (id, name)}} from MLB's public schedule API."""
    import json, urllib.request
    req = urllib.request.Request(PROBABLES_URL.format(start, end), headers={"User-Agent": "Mozilla/5.0 skiiipicks"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        js = json.loads(r.read().decode("utf-8"))
    out = {}
    for d in js.get("dates", []) or []:
        for g in d.get("games", []) or []:
            e = {}
            for side in ("home", "away"):
                pp = ((g.get("teams") or {}).get(side) or {}).get("probablePitcher") or {}
                if pp.get("id"):
                    e[side] = (int(pp["id"]), pp.get("fullName"))
            out[int(g["gamePk"])] = e
    return out

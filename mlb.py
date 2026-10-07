"""MLB: opponent-adjusted runs model + inning-by-inning scoring profile.

Team ratings come from runs. When MLB has posted probable starters, each one moves the projection by
how many runs his recent strikeout, walk and home run rates save over his usual outing, at a rate fitted
on the previous season.
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
        res.append({"date": g["date"].strftime("%Y-%m-%d"), "game_pk": int(g["game_pk"]), "home": g["home"], "away": g["away"],
                    "series": g.get("series"), "pred_home": ph, "pred_away": pa,
                    "win_home": win_prob(ph - pa, SIGMA), "line_home": None, "total_line": None})
    return res


# ---------------- starting pitchers ----------------
# Starters come from the play-by-play (the first pitcher each team used). Probable starters for upcoming
# games come from MLB's public stats API.

PROBABLES_URL = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={}&endDate={}"
                 "&hydrate=probablePitcher")
WALKS = {"walk", "intent_walk", "hit_by_pitch"}
SP_PRIOR_BF = 120      # batters faced of league-average pitching blended into every starter's rates


def starter_games(pbp, games):
    """One row per starting pitcher per game: batters faced, strikeouts, walks, homers allowed."""
    from .props import PA_EVENTS
    p = pbp.merge(games[["game_pk", "home", "away", "date", "season"]], on="game_pk")
    first = p.sort_values("at_bat_index").groupby(["game_pk", "half_inning"]).first().reset_index()
    first = first[first["inning"] == 1]
    top = first["half_inning"] == "top"           # home team pitches the top of the 1st
    st = pd.DataFrame({"game_pk": first["game_pk"], "half_inning": first["half_inning"], "pid": first["pitcher_id"],
                       "team": np.where(top, first["home"], first["away"]), "opp": np.where(top, first["away"], first["home"]),
                       "date": first["date"], "season": first["season"]})
    pa = p[p["event_type"].isin(PA_EVENTS)]
    pa = pa.merge(st[["game_pk", "half_inning", "pid"]], on=["game_pk", "half_inning"])
    pa = pa[pa["pitcher_id"] == pa["pid"]]
    ev = pa["event_type"]
    agg = pa.assign(bf=1, k=ev.str.startswith("strikeout").astype(int), bb=ev.isin(WALKS).astype(int),
                    hr=(ev == "home_run").astype(int)).groupby(["game_pk", "half_inning"])[["bf", "k", "bb", "hr"]].sum()
    st = st.merge(agg.reset_index(), on=["game_pk", "half_inning"], how="inner").drop(columns="half_inning")
    return st.rename(columns={"game_pk": "game_id"}).sort_values("date").reset_index(drop=True)


def pitcher_form(sg, as_of, cur_season):
    """Each starter's recent rates (shrunk toward league average) and usual batters faced, before as_of."""
    from .props import _ewma
    h = sg[sg["date"] < as_of].copy()
    if h.empty:
        return pd.DataFrame(columns=["k_pp", "bb_pp", "hr_pp", "bf", "fip_pp", "n_cur", "starts", "team", "last_date"])
    lg = {c: h[c].sum() / h["bf"].sum() for c in ("k", "bb", "hr")}
    tot = h.groupby("pid")[["bf", "k", "bb", "hr"]].sum()
    rec = _ewma(h.assign(**{c + "_r": h[c] / h["bf"] for c in ("k", "bb", "hr")}),
                ["k_r", "bb_r", "hr_r"], "pid", hl=8, cur_season=cur_season, prior_w=0.5, last_n=25)
    bf = _ewma(h, ["bf"], "pid", hl=3, cur_season=cur_season, prior_w=0.3, last_n=8)
    f = rec.join(bf[["bf"]])
    w = tot["bf"].reindex(f.index).clip(upper=400)
    for c in ("k", "bb", "hr"):
        f[c + "_pp"] = (f[c + "_r"] * w + lg[c] * SP_PRIOR_BF) / (w + SP_PRIOR_BF)
    f["fip_pp"] = 13 * f["hr_pp"] + 3 * f["bb_pp"] - 2 * f["k_pp"]
    f.attrs["lg_fip"] = 13 * lg["hr"] + 3 * lg["bb"] - 2 * lg["k"]
    last = h.groupby("pid").tail(1).set_index("pid")
    f["team"], f["last_date"] = last["team"], last["date"]
    f["starts"] = h.groupby("pid").size()
    return f


def starter_effects(sg, games, cur_season):
    """Walk-forward: each game's starters' run-prevention edge vs. league (runs above average, before
    scaling), using only starts made before that date."""
    rows = []
    starts = sg.set_index(["game_id", "team"])["pid"]
    for d, day in games.groupby("date"):
        f = pitcher_form(sg, d, cur_season)
        lg = f.attrs.get("lg_fip", 0)
        for g in day.itertuples():
            r = {"game_pk": g.game_pk}
            for side, team in (("home", g.home), ("away", g.away)):
                pid = starts.get((g.game_pk, team))
                r[side + "_sp"] = sp_value(f, pid, lg)
            rows.append(r)
    return pd.DataFrame(rows)


def sp_value(f, pid, lg):
    """Runs a starter saves vs. a league-average starter over his usual outing (positive = better), unscaled."""
    if pid is None or pid not in f.index:
        return 0.0
    r = f.loc[pid]
    return float((lg - r["fip_pp"]) * r["bf"])


def fetch_probables(dates, timeout=15):
    """{game_pk: {"home": (id, name) or None, "away": ...}} from MLB's stats API for the given dates."""
    import json
    import urllib.request
    if not dates:
        return {}
    url = PROBABLES_URL.format(min(dates), max(dates))
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 skiiipicks"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        js = json.loads(r.read().decode("utf-8"))
    out = {}
    for d in js.get("dates", []) or []:
        for g in d.get("games", []) or []:
            sides = {}
            for side in ("home", "away"):
                pp = ((g.get("teams") or {}).get(side) or {}).get("probablePitcher") or {}
                sides[side] = (int(pp["id"]), pp.get("fullName")) if pp.get("id") else None
            out[int(g["gamePk"])] = sides
    return out


def fit_starter_effect(bt, se):
    """Runs per unit of starter value (sp_value), least squares on the backtest's run residuals."""
    b = bt.merge(se, on="game_pk")
    x = np.r_[b["away_sp"], b["home_sp"]]
    y = np.r_[b["home_pts"] - b["pred_home"], b["away_pts"] - b["pred_away"]]
    y = y - y.mean()
    return float(-(x @ y) / (x @ x)) if (x @ x) > 0 else 0.0


def apply_starter_effect(bt, se, c):
    b = bt.merge(se, on="game_pk", how="left").fillna({"home_sp": 0.0, "away_sp": 0.0})
    b["pred_home"] = b["pred_home"] - c * b["away_sp"]
    b["pred_away"] = b["pred_away"] - c * b["home_sp"]
    return b


def attach_starters(ups, probables, form, lg, c, names):
    """Add probable starters to upcoming games and move projections by how good each one is."""
    for u in ups:
        pr = probables.get(u.get("game_pk")) or {}
        sides, val = {}, {}
        for side in ("home", "away"):
            pp = pr.get(side)
            if not pp:
                sides[side], val[side] = None, 0.0
                continue
            pid, nm = pp
            known = pid in form.index
            r = form.loc[pid] if known else None
            sides[side] = {"id": pid, "name": nm or names.get(pid, f"Pitcher {pid}"),
                           "starts": int(r["starts"]) if known else 0,
                           "k_pct": float(r["k_pp"]) if known else None, "bb_pct": float(r["bb_pp"]) if known else None,
                           "bf": float(r["bf"]) if known else None}
            val[side] = sp_value(form, pid, lg)
        u["starters"] = sides
        if not (sides["home"] or sides["away"]):
            continue
        u["pred_home"] -= c * val["away"]
        u["pred_away"] -= c * val["home"]
        u["win_home"] = win_prob(u["pred_home"] - u["pred_away"], SIGMA)
        u["sp_adj"] = {"home": -c * val["away"], "away": -c * val["home"]}
    return ups

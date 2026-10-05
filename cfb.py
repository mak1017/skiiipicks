"""College football: points model + success-rate efficiency + scoring zones.

The play-by-play feed uses NCAA team names ("Iowa St.") while schedules use
CFBD names ("Iowa State"), so teams are linked by matching games on week and
final score.
"""
import numpy as np
import pandas as pd

from .core import fit_ratings, team_game_rows, walk_forward, backtest_summary, win_prob, Blend

REPO = "https://raw.githubusercontent.com/sportsdataverse/cfbfastR-data/main/schedules/parquet/cfb_schedules_{}.parquet"
REL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
SCHED_URL = {2026: f"{REL}/cfb_schedules/cfb_schedules_2026.parquet"}
LINE_URL = REL + "/cfb_matchup_line/cfb_matchup_line_{}.parquet"
PBP_URL = REL + "/ncaa_mfb_pbp_cfbfastr/ncaa_mfb_pbp_cfbfastr_{}.parquet"

PARAMS = dict(alpha=1.0, half_life_days=200, prior_weight=0.4)
SR_PARAMS = dict(alpha=1.0, half_life_days=200, prior_weight=0.4)
SR_TO_PTS = 0.0  # set by calibrate_sr(); 0 disables the efficiency blend
BLEND_W = 0.7
SIGMA = 15.5


def load_games(sched_paths: dict, line_paths: dict):
    frames = []
    for season, path in sched_paths.items():
        s = pd.read_parquet(path)
        s = s[s.get("season_type", "regular").isin(["regular", "postseason"])] if "season_type" in s else s
        frames.append(pd.DataFrame({
            "game_id": s["game_id"], "season": season, "week": s["week"],
            "date": pd.to_datetime(s["start_date"], utc=True).dt.tz_localize(None).dt.normalize(),
            "home": s["home_team"], "away": s["away_team"],
            "home_pts": s["home_points"], "away_pts": s["away_points"],
            "neutral": s["neutral_site"].fillna(False).astype(bool),
            "home_div": s.get("home_division"), "away_div": s.get("away_division"),
            "home_conf": s.get("home_conference"), "away_conf": s.get("away_conference"),
        }))
    g = pd.concat(frames, ignore_index=True)
    lines = []
    for season, path in line_paths.items():
        l = pd.read_parquet(path)
        lines.append(pd.DataFrame({"game_id": l["game_id"], "line_home": -l["spread"]}))
    if lines:
        g = g.merge(pd.concat(lines).drop_duplicates("game_id"), on="game_id", how="left")
    g["total_line"] = np.nan
    return g


def make_fit(games, cur_season, sr_rows=None):
    rows = team_game_rows(games, "home_pts", "away_pts")

    def fit(as_of):
        pts = fit_ratings(rows, as_of, cur_season, **PARAMS)
        eff = None
        if sr_rows is not None and SR_TO_PTS:
            eff = fit_ratings(sr_rows, as_of, cur_season, **SR_PARAMS)
        return Blend(pts, eff, SR_TO_PTS, BLEND_W)
    return fit


def backtest(games, season, sr_rows=None):
    start = games.loc[games["season"] == season, "date"].min()
    prev = games[games["season"] >= season - 1]
    bt = walk_forward(prev, make_fit(prev, season, sr_rows), start)
    bt = bt[(bt["season"] == season)]
    fbs = bt[(bt["home_div"] == "fbs") & (bt["away_div"] == "fbs")] if "home_div" in bt else bt
    return fbs, backtest_summary(fbs, SIGMA)


# ---------- play-by-play features ----------

def prep_pbp(pbp):
    p = pbp.copy()
    p["is_play"] = ((p["rush"] == 1) | (p["pass"] == 1)) & p["down"].notna()
    need = np.select([p["down"] == 1, p["down"] == 2], [0.5 * p["distance"], 0.7 * p["distance"]], p["distance"])
    p["success"] = (p["yards_gained"] >= need).astype(float)
    p["explosive"] = ((p["pass"] == 1) & (p["yards_gained"] >= 20)) | ((p["rush"] == 1) & (p["yards_gained"] >= 10))
    p["home_score"] = np.where(p["pos_team"] == p["home"], p["pos_team_score"], p["def_pos_team_score"])
    p["away_score"] = np.where(p["pos_team"] == p["home"], p["def_pos_team_score"], p["pos_team_score"])
    return p


def name_map(pbp, games):
    """Map NCAA pbp names -> schedule names via games with identical week + final score."""
    fin = pbp.groupby("game_id").agg(week=("week", "first"), home=("home", "first"), away=("away", "first"),
                                     hs=("home_score", "max"), as_=("away_score", "max")).reset_index()
    sch = games.dropna(subset=["home_pts"])
    votes = {}
    for _, f in fin.iterrows():
        m = sch[(sch["week"] == f["week"]) & (
            ((sch["home_pts"] == f["hs"]) & (sch["away_pts"] == f["as_"])) |
            ((sch["home_pts"] == f["as_"]) & (sch["away_pts"] == f["hs"])))]
        if len(m) != 1:
            continue
        m = m.iloc[0]
        if m["home_pts"] == f["hs"] and m["home_pts"] != m["away_pts"]:
            pairs = [(f["home"], m["home"]), (f["away"], m["away"])]
        elif m["home_pts"] != m["away_pts"]:
            pairs = [(f["home"], m["away"]), (f["away"], m["home"])]
        else:
            continue
        for a, b in pairs:
            votes.setdefault(a, {}).setdefault(b, 0)
            votes[a][b] += 1
    return {a: max(v, key=v.get) for a, v in votes.items()}


def game_sr_rows(pbp, nmap, season):
    pl = pbp[pbp["is_play"]]
    ge = pl.groupby(["game_id", "pos_team", "def_pos_team"]).agg(
        sr=("success", "mean"), home=("home", "first"), wk=("week", "first")).reset_index()
    ge["team"] = ge["pos_team"].map(nmap)
    ge["opp"] = ge["def_pos_team"].map(nmap)
    ge = ge.dropna(subset=["team", "opp"])
    # approximate date from week (Saturday of that week)
    start = pd.Timestamp(f"{season}-08-29")
    return pd.DataFrame({"date": start + pd.to_timedelta((ge["wk"] - 1) * 7, unit="D"),
                         "team": ge["team"], "opp": ge["opp"], "y": ge["sr"],
                         "loc": np.where(ge["pos_team"] == ge["home"], 0.5, -0.5), "season": season})


def team_profiles(pbp, nmap, ratings, sr_ratings, games, season):
    p = pbp.copy()
    p["team"] = p["pos_team"].map(nmap)
    p["opp"] = p["def_pos_team"].map(nmap)
    pl = p[p["is_play"] & p["team"].notna() & p["opp"].notna()]
    dr = p[p["drive_id"].notna() & p["team"].notna()].groupby("drive_id").agg(
        team=("team", "first"), opp=("opp", "first"), start=("yards_to_goal", "first"),
        best=("yards_to_goal", "min"), result=("drive_result", "first")).reset_index()
    dr = dr[~dr["result"].isin(["HALF", "UNKNOWN"])]
    dr["pts"] = dr["result"].map({"TD": 7.0, "FG": 3.0}).fillna(0.0)
    dr["zone"] = pd.cut(dr["start"], [-1, 49, 74, 100], labels=["Opp territory", "Own 26-50", "Own 1-25"])
    dr["rz"] = dr["best"] <= 20

    # points by quarter
    q = p[p["period"].between(1, 7)].copy()
    q["qtr"] = q["period"].clip(upper=4)
    qs = q.groupby(["game_id", "qtr"]).agg(h=("home_score", "max"), a=("away_score", "max"),
                                            home=("home", "first"), away=("away", "first")).reset_index()
    qs = qs.sort_values(["game_id", "qtr"])
    qs[["h", "a"]] = qs.groupby("game_id")[["h", "a"]].cummax()
    qs["hp"] = qs.groupby("game_id")["h"].diff().fillna(qs["h"])
    qs["ap"] = qs.groupby("game_id")["a"].diff().fillna(qs["a"])
    qrows = pd.concat([
        pd.DataFrame({"team": qs["home"].map(nmap), "qtr": qs["qtr"], "pf": qs["hp"], "pa": qs["ap"]}),
        pd.DataFrame({"team": qs["away"].map(nmap), "qtr": qs["qtr"], "pf": qs["ap"], "pa": qs["hp"]}),
    ]).dropna(subset=["team"])
    qp = qrows.groupby(["team", "qtr"])[["pf", "pa"]].mean()

    cur = games[(games["season"] == season) & games["home_pts"].notna()]
    fbs = set(cur.loc[cur["home_div"] == "fbs", "home"]) | set(cur.loc[cur["away_div"] == "fbs", "away"])
    conf = pd.concat([cur[["home", "home_conf"]].set_axis(["t", "c"], axis=1),
                      cur[["away", "away_conf"]].set_axis(["t", "c"], axis=1)]).dropna().drop_duplicates("t").set_index("t")["c"]
    gp = pd.concat([cur["home"], cur["away"]]).value_counts()
    pf = pd.concat([cur.set_index("home")["home_pts"], cur.set_index("away")["away_pts"]]).groupby(level=0).mean()
    pa = pd.concat([cur.set_index("home")["away_pts"], cur.set_index("away")["home_pts"]]).groupby(level=0).mean()

    out = []
    og, dg = pl.groupby("team"), pl.groupby("opp")
    for t in sorted(fbs):
        if t not in og.groups or t not in dg.groups:
            continue
        o, d = og.get_group(t), dg.get_group(t)
        od, dd = dr[dr["team"] == t], dr[dr["opp"] == t]
        rz_o, rz_d = od[od["rz"]], dd[dd["rz"]]
        qq = qp.loc[t] if t in qp.index.get_level_values(0) else None
        out.append({
            "team": t, "conf": conf.get(t, ""), "gp": int(gp.get(t, 0)),
            "pf": float(pf.get(t, np.nan)), "pa": float(pa.get(t, np.nan)),
            "off_rating": ratings.off.get(t, 0), "def_rating": ratings.dfn.get(t, 0),
            "net_rating": ratings.off.get(t, 0) + ratings.dfn.get(t, 0),
            "off_sr": float(o["success"].mean()), "def_sr": float(d["success"].mean()),
            "adj_off_sr": sr_ratings.off.get(t, 0), "adj_def_sr": sr_ratings.dfn.get(t, 0),
            "off_ypp": float(o["yards_gained"].mean()), "def_ypp": float(d["yards_gained"].mean()),
            "off_explosive": float(o["explosive"].mean()), "def_explosive": float(d["explosive"].mean()),
            "off_pass_rate": float((o["pass"] == 1).mean()),
            "plays_pg": len(o) / max(gp.get(t, 1), 1),
            "off_ppd": float(od["pts"].mean()), "def_ppd": float(dd["pts"].mean()),
            "off_rz_td": float((rz_o["result"] == "TD").mean()) if len(rz_o) else None,
            "def_rz_td": float((rz_d["result"] == "TD").mean()) if len(rz_d) else None,
            "zones_off": {str(k): (None if pd.isna(v) else float(v)) for k, v in od.groupby("zone", observed=False)["pts"].mean().items()},
            "zones_def": {str(k): (None if pd.isna(v) else float(v)) for k, v in dd.groupby("zone", observed=False)["pts"].mean().items()},
            "q_for": [float(qq.loc[i, "pf"]) if qq is not None and i in qq.index else 0 for i in (1, 2, 3, 4)],
            "q_against": [float(qq.loc[i, "pa"]) if qq is not None and i in qq.index else 0 for i in (1, 2, 3, 4)],
        })
    return out


def upcoming(games, ratings, fbs_only=True, n_weeks=1):
    fut = games[games["home_pts"].isna() & (games["season"] == games["season"].max())].sort_values("date")
    fut = fut[fut["date"] >= games.loc[games["home_pts"].notna(), "date"].max()]
    if fbs_only:
        fut = fut[(fut["home_div"] == "fbs") | (fut["away_div"] == "fbs")]
    if fut.empty:
        return []
    wk = sorted(fut["week"].unique())[:n_weeks]
    fut = fut[fut["week"].isin(wk)]
    res = []
    for _, g in fut.iterrows():
        if g["home"] not in ratings.off or g["away"] not in ratings.off:
            continue
        ph, pa = ratings.predict(g["home"], g["away"], bool(g["neutral"]))
        res.append({"date": g["date"].strftime("%Y-%m-%d"), "week": int(g["week"]), "home": g["home"],
                    "away": g["away"], "neutral": bool(g["neutral"]), "pred_home": ph, "pred_away": pa,
                    "win_home": win_prob(ph - pa, SIGMA),
                    "line_home": None if pd.isna(g.get("line_home")) else float(g["line_home"]),
                    "total_line": None})
    return res

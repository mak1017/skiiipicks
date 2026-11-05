"""NBA: opponent-adjusted efficiency (points per 100 possessions) x pace.

Projected points = (league eff + team offense - opponent defense + home edge)
                   * projected possessions / 100
"""
import ast
import re

import numpy as np
import pandas as pd

from .core import fit_ratings, walk_forward, backtest_summary, win_prob

REL = "https://github.com/sportsdataverse/sportsdataverse-data/releases/download"
BOX_URL = REL + "/espn_nba_team_boxscores/team_box_{}.parquet"
SCHED_URL = REL + "/espn_nba_schedules/nba_schedule_{}.parquet"

EFF_PARAMS = dict(alpha=3.0, half_life_days=60, prior_weight=0.3)
PACE_PARAMS = dict(alpha=3.0, half_life_days=60, prior_weight=0.3)
SIGMA = 12.5
NBA_TEAMS = {'ATL', 'BKN', 'BOS', 'CHA', 'CHI', 'CLE', 'DAL', 'DEN', 'DET', 'GS', 'HOU', 'IND', 'LAC', 'LAL',
             'MEM', 'MIA', 'MIL', 'MIN', 'NO', 'NY', 'OKC', 'ORL', 'PHI', 'PHX', 'POR', 'SA', 'SAC', 'TOR',
             'UTAH', 'WSH'}


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def load_box(paths: dict):
    out = []
    for season, path in paths.items():
        b = pd.read_parquet(path)
        b = b[b["season_type"].isin([2, 3]) & b["team_abbreviation"].isin(NBA_TEAMS)].copy()
        b["season"] = season
        out.append(b)
    b = pd.concat(out, ignore_index=True)
    for c in ["team_score", "opponent_team_score", "field_goals_attempted", "offensive_rebounds",
              "total_turnovers", "free_throws_attempted", "points_in_paint", "fast_break_points",
              "three_point_field_goals_made", "three_point_field_goals_attempted", "free_throws_made",
              "field_goals_made", "defensive_rebounds", "turnover_points"]:
        b[c] = _num(b[c])
    b["tov"] = b["total_turnovers"].fillna(_num(b["turnovers"]))
    b["poss"] = b["field_goals_attempted"] - b["offensive_rebounds"] + b["tov"] + 0.44 * b["free_throws_attempted"]
    b["date"] = pd.to_datetime(b["game_date"])
    # pair each team row with its opponent row
    opp = b[["game_id", "team_abbreviation", "poss", "offensive_rebounds", "defensive_rebounds"]].rename(
        columns={"team_abbreviation": "opp", "poss": "opp_poss", "offensive_rebounds": "opp_oreb",
                 "defensive_rebounds": "opp_dreb"})
    b = b.merge(opp, on="game_id")
    b = b[b["opp"] != b["team_abbreviation"]]
    b["game_poss"] = (b["poss"] + b["opp_poss"]) / 2
    b["eff"] = 100 * b["team_score"] / b["game_poss"]
    b["loc"] = np.where(b["team_home_away"] == "home", 0.5, -0.5)
    return b.rename(columns={"team_abbreviation": "team"})


def games_from_box(b):
    h = b[b["team_home_away"] == "home"]
    return pd.DataFrame({"game_id": h["game_id"].values, "season": h["season"].values, "date": h["date"].values,
                         "home": h["team"].values, "away": h["opp"].values,
                         "home_pts": h["team_score"].values, "away_pts": h["opponent_team_score"].values,
                         "neutral": False}).sort_values("date").reset_index(drop=True)


class NBAModel:
    def __init__(self, eff, pace):
        self.eff, self.pace = eff, pace
        self.off, self.dfn, self.home = eff.off, eff.dfn, eff.home

    def proj_pace(self, a, b):
        p1 = self.pace.mu + self.pace.off.get(a, 0) - self.pace.dfn.get(b, 0)
        p2 = self.pace.mu + self.pace.off.get(b, 0) - self.pace.dfn.get(a, 0)
        return (p1 + p2) / 2

    def predict(self, home, away, neutral=False):
        e_h, e_a = self.eff.predict(home, away, neutral)
        pace = self.proj_pace(home, away)
        return e_h * pace / 100, e_a * pace / 100


def make_fit(box, cur_season):
    er = pd.DataFrame({"date": box["date"], "team": box["team"], "opp": box["opp"], "y": box["eff"],
                       "loc": box["loc"], "season": box["season"]})
    pr = pd.DataFrame({"date": box["date"], "team": box["team"], "opp": box["opp"], "y": box["game_poss"],
                       "loc": 0.0, "season": box["season"]})

    def fit(as_of):
        return NBAModel(fit_ratings(er, as_of, cur_season, **EFF_PARAMS),
                        fit_ratings(pr, as_of, cur_season, **PACE_PARAMS))
    return fit


def backtest(box, season):
    """Walk-forward test of one season. Returns (empty frame, None) when it's too early in the season."""
    g = games_from_box(box)
    start = g.loc[g["season"] == season, "date"].min() + pd.Timedelta(days=14)
    if pd.isna(start) or (g.loc[g["season"] == season, "date"] >= start).sum() < 30:
        return pd.DataFrame(columns=list(g.columns) + ["pred_home", "pred_away"]), None
    bt = walk_forward(g, make_fit(box, season), start)
    bt = bt[bt["season"] == season]
    return bt, backtest_summary(bt, SIGMA)


def quarter_points(sched_path):
    s = pd.read_parquet(sched_path)
    s = s[s["season_type"].isin([2, 3]) & s["home_linescores"].notna()]
    rows = []

    def parse(x):
        if isinstance(x, str):  # numpy-style repr: dicts separated by newlines, no commas
            pairs = re.findall(r"'period': (\d+), 'value': ([\d.]+)", x)
            x = [{"period": a, "value": b} for a, b in pairs]
        if x is None:
            return None
        try:
            vals = {int(d["period"]): float(d["value"]) for d in list(x)}
        except Exception:
            return None
        return [vals.get(i, 0) for i in (1, 2, 3, 4)]

    for _, g in s.iterrows():
        h, a = parse(g["home_linescores"]), parse(g["away_linescores"])
        if not h or not a:
            continue
        for i in range(4):
            rows.append((g["home_abbreviation"], i + 1, h[i], a[i]))
            rows.append((g["away_abbreviation"], i + 1, a[i], h[i]))
    q = pd.DataFrame(rows, columns=["team", "qtr", "pf", "pa"])
    return q.groupby(["team", "qtr"])[["pf", "pa"]].mean()


def team_profiles(box, model, season, qp):
    cur = box[(box["season"] == season)]
    opp_side = cur[["game_id", "team", "team_score", "points_in_paint", "fast_break_points",
                    "three_point_field_goals_made", "three_point_field_goals_attempted", "free_throws_made",
                    "field_goals_made", "field_goals_attempted", "tov", "free_throws_attempted",
                    "offensive_rebounds", "poss"]].add_prefix("o_").rename(columns={"o_game_id": "game_id"})
    c = cur.merge(opp_side, on="game_id")
    c = c[c["o_team"] != c["team"]]

    def four(x, p=""):
        fga, fgm, tpm = x[p + "field_goals_attempted"].sum(), x[p + "field_goals_made"].sum(), x[p + "three_point_field_goals_made"].sum()
        return {"efg": (fgm + 0.5 * tpm) / fga, "tov": x[p + "tov"].sum() / x[p + "poss"].sum(),
                "ftr": x[p + "free_throws_attempted"].sum() / fga}

    out = []
    for t, x in c.groupby("team"):
        pts, opts = x["team_score"].sum(), x["o_team_score"].sum()
        mix_o = {"Paint": x["points_in_paint"].sum() / pts, "3-pointers": 3 * x["three_point_field_goals_made"].sum() / pts,
                 "Free throws": x["free_throws_made"].sum() / pts}
        mix_o["Midrange/other"] = max(0.0, 1 - sum(mix_o.values()))
        mix_d = {"Paint": x["o_points_in_paint"].sum() / opts, "3-pointers": 3 * x["o_three_point_field_goals_made"].sum() / opts,
                 "Free throws": x["o_free_throws_made"].sum() / opts}
        mix_d["Midrange/other"] = max(0.0, 1 - sum(mix_d.values()))
        f_o, f_d = four(x), four(x, "o_")
        oreb = x["offensive_rebounds"].sum() / (x["offensive_rebounds"].sum() + x["opp_dreb"].sum())
        dreb = x["defensive_rebounds"].sum() / (x["defensive_rebounds"].sum() + x["opp_oreb"].sum())
        q = qp.loc[t] if t in qp.index.get_level_values(0) else None
        reg = x[x["season_type"] == 2]
        out.append({
            "team": t, "gp": int(len(x)), "record": f"{int((reg['team_score'] > reg['o_team_score']).sum())}-{int((reg['team_score'] < reg['o_team_score']).sum())}",
            "pf": float(x["team_score"].mean()), "pa": float(x["o_team_score"].mean()),
            "ortg": float(100 * pts / x["game_poss"].sum()), "drtg": float(100 * opts / x["game_poss"].sum()),
            "off_rating": model.eff.off.get(t, 0), "def_rating": model.eff.dfn.get(t, 0),
            "net_rating": model.eff.off.get(t, 0) + model.eff.dfn.get(t, 0),
            "pace": float(x["game_poss"].mean()), "proj_pace": model.pace.mu + model.pace.off.get(t, 0) - model.pace.dfn.get(t, 0),
            "off_efg": f_o["efg"], "def_efg": f_d["efg"], "off_tov": f_o["tov"], "def_tov": f_d["tov"],
            "off_ftr": f_o["ftr"], "def_ftr": f_d["ftr"], "oreb_pct": oreb, "dreb_pct": dreb,
            "off_3pa_rate": float(x["three_point_field_goals_attempted"].sum() / x["field_goals_attempted"].sum()),
            "fastbreak_pg": float(x["fast_break_points"].mean()), "opp_fastbreak_pg": float(x["o_fast_break_points"].mean()),
            "mix_off": mix_o, "mix_def": mix_d,
            "q_for": [float(q.loc[i, "pf"]) if q is not None and i in q.index else 0 for i in (1, 2, 3, 4)],
            "q_against": [float(q.loc[i, "pa"]) if q is not None and i in q.index else 0 for i in (1, 2, 3, 4)],
        })
    return out


def upcoming(sched_path, model, days=8):
    """Next `days` of scheduled games, with back-to-back alerts (worth about 2.5 points in testing)."""
    from .stack import tier
    s = pd.read_parquet(sched_path)
    s = s[s["season_type"].isin([2, 3]) & s["home_abbreviation"].isin(NBA_TEAMS) & s["away_abbreviation"].isin(NBA_TEAMS)]
    s = s.assign(date=pd.to_datetime(s["game_date"])).sort_values("date")
    played = {}
    for _, g in s.iterrows():
        for t in (g["home_abbreviation"], g["away_abbreviation"]):
            played.setdefault(t, set()).add(g["date"])
    fut = s[~s["status_type_completed"].astype(bool)]
    if fut.empty:
        return []
    first = max(fut["date"].min(), pd.Timestamp.today().normalize())
    fut = fut[(fut["date"] >= first) & (fut["date"] < first + pd.Timedelta(days=days))]
    out = []
    for _, g in fut.iterrows():
        h, a = g["home_abbreviation"], g["away_abbreviation"]
        ph, pa = model.predict(h, a, bool(g.get("neutral_site", False)))
        notes = [f"{t} on the second night of a back-to-back" for t in (h, a)
                 if g["date"] - pd.Timedelta(days=1) in played.get(t, set())]
        wp = win_prob(ph - pa, SIGMA)
        out.append({"date": g["date"].strftime("%Y-%m-%d"), "home": h, "away": a, "neutral": bool(g.get("neutral_site", False)),
                    "pred_home": ph, "pred_away": pa, "win_home": wp, "tier": tier(wp), "line_home": None,
                    "total_line": None, "cover_home": None, "notes": notes})
    return out


# ---------------- lineup strength (who is actually playing) ----------------
# Player value = recent game score (Hollinger), decayed. A team's lineup strength for a game is the sum of
# its active players' values; the stage-2 feature is how far that sits from the team's own recent norm,
# because the ratings already reflect whoever has been playing lately. Worth about 1 point of winners
# in walk-forward testing (68.5% -> 70.0% on 2025-26).

VAL_HL, BASE_HL = 15, 20


def _player_games(pbox_paths):
    pb = pd.concat([pd.read_parquet(p) for p in pbox_paths], ignore_index=True)
    pb = pb[pb["season_type"].isin([2, 3]) & pb["team_abbreviation"].isin(NBA_TEAMS)].copy()
    pb["date"] = pd.to_datetime(pb["game_date"])
    n = lambda c: _num(pb[c]).fillna(0)
    pb["min"] = n("minutes")
    pb["gs"] = (n("points") + .4 * n("field_goals_made") - .7 * n("field_goals_attempted")
                - .4 * (n("free_throws_attempted") - n("free_throws_made")) + .7 * n("offensive_rebounds")
                + .3 * n("defensive_rebounds") + n("steals") + .7 * n("assists") + .7 * n("blocks")
                - .4 * n("fouls") - n("turnovers"))
    pb["played"] = (pb["min"] > 0) & ~pb["did_not_play"].fillna(False).astype(bool)
    pb["game_id"] = pb["game_id"].astype(str)
    pb = pb.sort_values(["athlete_id", "date"])
    g = pb.groupby("athlete_id")
    pb["gs_p"] = pb["gs"].where(pb["played"])
    pb["val_pre"] = g["gs_p"].transform(lambda s: s.shift().ewm(halflife=VAL_HL, ignore_na=True).mean())
    pb["val_post"] = g["gs_p"].transform(lambda s: s.ewm(halflife=VAL_HL, ignore_na=True).mean())
    pb["n_pre"] = g["played"].cumsum() - pb["played"]
    pb["v"] = pb["val_pre"].fillna(0).clip(lower=0) * (pb["n_pre"] >= 3)
    return pb


def lineup_table(pbox_paths):
    """Per (game_id, team): lineup strength vs the team's recent norm, using only earlier games for the norm."""
    pb = _player_games(pbox_paths)
    t = pb.assign(hv=np.where(pb["played"], pb["v"], 0)).groupby(["game_id", "team_abbreviation"]).agg(
        hv=("hv", "sum"), date=("date", "first")).reset_index().sort_values("date")
    t["base"] = t.groupby("team_abbreviation")["hv"].transform(lambda s: s.shift().ewm(halflife=BASE_HL).mean())
    t["shift"] = (t["hv"] - t["base"]).fillna(0)
    return t.rename(columns={"team_abbreviation": "team"}), pb


def add_lineup(df, table):
    """df has game_id, home, away. Adds lineup_diff (home minus away)."""
    t = table[["game_id", "team", "shift"]]
    d = df.copy(); d["game_id"] = d["game_id"].astype(str)
    d = d.merge(t.rename(columns={"team": "home", "shift": "h_shift"}), on=["game_id", "home"], how="left")
    d = d.merge(t.rename(columns={"team": "away", "shift": "a_shift"}), on=["game_id", "away"], how="left")
    d["lineup_diff"] = d["h_shift"].fillna(0) - d["a_shift"].fillna(0)
    return d


def lineup_upcoming(table, pb, ups, inj, today=None):
    """Expected lineup strength for upcoming games: players who played for the team in the last 30 days,
    minus anyone listed Out/Doubtful/Suspended. Adds lineup_diff and a readable note per game."""
    from .injuries import match, RULED_OUT
    today = pd.Timestamp(today or pd.Timestamp.today().normalize())
    last = pb[pb["played"]].sort_values("date").groupby("athlete_id").tail(1)
    last = last[last["date"] >= today - pd.Timedelta(days=30)]
    pv = pb[pb["played"]].groupby("athlete_id")["val_post"].last()
    npl = pb.groupby("athlete_id")["played"].sum()
    base = table.sort_values("date").groupby("team").apply(
        lambda x: x["hv"].ewm(halflife=BASE_HL).mean().iloc[-1])
    out_names = {}
    for u in ups:
        sh = {}
        for side in ("home", "away"):
            team = u[side]
            roster = last[last["team_abbreviation"] == team]
            tot, missing = 0.0, []
            for _, r in roster.iterrows():
                v = max(float(pv.get(r["athlete_id"], 0) or 0), 0) if npl.get(r["athlete_id"], 0) >= 3 else 0
                row = match(r["athlete_display_name"], team, inj) if inj is not None and len(inj) else None
                if row is not None and row["status"] in RULED_OUT:
                    if v >= 8: missing.append(r["athlete_display_name"])
                    continue
                tot += v
            sh[side] = tot - float(base.get(team, tot))
            out_names[(u["date"], team)] = missing
        u["lineup_diff"] = sh["home"] - sh["away"]
        u["lineup_shift"] = {"home": sh["home"], "away": sh["away"]}
        u["lineup_out"] = {s_: out_names.get((u["date"], u[s_]), []) for s_ in ("home", "away")}
    return ups

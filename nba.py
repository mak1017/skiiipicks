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
    g = games_from_box(box)
    start = g.loc[g["season"] == season, "date"].min() + pd.Timedelta(days=14)
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


# ---------------- injuries: what a missing player costs ----------------
# Each player's value is his Game Score per minute above a bench-level rate, times his usual minutes.
# A missing player counts in proportion to how many of his team's last 10 games he played, because the
# team ratings already reflect games he sat out. Points per unit of missing value are fitted on past games.

AVAIL_GAMES = 10       # team games used to judge how often a player has been playing
REPLACEMENT = 0.25     # Game Score per minute of a deep-bench player
ROTATION_MIN = 15      # usual minutes needed to count as a rotation player


def player_games(pbox_paths: dict):
    """One row per player per team game, including players who didn't play (minutes 0)."""
    fr = []
    for season, path in pbox_paths.items():
        b = pd.read_parquet(path)
        b = b[b["season_type"].isin([2, 3]) & b["team_abbreviation"].isin(NBA_TEAMS)].copy()
        b["season"] = season
        fr.append(b)
    b = pd.concat(fr, ignore_index=True)
    n = {c: _num(b[c]).fillna(0) for c in ["minutes", "points", "field_goals_made", "field_goals_attempted",
                                           "free_throws_made", "free_throws_attempted", "offensive_rebounds",
                                           "defensive_rebounds", "steals", "assists", "blocks", "fouls", "turnovers"]}
    gmsc = (n["points"] + 0.4 * n["field_goals_made"] - 0.7 * n["field_goals_attempted"]
            - 0.4 * (n["free_throws_attempted"] - n["free_throws_made"]) + 0.7 * n["offensive_rebounds"]
            + 0.3 * n["defensive_rebounds"] + n["steals"] + 0.7 * n["assists"] + 0.7 * n["blocks"]
            - 0.4 * n["fouls"] - n["turnovers"])
    return pd.DataFrame({"game_id": b["game_id"], "season": b["season"], "date": pd.to_datetime(b["game_date"]),
                         "pid": b["athlete_id"], "name": b["athlete_display_name"], "team": b["team_abbreviation"],
                         "minutes": n["minutes"], "gmsc": gmsc}).drop_duplicates(["game_id", "pid"])


class _Tracker:
    """Walks through games in date order, keeping each player's recent value and each team's recent games."""

    def __init__(self, hl_rate=15, hl_min=6):
        self.dr, self.dm = 0.5 ** (1 / hl_rate), 0.5 ** (1 / hl_min)
        self.p = {}       # pid -> [gmsc sum, minutes sum (rate), minutes sum, weight, team, name]
        self.team = {}    # team -> list of recent game rosters (sets of pids who played)

    def players(self, team):
        """Rotation players currently on `team`: {pid: (name, value, availability)}."""
        recent = self.team.get(team, [])[-AVAIL_GAMES:]
        out = {}
        for pid, (g, mr, m, w, t, name) in self.p.items():
            if t != team or w == 0:
                continue
            mins = m / w
            if mins < ROTATION_MIN:
                continue
            rate = g / mr if mr > 0 else 0
            avail = sum(pid in r for r in recent) / len(recent) if recent else 0.0
            out[pid] = (name, max(rate - REPLACEMENT, 0) * mins, avail)
        return out

    def update(self, day):
        """day: player rows for one date."""
        for team, x in day.groupby("team"):
            self.team.setdefault(team, []).append(set(x.loc[x["minutes"] > 0, "pid"]))
        for r in day[day["minutes"] > 0].itertuples():
            s = self.p.get(r.pid, [0.0, 0.0, 0.0, 0.0, None, r.name])
            s[0] = s[0] * self.dr + r.gmsc
            s[1] = s[1] * self.dr + r.minutes
            s[2] = s[2] * self.dm + r.minutes
            s[3] = s[3] * self.dm + 1
            s[4], s[5] = r.team, r.name
            self.p[r.pid] = s


def missing_value(pg):
    """Walk-forward: for every team game, the value of rotation players who didn't play (known only from
    earlier games). Returns game_id, team, v_out, who."""
    tr, rows = _Tracker(), []
    for d, day in pg.sort_values("date").groupby("date"):
        for (gid, team), x in day.groupby(["game_id", "team"]):
            played = set(x.loc[x["minutes"] > 0, "pid"])
            out = [(name, val * av) for pid, (name, val, av) in tr.players(team).items() if pid not in played]
            rows.append({"game_id": gid, "team": team, "v_out": sum(v for _, v in out),
                         "who": ", ".join(n for n, v in sorted(out, key=lambda z: -z[1]) if v >= 2)})
        tr.update(day)
    return pd.DataFrame(rows)


def current_players(pg):
    """Everyone's value and availability as of the latest game in the data."""
    tr = _Tracker()
    for d, day in pg.sort_values("date").groupby("date"):
        tr.update(day)
    return tr


def _side_values(bt, mv):
    m = dict(zip(zip(mv["game_id"], mv["team"]), mv["v_out"]))  # later rows win on duplicates
    vh = np.array([m.get((g, t), 0.0) for g, t in zip(bt["game_id"], bt["home"])])
    va = np.array([m.get((g, t), 0.0) for g, t in zip(bt["game_id"], bt["away"])])
    return vh, va


def fit_injury_effect(bt, mv):
    """Least squares on backtest residuals: points a team scores (own) and allows (opp) per unit of its
    missing value. bt: walk-forward rows with game_id, home, away, pred/actual points."""
    vh, va = _side_values(bt, mv)
    # stack the two offenses: points scored residual ~ own missing value + opponent missing value
    X = np.r_[np.c_[vh, va], np.c_[va, vh]]
    y = np.r_[bt["home_pts"] - bt["pred_home"], bt["away_pts"] - bt["pred_away"]]
    y = y - y.mean()
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"own": float(coef[0]), "opp": float(coef[1])}


def apply_injury_effect(bt, mv, eff):
    vh, va = _side_values(bt, mv)
    bt = bt.copy()
    bt["pred_home"] = bt["pred_home"] + eff["own"] * vh + eff["opp"] * va
    bt["pred_away"] = bt["pred_away"] + eff["own"] * va + eff["opp"] * vh
    return bt


def injury_adjust(ups, inj, tracker, eff, ruled_out, norm):
    """Move upcoming projections for rotation players listed as ruled out. eff from fit_injury_effect."""
    from .stack import tier
    for u in ups:
        lost, v = {}, {}
        for side in ("home", "away"):
            t = inj[(inj["team"] == u[side]) & inj["status"].isin(ruled_out)]
            names = set(t["name"].map(norm))
            hit = sorted(((name, val * av) for name, val, av in tracker.players(u[side]).values()
                          if norm(name) in names and val * av >= 0.5), key=lambda z: -z[1])
            lost[side], v[side] = [n for n, _ in hit], sum(x for _, x in hit)
        dh = eff["own"] * v["home"] + eff["opp"] * v["away"]
        da = eff["own"] * v["away"] + eff["opp"] * v["home"]
        if not (lost["home"] or lost["away"]):
            continue
        u["pred_home"] += dh
        u["pred_away"] += da
        u["win_home"] = win_prob(u["pred_home"] - u["pred_away"], SIGMA)
        u["tier"] = tier(u["win_home"])
        u["injury_adj"] = {"home": dh, "away": da, "out": lost}
        for side in ("home", "away"):
            if lost[side]:  # this side's own absences only: it scores less and allows more
                d = (eff["own"] - eff["opp"]) * v[side]
                u.setdefault("notes", []).append(
                    f"Projection accounts for {u[side]} without {', '.join(lost[side])} ({d:+.1f} pts for {u[side]})")
    return ups

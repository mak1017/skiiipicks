"""Player props: projection + spread for each stat, so any sportsbook line becomes an over/under chance.

projection = player's recent per-game form (exponentially weighted, last season counts less)
             x opponent factor (how much that defense allows of this stat vs. league average, shrunk
               toward 1.0 when the defense has played few games)
NBA projects per-minute rates x projected minutes.

Spreads are fitted from walk-forward backtests (see backtest_*), not assumed.
"""
import numpy as np
import pandas as pd

NFL_STATS = {  # stat: (opponent category, kind)
    "pass_yds": ("pass_yds", "cont"), "pass_td": ("pass_td", "count"), "rush_yds": ("rush_yds", "cont"),
    "rec_yds": ("rec_yds", "cont"), "receptions": ("receptions", "count"), "any_td": ("any_td", "td"),
}
NFL_LABELS = {"pass_yds": "Passing yards", "pass_td": "Passing TDs", "rush_yds": "Rushing yards",
              "rec_yds": "Receiving yards", "receptions": "Receptions", "any_td": "Anytime TD"}
NBA_STATS = {"points": "cont", "rebounds": "count", "assists": "count", "threes": "count", "pra": "cont"}
NBA_LABELS = {"points": "Points", "rebounds": "Rebounds", "assists": "Assists", "threes": "3-pointers made",
              "pra": "Pts + Reb + Ast"}


# ---------------- NFL ----------------

def nfl_player_games(pbp):
    p = pbp[pbp["play_type"].isin(["pass", "run"]) & (pbp["two_point_attempt"] != 1)]
    base = ["game_id", "season", "week", "game_date", "posteam", "defteam"]
    ps = p[p["passer_player_id"].notna() & (p["sack"] != 1)].groupby(base + ["passer_player_id"]).agg(
        name=("passer_player_name", "first"), pass_att=("pass_attempt", "sum"),
        pass_yds=("passing_yards", "sum"), pass_td=("pass_touchdown", "sum")).reset_index().rename(columns={"passer_player_id": "pid"})
    ru = p[p["rusher_player_id"].notna()].groupby(base + ["rusher_player_id"]).agg(
        name=("rusher_player_name", "first"), rush_att=("rush_attempt", "sum"),
        rush_yds=("rushing_yards", "sum"), rush_td=("rush_touchdown", "sum")).reset_index().rename(columns={"rusher_player_id": "pid"})
    rc = p[p["receiver_player_id"].notna()].groupby(base + ["receiver_player_id"]).agg(
        name=("receiver_player_name", "first"), targets=("pass_attempt", "sum"), receptions=("complete_pass", "sum"),
        rec_yds=("receiving_yards", "sum"), rec_td=("pass_touchdown", "sum")).reset_index().rename(columns={"receiver_player_id": "pid"})
    pg = ps.merge(ru, on=base + ["pid"], how="outer", suffixes=("", "_r")).merge(rc, on=base + ["pid"], how="outer", suffixes=("", "_c"))
    pg["name"] = pg["name"].fillna(pg.get("name_r")).fillna(pg.get("name_c"))
    pg = pg.drop(columns=[c for c in pg.columns if c.startswith("name_")])
    num = ["pass_att", "pass_yds", "pass_td", "rush_att", "rush_yds", "rush_td", "targets", "receptions", "rec_yds", "rec_td"]
    pg[num] = pg[num].fillna(0)
    pg["any_td"] = pg["rush_td"] + pg["rec_td"]
    pg["date"] = pd.to_datetime(pg["game_date"])
    pg = pg.rename(columns={"posteam": "team", "defteam": "opp"})
    return pg


def _ewma(hist, cols, key, hl, cur_season, prior_w, last_n):
    h = hist.sort_values("date", ascending=False)
    h = h.assign(rank=h.groupby(key).cumcount())
    h = h[h["rank"] < last_n]
    w = np.power(0.5, h["rank"] / hl) * np.where(h["season"] < cur_season, prior_w, 1.0)
    h = h.assign(_w=w)
    agg = {c: (h[c] * h["_w"]).groupby(h[key]).sum() / h["_w"].groupby(h[key]).sum() for c in cols}
    out = pd.DataFrame(agg)
    out["n_cur"] = h[h["season"] == cur_season].groupby(key).size()
    out["n_cur"] = out["n_cur"].fillna(0)
    return out


def _opp_factors(hist, cats, cur_season, k):
    """Per-defense allowed per game for each category, relative to league average, shrunk toward 1."""
    tg = hist.groupby(["game_id", "opp", "season"])[cats].sum().reset_index()
    tg["w"] = np.where(tg["season"] < cur_season, 0.3, 1.0)
    out = {}
    for c in cats:
        allowed = (tg[c] * tg["w"]).groupby(tg["opp"]).sum() / tg["w"].groupby(tg["opp"]).sum()
        n = tg["w"].groupby(tg["opp"]).sum()
        lg = (tg[c] * tg["w"]).sum() / tg["w"].sum()
        raw = allowed / lg if lg > 0 else allowed * 0 + 1
        out[c] = 1 + (raw - 1) * n / (n + k)
    return pd.DataFrame(out)


def nfl_project(pg, as_of, cur_season):
    hist = pg[pg["date"] < as_of]
    cols = ["pass_att", "rush_att", "targets"] + list(NFL_STATS)
    form = _ewma(hist, cols, "pid", hl=4, cur_season=cur_season, prior_w=0.5, last_n=12)
    last = hist.sort_values("date").groupby("pid").tail(1).set_index("pid")
    form["name"], form["team"], form["last_date"] = last["name"], last["team"], last["date"]
    team_last = hist.groupby("team")["date"].max()
    form["active"] = form["last_date"].values == team_last.reindex(form["team"]).values
    opp = _opp_factors(hist[hist["season"] >= cur_season - 1], [v[0] for v in NFL_STATS.values()], cur_season, k=4)
    return form, opp


def nfl_role(r):
    if r["pass_att"] >= 15:
        return "QB"
    if r["rush_att"] >= 6:
        return "RB"
    if r["targets"] >= 3:
        return "WR/TE"
    return None


ROLE_STATS = {"QB": ["pass_yds", "pass_td", "rush_yds"], "RB": ["rush_yds", "receptions", "rec_yds", "any_td"],
              "WR/TE": ["receptions", "rec_yds", "any_td"]}
ROLE_SLOTS = {"QB": 1, "RB": 2, "WR/TE": 4}


def nfl_game_props(form, opp, team, opp_team, spreads):
    f = form[(form["team"] == team) & form["active"] & (form["n_cur"] >= 1)].copy()
    f["role"] = f.apply(nfl_role, axis=1)
    out = []
    for role, n in ROLE_SLOTS.items():
        key = {"QB": "pass_att", "RB": "rush_att", "WR/TE": "targets"}[role]
        for pid, r in f[f["role"] == role].sort_values(key, ascending=False).head(n).iterrows():
            props = {}
            for st in ROLE_STATS[role]:
                if role == "QB" and st == "rush_yds" and r["rush_yds"] < 12:
                    continue
                cat, kind = NFL_STATS[st]
                fac = float(opp[cat].get(opp_team, 1.0)) if cat in opp else 1.0
                proj = float(r[st]) * fac
                props[st] = {"proj": proj, "kind": kind, **spread_params(spreads, st, kind, proj)}
            out.append({"name": r["name"], "role": role, "props": props})
    return out


# ---------------- NBA ----------------

def nba_player_games(pbox_paths: dict):
    fr = []
    for season, path in pbox_paths.items():
        b = pd.read_parquet(path)
        b = b[b["season_type"].isin([2, 3])].copy()
        b["season"] = season
        fr.append(b)
    b = pd.concat(fr, ignore_index=True)
    b = b[~b["did_not_play"].fillna(False).astype(bool)]
    for c in ["minutes", "points", "rebounds", "assists", "three_point_field_goals_made"]:
        b[c] = pd.to_numeric(b[c], errors="coerce")
    b = b[b["minutes"] > 0]
    return pd.DataFrame({
        "pid": b["athlete_id"], "name": b["athlete_display_name"], "team": b["team_abbreviation"],
        "opp": b["opponent_team_abbreviation"], "game_id": b["game_id"], "season": b["season"],
        "date": pd.to_datetime(b["game_date"]), "minutes": b["minutes"], "points": b["points"],
        "rebounds": b["rebounds"], "assists": b["assists"], "threes": b["three_point_field_goals_made"],
        "pra": b["points"] + b["rebounds"] + b["assists"], "starter": b["starter"].fillna(False).astype(bool),
    })


def nba_project(pg, as_of, cur_season):
    hist = pg[pg["date"] < as_of].copy()
    stats = list(NBA_STATS)
    for s in stats:
        hist[s + "_pm"] = hist[s] / hist["minutes"]
    rates = _ewma(hist, [s + "_pm" for s in stats], "pid", hl=10, cur_season=cur_season, prior_w=0.5, last_n=25)
    mins = _ewma(hist, ["minutes"], "pid", hl=4, cur_season=cur_season, prior_w=0.3, last_n=10)
    form = rates.join(mins[["minutes"]])
    last = hist.sort_values("date").groupby("pid").tail(1).set_index("pid")
    form["name"], form["team"], form["last_date"] = last["name"], last["team"], last["date"]
    team_last = hist.groupby("team")["date"].max()
    form["active"] = form["last_date"].values == team_last.reindex(form["team"]).values
    opp = _opp_factors(hist[hist["season"] >= cur_season - 1], stats, cur_season, k=10)
    return form, opp


def nba_game_props(form, opp, team, opp_team, spreads, min_minutes=18):
    f = form[(form["team"] == team) & form["active"] & (form["minutes"] >= min_minutes) & (form["n_cur"] >= 3)]
    out = []
    for pid, r in f.sort_values("minutes", ascending=False).head(8).iterrows():
        props = {}
        for st, kind in NBA_STATS.items():
            fac = float(opp[st].get(opp_team, 1.0))
            proj = float(r[st + "_pm"] * r["minutes"]) * fac
            props[st] = {"proj": proj, "kind": kind, **spread_params(spreads, st, kind, proj)}
        out.append({"name": r["name"], "role": f"{r['minutes']:.0f} min", "props": props})
    return out


# ---------------- spreads and backtests ----------------

def spread_params(spreads, st, kind, proj):
    s = spreads.get(st, {})
    if kind == "cont":
        return {"sd": max(s.get("a", 0) + s.get("b", 0.5) * proj, 1.0)}
    if kind == "count":
        return {"phi": s.get("phi", 1.2)}
    return {}


def fit_spreads(bt, kinds):
    """bt rows: stat, proj, actual. Continuous: SD = a + b*proj. Counts: variance = phi * proj."""
    out = {}
    for st, kind in kinds.items():
        d = bt[(bt["stat"] == st) & (bt["proj"] > 0)]
        if len(d) < 50:
            continue
        if kind == "cont":
            sd = np.abs(d["actual"] - d["proj"]) * 1.2533  # mean abs dev -> SD for a normal
            b, a = np.polyfit(d["proj"], sd, 1)
            if b < 0:  # spread doesn't grow with the projection: use a flat SD
                a, b = float(sd.mean()), 0.0
            out[st] = {"a": float(a), "b": float(b)}
        elif kind == "count":
            out[st] = {"phi": float(max(np.mean((d["actual"] - d["proj"]) ** 2) / d["proj"].mean(), 0.8))}
    return out


def _norm_cdf(x):
    from math import erf, sqrt
    return 0.5 * (1 + erf(x / sqrt(2)))


def p_over(proj, kind, line, sd=None, phi=None):
    if kind == "td":
        return 1 - np.exp(-proj) if line < 1 else None
    if kind == "count":
        sd = np.sqrt(max(phi * proj, 0.05))
    return 1 - _norm_cdf((line - proj) / sd)


def backtest(pg, project_fn, kinds, cur_season, cuts, row_filter):
    """Walk-forward: project before each cut, compare with games in [cut, next cut)."""
    rows = []
    for i, cut in enumerate(cuts[:-1]):
        form, opp = project_fn(pg, cut, cur_season)
        nxt = pg[(pg["date"] >= cut) & (pg["date"] < cuts[i + 1]) & (pg["season"] == cur_season)]
        cur = pg[(pg["date"] < cut) & (pg["season"] == cur_season)]
        naive = cur.groupby("pid")[list(kinds)].mean()
        ncount = cur.groupby("pid").size()
        for _, g in nxt.iterrows():
            if g["pid"] not in form.index or ncount.get(g["pid"], 0) < 3:
                continue
            r = form.loc[g["pid"]]
            for st in row_filter(r, g):
                cat = NFL_STATS[st][0] if st in NFL_STATS else st
                fac = float(opp[cat].get(g["opp"], 1.0)) if cat in opp else 1.0
                base = r[st] if st in r else r[st + "_pm"] * r["minutes"]
                rows.append({"stat": st, "proj": float(base) * fac, "naive": float(naive.loc[g["pid"], st]),
                             "actual": float(g[st])})
    return pd.DataFrame(rows)


def summarize(bt, kinds, spreads, labels):
    out = []
    for st, kind in kinds.items():
        d = bt[bt["stat"] == st]
        if len(d) < 50:
            continue
        r = {"stat": labels[st], "n": int(len(d)), "mae": float(np.abs(d["actual"] - d["proj"]).mean()),
             "naive_mae": float(np.abs(d["actual"] - d["naive"]).mean())}
        # calibration at a book-style line: the player's season average rounded to the nearest .5
        if kind != "td":
            line = np.floor(d["naive"]) + 0.5
            sp = spreads.get(st, {})
            p = [p_over(pj, kind, ln, sd=max(sp.get("a", 0) + sp.get("b", 0.5) * pj, 1), phi=sp.get("phi", 1.2))
                 for pj, ln in zip(d["proj"], line)]
            p = np.array(p)
            hit = (d["actual"] > line).values
            conf = np.abs(p - 0.5) >= 0.1
            r["lean_hit"] = float(((p[conf] > 0.5) == hit[conf]).mean()) if conf.sum() >= 30 else None
            r["lean_n"] = int(conf.sum())
            r["brier"] = float(np.mean((p - hit) ** 2))
        else:
            p = 1 - np.exp(-d["proj"].values)
            hit = (d["actual"] >= 1).values
            r["brier"] = float(np.mean((p - hit) ** 2))
            r["td_pred"] = float(p.mean())
            r["td_actual"] = float(hit.mean())
        out.append(r)
    return out

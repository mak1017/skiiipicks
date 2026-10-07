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


def count_cdf(n, mu, phi):
    """P(X <= n) for a count with mean mu and variance phi*mu: Poisson if phi ~ 1, else negative binomial."""
    if n < 0:
        return 0.0
    mu = max(mu, 1e-6)
    if phi <= 1.05:
        pm, tot = np.exp(-mu), 0.0
        for k in range(int(n) + 1):
            tot += pm
            pm *= mu / (k + 1)
        return min(tot, 1.0)
    r = mu / (phi - 1)
    q = r / (r + mu)
    pm, tot = q ** r, 0.0
    for k in range(int(n) + 1):
        tot += pm
        pm *= (k + r) / (k + 1) * (1 - q)
    return min(tot, 1.0)


def p_over(proj, kind, line, sd=None, phi=None):
    if kind == "td":
        return 1 - np.exp(-proj) if line < 1 else None
    if kind == "count":
        return 1 - count_cdf(np.floor(line), proj, phi or 1.2)
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
                if st in r:
                    base = r[st]
                elif st + "_pp" in r:
                    base = r[st + "_pp"] * r["pa"]
                else:
                    base = r[st + "_pm"] * r["minutes"]
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


# ---------------- college football (same roles and stats as NFL) ----------------

def cfb_player_games(pbp, nmap, season):
    """cfbfastR play-by-play -> one row per player-game, with schedule team names."""
    p = pbp.copy()
    p["team"], p["opp"] = p["pos_team"].map(nmap), p["def_pos_team"].map(nmap)
    p = p[p["team"].notna() & p["opp"].notna()]
    p["date"] = pd.Timestamp(f"{season}-08-29") + pd.to_timedelta((p["week"] - 1) * 7, unit="D")
    base = ["game_id", "week", "date", "team", "opp"]
    passes = p[(p["pass"] == 1) & p["passer_player_name"].notna() & p["yds_sacked"].isna()]
    comp = passes["completion"].fillna(False).astype(bool)
    ps = passes.assign(c=comp, y=np.where(comp, passes["yds_receiving"].fillna(0), 0), td=passes["pass_td"].fillna(0)).groupby(
        base + ["passer_player_name"]).agg(pass_att=("c", "size"), pass_yds=("y", "sum"), pass_td=("td", "sum")).reset_index().rename(
        columns={"passer_player_name": "name"})
    rush = p[(p["rush"] == 1) & p["rusher_player_name"].notna()]
    ru = rush.assign(y=rush["yds_rushed"].fillna(0), td=rush["rush_td"].fillna(0)).groupby(base + ["rusher_player_name"]).agg(
        rush_att=("y", "size"), rush_yds=("y", "sum"), rush_td=("td", "sum")).reset_index().rename(columns={"rusher_player_name": "name"})
    tg = passes[passes["receiver_player_name"].notna()]
    tc = tg["completion"].fillna(False).astype(bool)
    rc = tg.assign(c=tc.astype(int), y=np.where(tc, tg["yds_receiving"].fillna(0), 0), td=tg["pass_td"].fillna(0)).groupby(
        base + ["receiver_player_name"]).agg(targets=("c", "size"), receptions=("c", "sum"), rec_yds=("y", "sum"),
                                             rec_td=("td", "sum")).reset_index().rename(columns={"receiver_player_name": "name"})
    pg = ps.merge(ru, on=base + ["name"], how="outer").merge(rc, on=base + ["name"], how="outer")
    num = ["pass_att", "pass_yds", "pass_td", "rush_att", "rush_yds", "rush_td", "targets", "receptions", "rec_yds", "rec_td"]
    pg[num] = pg[num].fillna(0)
    pg["any_td"] = pg["rush_td"] + pg["rec_td"]
    pg["season"] = season
    pg["pid"] = pg["team"] + "|" + pg["name"]
    return pg


# ---------------- MLB batters ----------------

MLB_STATS = {"hits": "count", "total_bases": "count", "home_runs": "td", "strikeouts": "count", "rbi": "count"}
MLB_LABELS = {"hits": "Hits", "total_bases": "Total bases", "home_runs": "Home run", "strikeouts": "Strikeouts (batter)",
              "rbi": "RBIs"}
PA_EVENTS = {"single", "double", "triple", "home_run", "walk", "intent_walk", "hit_by_pitch", "strikeout",
             "strikeout_double_play", "field_out", "force_out", "grounded_into_double_play", "double_play", "triple_play",
             "fielders_choice", "fielders_choice_out", "field_error", "sac_fly", "sac_bunt", "sac_fly_double_play",
             "sac_bunt_double_play", "catcher_interf", "other_out"}
REGISTER_URL = "https://raw.githubusercontent.com/chadwickbureau/register/master/data/people-{}.csv"


def mlb_names(path_fmt=REGISTER_URL):
    fr = [pd.read_csv(path_fmt.format(h), usecols=["key_mlbam", "name_first", "name_last"], low_memory=False)
          for h in "0123456789abcdef"]
    r = pd.concat(fr).dropna(subset=["key_mlbam"])
    return dict(zip(r["key_mlbam"].astype(int), (r["name_first"].fillna("") + " " + r["name_last"].fillna("")).str.strip()))


def mlb_player_games(pbp, games, names):
    p = pbp[pbp["event_type"].isin(PA_EVENTS)].merge(games[["game_pk", "home", "away", "date", "season"]], on="game_pk")
    top = p["half_inning"] == "top"
    p["team"], p["opp"] = np.where(top, p["away"], p["home"]), np.where(top, p["home"], p["away"])
    ev = p["event_type"]
    tb = ev.map({"single": 1, "double": 2, "triple": 3, "home_run": 4}).fillna(0)
    p = p.assign(pa=1, hits=(tb > 0).astype(int), total_bases=tb, home_runs=(ev == "home_run").astype(int),
                 strikeouts=ev.str.startswith("strikeout").astype(int), rbi=p["rbi"].fillna(0))
    pg = p.groupby(["game_pk", "batter_id", "team", "opp", "date", "season"])[
        ["pa", "hits", "total_bases", "home_runs", "strikeouts", "rbi"]].sum().reset_index()
    pg = pg.rename(columns={"batter_id": "pid", "game_pk": "game_id"})
    pg["name"] = pg["pid"].map(names).fillna("Player " + pg["pid"].astype(str))
    return pg


def mlb_project(pg, as_of, cur_season):
    hist = pg[pg["date"] < as_of].copy()
    for s in MLB_STATS:
        hist[s + "_pp"] = hist[s] / hist["pa"]
    rates = _ewma(hist, [s + "_pp" for s in MLB_STATS], "pid", hl=20, cur_season=cur_season, prior_w=0.5, last_n=60)
    pa = _ewma(hist, ["pa"], "pid", hl=4, cur_season=cur_season, prior_w=0.3, last_n=12)
    form = rates.join(pa[["pa"]])
    last = hist.sort_values("date").groupby("pid").tail(1).set_index("pid")
    form["name"], form["team"], form["last_date"] = last["name"], last["team"], last["date"]
    team_last = hist.groupby("team")["date"].max()
    form["active"] = form["last_date"].values == team_last.reindex(form["team"]).values
    opp = _opp_factors(hist[hist["season"] >= cur_season - 1], list(MLB_STATS), cur_season, k=25)
    return form, opp


def mlb_game_props(form, opp, team, opp_team, spreads):
    f = form[(form["team"] == team) & form["active"] & (form["pa"] >= 3.0) & (form["n_cur"] >= 10)]
    out = []
    for pid, r in f.sort_values("pa", ascending=False).head(9).iterrows():
        props = {}
        for st, kind in MLB_STATS.items():
            proj = float(r[st + "_pp"] * r["pa"]) * float(opp[st].get(opp_team, 1.0))
            props[st] = {"proj": proj, "kind": kind, **spread_params(spreads, st, kind, proj)}
        out.append({"name": r["name"], "role": f"about {r['pa']:.1f} plate appearances", "props": props})
    return out


# ---------------- MLB starting pitchers ----------------

PITCHER_STATS = {"pitcher_k": "count"}
PITCHER_LABELS = {"pitcher_k": "Strikeouts (pitcher)"}


def mlb_pitcher_games(starts, names):
    """starts: mlb.starter_games rows -> the same player-game shape the batter props use (pa = batters faced)."""
    pg = starts.rename(columns={"bf": "pa", "k": "pitcher_k"}).copy()
    pg["name"] = pg["pid"].map(names).fillna("Pitcher " + pg["pid"].astype(str))
    return pg


def mlb_pitcher_project(pg, as_of, cur_season):
    """Strikeouts per batter faced (shrunk toward league) x usual batters faced; opponent factor is how many
    strikeouts starters get against that lineup per game."""
    from .mlb import pitcher_form
    hist = pg[pg["date"] < as_of]
    f = pitcher_form(hist.rename(columns={"pa": "bf", "pitcher_k": "k"}), as_of, cur_season)
    form = pd.DataFrame({"pitcher_k_pp": f["k_pp"], "pa": f["bf"], "n_cur": f["n_cur"], "starts": f["starts"],
                         "team": f["team"], "last_date": f["last_date"]})
    form["name"] = hist.groupby("pid")["name"].last().reindex(form.index)
    opp = _opp_factors(hist[hist["season"] >= cur_season - 1], ["pitcher_k"], cur_season, k=25)
    return form, opp


def mlb_pitcher_prop(form, opp, pid, opp_team, spreads):
    """Prop entry for one probable starter, or None if he hasn't made enough starts this season."""
    if pid not in form.index or form.loc[pid, "n_cur"] < 3:
        return None
    r = form.loc[pid]
    proj = float(r["pitcher_k_pp"] * r["pa"]) * float(opp["pitcher_k"].get(opp_team, 1.0))
    return {"name": r["name"], "role": f"Starting pitcher, about {r['pa']:.0f} batters faced", "pitcher": True,
            "props": {"pitcher_k": {"proj": proj, "kind": "count", **spread_params(spreads, "pitcher_k", "count", proj)}}}

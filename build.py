"""Run every sport: fetch data, fit ratings, backtest, export dashboard JSON.

    python -m skiiipicks.build                      # download fresh data from the web
    python -m skiiipicks.build --data-dir ./data    # use local copies (see README for file names)
"""
import argparse
import json
import math
import os
import warnings
from datetime import date

import pandas as pd

from . import nfl, cfb, nba, mlb, stack, props, pnl, injuries
from .core import win_prob
from .core import fit_ratings, rank_dict

warnings.filterwarnings("ignore")


def clean(o):
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else round(o, 4)
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if hasattr(o, "item"):
        return clean(o.item())
    return o


def add_ranks(rows, keys):
    for k, hi in keys.items():
        d = {r["team"]: r[k] for r in rows if r.get(k) is not None}
        rk = rank_dict(d, hi)
        for r in rows:
            r[k + "_rk"] = rk.get(r["team"])
    return rows


def ratings_block(model):
    return {"mu": model.mu if hasattr(model, "mu") else model.eff.mu, "home": model.home,
            "off": model.off, "dfn": model.dfn}


def finish_upcoming(up, st, notes_fn):
    """Apply the stage-2 model to upcoming games and attach pick info."""
    up = up.copy()
    margin = st.predict(up)
    tot = up["pred_home"] + up["pred_away"]
    up["pred_home"], up["pred_away"] = tot / 2 + margin / 2, tot / 2 - margin / 2
    up["win_home"] = [win_prob(m, st.sigma) for m in margin]
    up["tier"] = [stack.tier(p) for p in up["win_home"]]
    covers = []
    for m, line in zip(margin, up["line_home"]):
        if line is None or pd.isna(line):
            covers.append(None); continue
        covers.append(st.p_cover_home(m - line))
    up["cover_home"] = covers
    up["notes"] = [notes_fn(r) for _, r in up.iterrows()]
    keep = ["date", "week", "home", "away", "neutral", "pred_home", "pred_away", "win_home", "tier",
            "line_home", "total_line", "cover_home", "notes", "series"]
    return up[[c for c in keep if c in up.columns]]


def nfl_notes(r):
    n = []
    for s in ("home", "away"):
        if r.get(s + "_qbc") == 1 and isinstance(r.get(s + "_qb_name"), str):
            usual = r.get(s + "_usual_qb")
            n.append(f"{r[s]} starting {r[s + '_qb_name']}" + (f" (usual starter {usual})" if isinstance(usual, str) else ""))
    rd = r.get("rest_diff")
    if rd is not None and not pd.isna(rd) and abs(rd) >= 3:
        n.append(f"{r['home'] if rd > 0 else r['away']} has {abs(int(rd))} more days of rest")
    if r.get("div_game") == 1:
        n.append("Division game")
    return n


def cfb_notes(r):
    n = []
    t = r.get("travel")
    if t is not None and not pd.isna(t) and t >= 2.0 and not r.get("neutral"):
        n.append(f"{r['away']} travels about {int(t * 621):,} miles")
    q = r.get("rqb_diff")
    if q is not None and not pd.isna(q) and q != 0:
        n.append(f"{r['home'] if q > 0 else r['away']} has a returning starting QB, the other side doesn't")
    return n


def total_sd(summ):
    """Spread of game totals around the projection, from the backtest's average miss."""
    return float(summ.get("total_mae", 10)) * 1.2533


def _attach(ups, team_props):
    for u in ups:
        u["props"] = {"home": team_props(u["home"], u["away"]), "away": team_props(u["away"], u["home"])}


def add_nfl_props(pbp_all, up, season):
    pg = props.nfl_player_games(pbp_all[pbp_all["season"] >= season - 2])
    kinds = {k: v[1] for k, v in props.NFL_STATS.items()}
    prev = pg[pg["season"] == season - 1]
    wk = prev.groupby("week")["date"].min().sort_values()
    cuts = [pd.Timestamp(c) for c in wk[wk.index >= 3].values] + [prev["date"].max() + pd.Timedelta(days=1)]
    role_stats = lambda r, g: props.ROLE_STATS.get(props.nfl_role(r), []) if props.nfl_role(r) else []
    bt = props.backtest(pg[pg["season"] <= season - 1], props.nfl_project, kinds, season - 1, cuts, role_stats)
    spreads = props.fit_spreads(bt, kinds)
    form, opp = props.nfl_project(pg, pd.Timestamp.today() + pd.Timedelta(days=1), season)
    _attach(up, lambda t, o: props.nfl_game_props(form, opp, t, o, spreads))
    return {"rows": props.summarize(bt, kinds, spreads, props.NFL_LABELS), "season": season - 1}


def add_nba_props(src, ups, season):
    try:
        paths = {season - 1: src("nba_pbox", season - 1), season: src("nba_pbox", season)}
        pg = props.nba_player_games(paths)
    except Exception as e:
        return None, f"Player data unavailable ({type(e).__name__})."
    test_season = season if (pg["season"] == season).sum() > 15000 else season - 1
    ts = pg[pg["season"] == test_season]
    cuts = list(pd.date_range(ts["date"].min() + pd.Timedelta(days=40), ts["date"].max() + pd.Timedelta(days=1), freq="7D"))
    bt = props.backtest(pg[pg["season"] <= test_season], props.nba_project, props.NBA_STATS, test_season, cuts,
                        lambda r, g: list(props.NBA_STATS) if r["minutes"] >= 18 else [])
    spreads = props.fit_spreads(bt, props.NBA_STATS)
    summary = {"rows": props.summarize(bt, props.NBA_STATS, spreads, props.NBA_LABELS), "season": test_season}
    if not ups:
        return summary, None
    first = pd.Timestamp(min(u["date"] for u in ups))
    if (first - pg["date"].max()).days > 10:
        return summary, ("Player props start a few days into the season. Last season's numbers can't see "
                         "summer trades and signings, so players could show up on the wrong team.")
    form, opp = props.nba_project(pg, pd.Timestamp.today() + pd.Timedelta(days=1), season)
    _attach(ups, lambda t, o: props.nba_game_props(form, opp, t, o, spreads))
    return summary, None


def result_rows(sport, g):
    d = g.dropna(subset=["home_pts", "away_pts"])
    return pd.DataFrame({"sport": sport, "date": pd.to_datetime(d["date"]).dt.strftime("%Y-%m-%d"),
                         "home": d["home"], "away": d["away"], "home_pts": d["home_pts"], "away_pts": d["away_pts"]}
                        ).drop_duplicates(["sport", "date", "home", "away"])


def hist_rows(sport, h, st, mlk):
    """Backtest picks for this season using the stage-2 model trained only on earlier seasons."""
    if h.empty:
        return []
    m = st.predict(h)
    rows = []
    for (_, r), mg in zip(h.iterrows(), m):
        d = pd.Timestamp(r["date"]).strftime("%Y-%m-%d")
        line = r.get("line_home")
        cov = st.p_cover_home(mg - line) if line is not None and not pd.isna(line) else None
        rows.append(pnl.make_row("backtest", sport, d, r["home"], r["away"], win_prob(mg, st.sigma), line, cov,
                                 mlk.get((d, r["home"], r["away"]))))
    return rows


def simple_rows(sport, bt, sigma):
    return [pnl.make_row("backtest", sport, pd.Timestamp(r["date"]).strftime("%Y-%m-%d"), r["home"], r["away"],
                         win_prob(r["pred_home"] - r["pred_away"], sigma)) for _, r in bt.iterrows()]


def nba_key_players(src, season):
    """Players who averaged 20+ minutes in their latest season, keyed by their latest team."""
    try:
        pg = props.nba_player_games({season: src("nba_pbox", season)})
    except Exception:
        return {}
    last = pg.sort_values("date").groupby("pid").tail(1)
    mins = pg.groupby("pid")["minutes"].mean()
    keys = {}
    for _, r in last.iterrows():
        if mins.get(r["pid"], 0) >= 20:
            keys.setdefault(r["team"], set()).add(r["name"])
    return keys


def nba_upcoming(src, model, season):
    """Use next season's schedule once it's published (ratings carry over at full strength, best in testing)."""
    for path in (src("nba_sched", season + 1), src("nba_sched", season)):
        try:
            ups = nba.upcoming(path, model)
        except Exception:
            continue
        if ups:
            return ups
    return []


def run(src, season_nfl=2026, season_cfb=2026, season_nba=2026, season_mlb=2026, log_path=None):
    out = {"built": date.today().isoformat(),
           "built_at": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    LIVE, BACK, RES = [], [], []

    # ---------------- NFL ----------------
    for season_nfl in (season_nfl, season_nfl - 1):
        try:
            print("NFL: walk-forward history 2021-%d..." % season_nfl)
            hist_seasons = list(range(season_nfl - 5, season_nfl + 1))
            games = nfl.load_games(src("nfl_games"), seasons=range(season_nfl - 6, season_nfl + 1))
            pbp_all = nfl.load_pbp(range(season_nfl - 6, season_nfl + 1), src("nfl_pbp"))
            erows = nfl.game_epa(pbp_all)
            parts, base_bt = [], {}
            for s in hist_seasons:
                bt, sm = nfl.backtest(games[games["season"] <= s], s, erows)
                parts.append(bt); base_bt[s] = sm
            hist = pd.concat(parts)
            feats = stack.nfl_features(pd.read_csv(src("nfl_games")))
            hist = hist.merge(feats, on="game_id", how="left")
            hist["pm"] = hist["pred_home"] - hist["pred_away"]
            hist["am"] = hist["home_pts"] - hist["away_pts"]
            seasons_tbl = stack.season_eval(hist, stack.NFL_X, hist_seasons[2:])
            st = stack.Stack(stack.NFL_X).fit(hist)
            model = nfl.make_fit(games, season_nfl, erows)(pd.Timestamp.today() + pd.Timedelta(days=1))
            pbp_cur = pbp_all[pbp_all["season"] == season_nfl]
            teams = nfl.team_profiles(pbp_cur, model.pts, season_nfl)
            add_ranks(teams, {"off_rating": True, "def_rating": True, "net_rating": True, "adj_off_epa": True,
                              "adj_def_epa": True, "off_ppd": True, "def_ppd": False, "off_sr": True, "def_sr": False})
            up = pd.DataFrame(nfl.upcoming(games, model, n_weeks=2))
            if not up.empty:
                gid = games[games["home_pts"].isna()][["game_id", "home", "away", "week"]]
                up = up.merge(gid, on=["home", "away", "week"], how="left").merge(feats, on="game_id", how="left")
                up["pm"] = up["pred_home"] - up["pred_away"]
                up = finish_upcoming(up, st, nfl_notes)
            up_recs = up.to_dict("records") if not up.empty else []
            raw = pd.read_csv(src("nfl_games"))
            mlk = {(str(r.gameday)[:10], r.home_team, r.away_team): {"home": r.home_moneyline, "away": r.away_moneyline}
                   for r in raw.itertuples() if r.season >= season_nfl - 1}
            for u in up_recs:
                LIVE.append(pnl.make_row("live", "nfl", u["date"], u["home"], u["away"], u["win_home"],
                                         u.get("line_home"), u.get("cover_home"), mlk.get((u["date"], u["home"], u["away"]))))
            st_prev = stack.Stack(stack.NFL_X).fit(hist[hist["season"] < season_nfl])
            BACK += hist_rows("nfl", hist[hist["season"] == season_nfl], st_prev, mlk)
            RES.append(result_rows("nfl", games))
            nfl_props_summary = add_nfl_props(pbp_all, up_recs, season_nfl)
            ninj, ninfo = injuries.get("nfl", src("nfl_inj", season_nfl), injuries.NFL_TEAMS)
            injuries.attach(up_recs, ninj, "nfl", {})
            out["nfl"] = {"label": "NFL", "season": season_nfl, "unit": "pts", "teams": teams,
                          "model": {**ratings_block(model.pts), "eff_off": model.eff.off, "eff_dfn": model.eff.dfn,
                                    "eff_home": model.eff.home, "scale": nfl.EPA_TO_PTS, "w": nfl.BLEND_W},
                          "sigma": st.sigma, "upcoming": up_recs,
                          "stack": st.coefs(), "props_summary": nfl_props_summary, "injury_info": ninfo, "prop_labels": props.NFL_LABELS,
                          "total_sd": total_sd(base_bt[season_nfl - 1]),
                          "seasons": seasons_tbl + [stack.pooled(seasons_tbl)],
                          "backtest": {"prior": {"season": season_nfl - 1, **base_bt[season_nfl - 1]},
                                       "current": {"season": season_nfl, **base_bt[season_nfl]}}}
            break
        except Exception as e:  # season data not out yet, or a source is down
            print("NFL failed for season", season_nfl, "-", repr(e)[:200])
            out.pop("nfl", None)

    # ---------------- CFB ----------------
    for season_cfb in (season_cfb, season_cfb - 1):
        try:
            print("CFB: walk-forward history...")
            cs = list(range(season_cfb - 5, season_cfb + 1))
            cg = cfb.load_games({y: src("cfb_sched", y) for y in range(season_cfb - 6, season_cfb + 1)},
                                {y: src("cfb_line", y) for y in cs})
            parts, cbase = [], {}
            for s in cs:
                bt, sm = cfb.backtest(cg[cg["season"] <= s], s)
                parts.append(bt); cbase[s] = sm
            chist = pd.concat(parts)
            cfeats = stack.cfb_features(pd.concat([pd.read_parquet(src("cfb_line", y)) for y in cs], ignore_index=True))
            chist = chist.merge(cfeats, on="game_id", how="left")
            chist["pm"] = chist["pred_home"] - chist["pred_away"]
            chist["am"] = chist["home_pts"] - chist["away_pts"]
            cseasons = stack.season_eval(chist, stack.CFB_X, cs[2:])
            cst = stack.Stack(stack.CFB_X).fit(chist)
            cst_prev = stack.Stack(stack.CFB_X).fit(chist[chist["season"] < season_cfb])
            BACK += hist_rows("cfb", chist[chist["season"] == season_cfb], cst_prev, {})
            RES.append(result_rows("cfb", cg))
            cmodel = cfb.make_fit(cg, season_cfb)(pd.Timestamp.today() + pd.Timedelta(days=1))
            cpbp = cfb.prep_pbp(pd.read_parquet(src("cfb_pbp", season_cfb)))
            nmap = cfb.name_map(cpbp, cg[cg["season"] == season_cfb])
            sr = fit_ratings(cfb.game_sr_rows(cpbp, nmap, season_cfb), pd.Timestamp.today() + pd.Timedelta(days=2), season_cfb,
                             alpha=2.0, half_life_days=9999, prior_weight=1)
            cteams = cfb.team_profiles(cpbp, nmap, cmodel.pts, sr, cg, season_cfb)
            add_ranks(cteams, {"off_rating": True, "def_rating": True, "net_rating": True, "adj_off_sr": True,
                               "adj_def_sr": True, "off_ppd": True, "def_ppd": False})
            fbs = {t["team"] for t in cteams}
            cup = pd.DataFrame(cfb.upcoming(cg, cmodel.pts))
            if not cup.empty:
                gid = cg[cg["home_pts"].isna() & (cg["season"] == season_cfb)][["game_id", "home", "away", "week"]]
                cup = cup.merge(gid, on=["home", "away", "week"], how="left").merge(cfeats, on="game_id", how="left")
                cup["pm"] = cup["pred_home"] - cup["pred_away"]
                cup = finish_upcoming(cup, cst, cfb_notes)
            for u in (cup.to_dict("records") if not cup.empty else []):
                LIVE.append(pnl.make_row("live", "cfb", u["date"], u["home"], u["away"], u["win_home"],
                                         u.get("line_home"), u.get("cover_home")))
            out["cfb"] = {"label": "College football", "season": season_cfb, "unit": "pts", "teams": cteams,
                          "model": {k: ({t: v for t, v in d.items() if t in fbs} if isinstance(d, dict) else d)
                                    for k, d in ratings_block(cmodel.pts).items()},
                          "sigma": cst.sigma, "upcoming": cup.to_dict("records") if not cup.empty else [],
                          "stack": cst.coefs(), "total_sd": total_sd(cbase[season_cfb - 1]),
                          "seasons": cseasons + [stack.pooled(cseasons)],
                          "backtest": {"prior": {"season": season_cfb - 1, **cbase[season_cfb - 1]},
                                       "current": {"season": season_cfb, **cbase[season_cfb]}},
                          "name_matches": len(nmap)}
            break
        except Exception as e:  # season data not out yet, or a source is down
            print("CFB failed for season", season_cfb, "-", repr(e)[:200])
            out.pop("cfb", None)

    # ---------------- NBA ----------------
    for season_nba in (season_nba, season_nba - 1):
        try:
            print("NBA...")
            try:
                box = nba.load_box({season_nba - 1: src("nba_box", season_nba - 1), season_nba: src("nba_box", season_nba)})
            except Exception:  # new season's box scores not published yet: use the last full season
                season_nba -= 1
                box = nba.load_box({season_nba - 1: src("nba_box", season_nba - 1), season_nba: src("nba_box", season_nba)})
            nbt, nsumm = nba.backtest(box, season_nba)
            BACK += simple_rows("nba", nbt, nba.SIGMA)
            RES.append(result_rows("nba", nba.games_from_box(box)))
            nmodel = nba.make_fit(box, season_nba)(pd.Timestamp.today())
            qp = nba.quarter_points(src("nba_sched", season_nba))
            nteams = nba.team_profiles(box, nmodel, season_nba, qp)
            add_ranks(nteams, {"off_rating": True, "def_rating": True, "net_rating": True, "ortg": True,
                               "drtg": False, "pace": True})
            nba_ups = nba_upcoming(src, nmodel, season_nba)
            nba_props_summary, nba_props_note = add_nba_props(src, nba_ups, season_nba)
            LIVE += [pnl.make_row("live", "nba", u["date"], u["home"], u["away"], u["win_home"]) for u in nba_ups]
            nba_map = {r.team_display_name: r.team for r in box.drop_duplicates("team").itertuples()}
            binj, binfo = injuries.get("nba", src("nba_inj", season_nba + 1), nba_map)
            injuries.attach(nba_ups, binj, "nba", nba_key_players(src, season_nba))
            out["nba"] = {"label": "NBA", "season": season_nba, "unit": "pts", "teams": nteams,
                          "model": {**ratings_block(nmodel.eff), "pace_mu": nmodel.pace.mu,
                                    "pace_off": nmodel.pace.off, "pace_dfn": nmodel.pace.dfn},
                          "sigma": nba.SIGMA, "upcoming": nba_ups, "props_summary": nba_props_summary,
                          "props_note": nba_props_note, "injury_info": binfo, "prop_labels": props.NBA_LABELS, "total_sd": total_sd(nsumm),
                          "backtest": {"current": {"season": season_nba, **nsumm}}}
            break
        except Exception as e:  # season data not out yet, or a source is down
            print("NBA failed for season", season_nba, "-", repr(e)[:200])
            out.pop("nba", None)

    # ---------------- MLB ----------------
    for season_mlb in (season_mlb, season_mlb - 1):
        try:
            print("MLB...")
            mg = mlb.load_games(src("mlb_sched", season_mlb))
            mbt, msumm = mlb.backtest(mg, season_mlb)
            BACK += simple_rows("mlb", mbt, mlb.SIGMA)
            RES.append(result_rows("mlb", mg))

            mmodel = mlb.make_fit(mg, season_mlb)(pd.Timestamp.today() + pd.Timedelta(days=1))
            mups = mlb.upcoming(mg, mmodel)
            LIVE += [pnl.make_row("live", "mlb", u["date"], u["home"], u["away"], u["win_home"]) for u in mups]
            prof = mlb.inning_profile(pd.read_parquet(src("mlb_pbp", season_mlb)), mg)
            mteams = mlb.team_profiles(mg, mmodel, prof, season_mlb)
            add_ranks(mteams, {"off_rating": True, "def_rating": True, "net_rating": True})
            out["mlb"] = {"label": "MLB", "season": season_mlb, "unit": "runs", "teams": mteams,
                          "model": ratings_block(mmodel), "sigma": mlb.SIGMA, "total_sd": total_sd(msumm), "upcoming": mups,
                          "backtest": {"current": {"season": season_mlb, **msumm}}}
            break
        except Exception as e:  # season data not out yet, or a source is down
            print("MLB failed for season", season_mlb, "-", repr(e)[:200])
            out.pop("mlb", None)

    results = pd.concat(RES, ignore_index=True) if RES else pd.DataFrame(columns=["sport", "date", "home", "away", "home_pts", "away_pts"])
    log = pnl.update_log(log_path, LIVE, results)
    back = pnl.grade(pd.DataFrame(BACK, columns=pnl.COLS).drop_duplicates(["sport", "date", "home", "away"]), results) if BACK else pd.DataFrame(columns=pnl.COLS)
    out["pnl"] = pnl.summarize(pd.concat([log, back], ignore_index=True))
    out["pnl"]["live_since"] = str(log["logged"].min()) if not log.empty else date.today().isoformat()
    for sp in ("nfl", "cfb", "nba", "mlb"):
        for u in (out.get(sp) or {}).get("upcoming", []):
            u["rating"] = pnl.rating(u["win_home"])
            if u.get("cover_home") is not None and not pd.isna(u.get("cover_home")):
                u["ats_rating"] = pnl.ats_rating(max(u["cover_home"], 1 - u["cover_home"]))
    return clean(out)


def current_seasons(today=None):
    """Pick each sport's current season from the date, so the build keeps working year to year."""
    t = today or date.today()
    football = t.year if t.month >= 6 else t.year - 1      # NFL/CFB seasons are labeled by their start year
    nba_season = t.year + 1 if t.month >= 10 else t.year   # NBA seasons are labeled by their end year
    mlb_season = t.year if t.month >= 3 else t.year - 1
    return dict(season_nfl=football, season_cfb=football, season_nba=nba_season, season_mlb=mlb_season)


def sources(data_dir=None):
    local = {
        "nfl_games": "nfl_games.csv", "nfl_pbp": "nfl_pbp_{}.parquet", "cfb_sched": "cfb_sched_{}.parquet",
        "cfb_line": "cfb_line_{}.parquet", "cfb_pbp": "cfb_pbp_{}.parquet", "nba_box": "nba_box_{}.parquet",
        "nba_sched": "nba_sched_{}.parquet", "nba_pbox": "nba_pbox_{}.parquet",
        "nfl_inj": "nfl_inj_{}.parquet", "nba_inj": "nba_inj_{}.parquet", "mlb_sched": "mlb_sched_{}.parquet", "mlb_pbp": "mlb_pbp_{}.parquet",
    }
    remote = {
        "nfl_games": nfl.GAMES_URL, "nfl_pbp": nfl.PBP_URL, "cfb_line": cfb.LINE_URL, "cfb_pbp": cfb.PBP_URL,
        "nba_box": nba.BOX_URL,
        "nfl_inj": nba.REL + "/espn_nfl_injuries/injuries_{}.parquet", "nba_inj": nba.REL + "/espn_nba_injuries/injuries_{}.parquet", "nba_pbox": nba.REL + "/espn_nba_player_boxscores/player_box_{}.parquet", "nba_sched": nba.SCHED_URL, "mlb_sched": mlb.SCHED_URL, "mlb_pbp": mlb.PBP_URL,
    }

    def src(key, season=None):
        if key == "nfl_pbp":
            return os.path.join(data_dir, local[key]) if data_dir else remote[key]
        if data_dir:
            return os.path.join(data_dir, local[key].format(season))
        if key == "cfb_sched":
            return cfb.SCHED_URL.get(season, cfb.REPO.format(season))
        return remote[key].format(season)
    return src


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default="dashboard_data.json")
    ap.add_argument("--log", default="picks_log.csv", help="live picks ledger (kept between runs)")
    a = ap.parse_args()
    data = run(sources(a.data_dir), **current_seasons(), log_path=a.log)
    with open(a.out, "w") as f:
        json.dump(data, f, separators=(",", ":"))
    print("wrote", a.out, os.path.getsize(a.out) // 1024, "KB")

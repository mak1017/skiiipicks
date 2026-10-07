"""Pick ratings and a profit-and-loss ledger.

Every build logs that day's picks to picks_log.csv (live picks), then grades any logged game that
now has a final score. Picks for games that haven't started are refreshed each build; once a game is
played its pick is frozen. Backtested picks for the current season are rebuilt each time and kept
separate, so the live record is never mixed with hindsight.

Units: 1 unit per pick.
  Moneyline: real American odds when available (NFL), otherwise win-loss record only.
  Spread: standard -110 (win +0.909, loss -1, push 0).
"""
import os

import numpy as np
import pandas as pd

GREEN, YELLOW = 0.70, 0.58          # winner pick: chance the picked team wins
ATS_GREEN, ATS_YELLOW = 0.56, 0.524  # spread lean: chance to cover (52.4% = break-even at -110)
COLS = ["source", "sport", "date", "home", "away", "pick", "win_prob", "rating", "ml_odds",
        "ats_side", "ats_line", "cover_prob", "ats_rating", "logged", "home_pts", "away_pts",
        "ml_result", "ml_units", "ats_result", "ats_units", "top10",
        "ml_home", "ml_away", "open_ml_home", "open_ml_away", "mkt_prob", "ev", "book", "clv"]
INJ_PENALTY = 0.04  # ranking only: a pick whose own key player is Out/Doubtful drops in the Top 10


def rating(p):
    p = max(p, 1 - p)
    return "green" if p >= GREEN else "yellow" if p >= YELLOW else "red"


def ats_rating(pc):
    return None if pc is None or pd.isna(pc) else ("green" if pc >= ATS_GREEN else "yellow" if pc >= ATS_YELLOW else "red")


def payout(odds):
    if odds is None or pd.isna(odds):
        return None
    return odds / 100 if odds > 0 else 100 / abs(odds)


def make_row(source, sport, date, home, away, win_home, line_home=None, cover_home=None, ml=None):
    home_pick = win_home >= 0.5
    r = {"source": source, "sport": sport, "date": str(date)[:10], "home": home, "away": away,
         "pick": home if home_pick else away, "win_prob": float(max(win_home, 1 - win_home)),
         "rating": rating(win_home), "ml_odds": None, "ats_side": None, "ats_line": None,
         "cover_prob": None, "ats_rating": None, "logged": pd.Timestamp.today().strftime("%Y-%m-%d")}
    if ml:
        r["ml_odds"] = ml.get("home" if home_pick else "away")
    if line_home is not None and not pd.isna(line_home) and cover_home is not None and not pd.isna(cover_home):
        hs = cover_home >= 0.5
        r["ats_side"] = home if hs else away
        r["ats_line"] = float(-line_home if hs else line_home)  # e.g. -3.5 = favored by 3.5
        r["cover_prob"] = float(cover_home if hs else 1 - cover_home)
        r["ats_rating"] = ats_rating(r["cover_prob"])
    return r


def grade(df, results):
    """results: sport, date, home, away, home_pts, away_pts (finals only)."""
    if df.empty:
        return df
    df = df.drop(columns=["home_pts", "away_pts"], errors="ignore").merge(
        results, on=["sport", "date", "home", "away"], how="left")
    done = df["home_pts"].notna()
    hw = df["home_pts"] > df["away_pts"]
    tie = df["home_pts"] == df["away_pts"]
    won = np.where(df["pick"] == df["home"], hw, ~hw & ~tie)
    df["ml_result"] = np.where(~done, None, np.where(tie, "P", np.where(won, "W", "L")))
    pay = df["ml_odds"].map(payout)
    df["ml_units"] = np.where(~done | pay.isna(), np.nan,
                              np.where(tie, 0.0, np.where(won, pay.fillna(0), -1.0)))
    side_home = df["ats_side"] == df["home"]
    margin_side = np.where(side_home, df["home_pts"] - df["away_pts"], df["away_pts"] - df["home_pts"])
    cov = margin_side + df["ats_line"].astype(float)  # side covers if margin + its line > 0
    has = done & df["ats_side"].notna()
    df["ats_result"] = np.where(~has, None, np.where(cov > 0, "W", np.where(cov < 0, "L", "P")))
    df["ats_units"] = np.where(~has, np.nan, np.where(cov > 0, 100 / 110, np.where(cov < 0, -1.0, 0.0)))
    # closing line value: did the price on our side shorten between the first and last odds we logged?
    if "open_ml_home" in df.columns:
        def imp(a):
            a = pd.to_numeric(a, errors="coerce")
            return np.where(a > 0, 100 / (a + 100), -a / (-a + 100))
        ph = df["pick"] == df["home"]
        o = np.where(ph, imp(df["open_ml_home"]), imp(df["open_ml_away"]))
        c = np.where(ph, imp(df["ml_home"]), imp(df["ml_away"]))
        df["clv"] = np.where(done, c - o, np.nan)
    return df


def update_log(path, live_rows, results, today=None):
    today = (today or pd.Timestamp.today()).strftime("%Y-%m-%d")
    old = pd.read_csv(path, dtype={"date": str}) if path and os.path.exists(path) else pd.DataFrame(columns=COLS)
    new = pd.DataFrame(live_rows, columns=COLS) if live_rows else pd.DataFrame(columns=COLS)
    key = ["sport", "date", "home", "away"]
    opens = {}
    if not old.empty and "open_ml_home" in old.columns:
        for r in old.itertuples():
            if pd.notna(getattr(r, "open_ml_home", None)):
                opens[(r.sport, r.date, r.home, r.away)] = (r.open_ml_home, r.open_ml_away)
    if not old.empty:
        # freeze anything already played or dated before today; refresh the rest with today's picks
        frozen = old[(old["date"] < today) | old["home_pts"].notna()]
        new = new[new["date"] >= today]
        new = new.merge(frozen[key], on=key, how="left", indicator=True)
        new = new[new["_merge"] == "left_only"].drop(columns="_merge")
        keep_old = old.merge(new[key], on=key, how="left", indicator=True)
        keep_old = keep_old[keep_old["_merge"] == "left_only"].drop(columns="_merge")
        log = pd.concat([keep_old, new], ignore_index=True)
    else:
        log = new
    for c in COLS:
        if c not in log.columns:
            log[c] = None
    # opening price = first price ever logged for the game; later builds only update the latest price
    oh, oa = [], []
    for r in log.itertuples():
        o = opens.get((r.sport, r.date, r.home, r.away))
        oh.append(o[0] if o else r.ml_home)
        oa.append(o[1] if o else r.ml_away)
    log["open_ml_home"], log["open_ml_away"] = oh, oa
    log = log.drop_duplicates(key, keep="last")
    log = grade(log, results)
    log = log.sort_values(["date", "sport", "home"]).reset_index(drop=True)
    if path:
        log[COLS].to_csv(path, index=False)
    return log


def top10_today(games, today):
    """games: dicts with sport, date, home, away, win_home, notes."""
    import re
    cands = []
    for g in games:
        if g["date"] < today:
            continue
        p = max(g["win_home"], 1 - g["win_home"])
        pick = g["home"] if g["win_home"] >= 0.5 else g["away"]
        hurt = [n for n in (g.get("notes") or []) if n.startswith(pick + ":") and re.search(r"\((Out|Doubtful)\)", n)]
        priced_in = bool(g.get("injury_adj"))  # NBA: the projection already accounts for who is out
        cands.append({**g, "pick": pick, "conf": p, "score": p - (INJ_PENALTY if hurt and not priced_in else 0),
                      "injury_flag": bool(hurt)})
    # today's games only (Eastern time); fewer than 10 on light days
    day = [c for c in cands if c["date"] == today]
    return sorted(day, key=lambda c: -c["score"])[:10]


def mark_backtest_top10(df):
    """Daily top 10 across sports by win chance, for the backtested track record."""
    if df.empty:
        return df
    df = df.copy()
    df["top10"] = df.groupby("date")["win_prob"].rank(ascending=False, method="first")
    df.loc[df["top10"] > 10, "top10"] = None
    return df


def _tally(d):
    g = d[d["ml_result"].isin(["W", "L", "P"])]
    a = d[d["ats_result"].isin(["W", "L", "P"])]
    ml_u = g["ml_units"].dropna()
    out = {
        "picks": int(len(g)), "w": int((g["ml_result"] == "W").sum()), "l": int((g["ml_result"] == "L").sum()),
        "ml_units": float(ml_u.sum()) if len(ml_u) else None, "ml_bets": int(len(ml_u)),
        "ats_w": int((a["ats_result"] == "W").sum()), "ats_l": int((a["ats_result"] == "L").sum()),
        "ats_p": int((a["ats_result"] == "P").sum()), "ats_units": float(a["ats_units"].sum()) if len(a) else None,
        "pending": int((d["ml_result"].isna()).sum()),
    }
    if "clv" in d.columns:
        cv = pd.to_numeric(g["clv"], errors="coerce").dropna()
        cv = cv[pd.to_numeric(g.loc[cv.index, "open_ml_home"], errors="coerce").notna()]
        out["clv_n"] = int(len(cv))
        if len(cv):
            out["clv_avg"] = float(cv.mean())
            out["clv_beat"] = float((cv > 0).mean())
            out["clv_same"] = float((cv == 0).mean())
    return out


def summarize(df):
    out = {}
    if df.empty:
        return out
    for src in ("live", "backtest"):
        s = df[df["source"] == src]
        if s.empty:
            continue
        blk = {"all": _tally(s), "by_sport": {}, "by_rating": {}, "ats_by_rating": {}}
        for sp, d in s.groupby("sport"):
            blk["by_sport"][sp] = {"all": _tally(d), "by_rating": {r: _tally(d[d["rating"] == r]) for r in ("green", "yellow", "red")},
                                   "ats_by_rating": {r: _tally(d[d["ats_rating"] == r]) for r in ("green", "yellow", "red")},
                                   "curve": _curve(d), "recent": _recent(d)}
        blk["by_rating"] = {r: _tally(s[s["rating"] == r]) for r in ("green", "yellow", "red")}
        evs = pd.to_numeric(s["ev"], errors="coerce") if "ev" in s else pd.Series(dtype=float)
        blk["value"] = {"all": _tally(s[evs > 0]), "green": _tally(s[evs >= 0.05])}
        t10 = s[s["top10"].notna()] if "top10" in s else s.iloc[0:0]
        blk["top10"] = {"all": _tally(t10), "curve": _curve(t10)}
        blk["curve"] = _curve(s)
        out[src] = blk
    return out


def _curve(d):
    g = d[d["ml_result"].isin(["W", "L", "P"])].sort_values("date")
    if g.empty:
        return []
    u = g["ats_units"].fillna(0) + g["ml_units"].fillna(0)
    daily = u.groupby(g["date"]).sum().cumsum()
    rec = (g["ml_result"] == "W").astype(int).groupby(g["date"]).sum().cumsum()
    n = g.groupby("date").size().cumsum()
    return [{"date": k, "units": round(float(v), 2), "win_pct": round(float(rec[k] / n[k]), 3)} for k, v in daily.items()]


def _recent(d, n=15):
    g = d[d["ml_result"].isin(["W", "L", "P"])].sort_values("date", ascending=False).head(n)
    keep = ["date", "home", "away", "pick", "win_prob", "rating", "ml_result", "ml_units", "ats_side",
            "ats_line", "ats_result", "ats_units", "home_pts", "away_pts"]
    return g[keep].to_dict("records")

"""Second-stage ("stacked") model.

Stage 1 (core.py) rates teams. Stage 2 learns, from several seasons of
walk-forward predictions, how much to trust that rating and how much to adjust
for game context the ratings can't see:

  NFL  rest advantage, starting-QB changes, divisional games
  CFB  Elo, recruiting talent, returning production, returning QB, travel

Stage 2 is always trained only on seasons before the one being predicted, so
the per-season results reported on the dashboard are genuinely out of sample.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

NFL_X = ["pm", "rest_diff", "qbc", "div_game"]
CFB_X = ["pm", "elo_diff", "tal_diff", "rtp_diff", "rqb_diff", "travel", "tal_diff_e", "rtp_diff_e"]


# ---------------- features ----------------

def nfl_features(raw: pd.DataFrame) -> pd.DataFrame:
    """raw = nflverse games.csv. QB change = listed starter differs from the team's main starter over its last 4 games."""
    g = raw[raw["season"] >= 2018].sort_values(["gameday", "game_id"])
    q = pd.concat([pd.DataFrame({"game_id": g["game_id"], "date": g["gameday"], "team": g[s + "_team"],
                                 "qb": g[s + "_qb_id"].fillna(g[s + "_qb_name"]), "qbn": g[s + "_qb_name"], "side": s})
                   for s in ("home", "away")]).sort_values(["team", "date"])
    flags = []
    for _, df in q.groupby("team"):
        prev, main_names = [], []
        for qb, name in zip(df["qb"], df["qbn"]):
            last = [p for p in prev[-4:] if pd.notna(p[0])]
            ids = [p[0] for p in last]
            main = max(set(ids), key=ids.count) if ids else qb
            main_name = next((p[1] for p in reversed(last) if p[0] == main), None)
            flags.append((int(pd.notna(qb) and qb != main), main_name))
            prev.append((qb, name))
    q["qbc"] = [f[0] for f in flags]
    q["usual_qb"] = [f[1] for f in flags]
    f = g[["game_id", "home_rest", "away_rest", "div_game", "home_qb_name", "away_qb_name"]].copy()
    for s in ("home", "away"):
        f = f.merge(q[q["side"] == s][["game_id", "qbc", "usual_qb"]].rename(
            columns={"qbc": s + "_qbc", "usual_qb": s + "_usual_qb"}), on="game_id", how="left")
    f["rest_diff"] = (f["home_rest"] - f["away_rest"]).clip(-7, 7)
    f["qbc"] = f["home_qbc"].fillna(0) - f["away_qbc"].fillna(0)
    return f


def _hav(a1, o1, a2, o2):
    a1, o1, a2, o2 = map(np.radians, [a1, o1, a2, o2])
    return 6371 * 2 * np.arcsin(np.sqrt(np.sin((a2 - a1) / 2) ** 2 + np.cos(a1) * np.cos(a2) * np.sin((o2 - o1) / 2) ** 2))


def cfb_features(lines: pd.DataFrame) -> pd.DataFrame:
    L = lines
    f = pd.DataFrame({
        "game_id": L["game_id"],
        "elo_diff": L["home_pregame_elo"] - L["away_pregame_elo"],
        "tal_diff": L["home_talent"] - L["away_talent"],
        "rtp_diff": L["home_ovr_rtprod"] - L["away_ovr_rtprod"],
        "rqb_diff": L["home_returning_qb"].astype(float) - L["away_returning_qb"].astype(float),
        "travel": _hav(L["away_latitude"], L["away_longitude"], L["home_latitude"], L["home_longitude"]) / 1000,
        "week_l": L["week"],
    }).drop_duplicates("game_id")
    early = (f["week_l"] <= 4).astype(int)
    f["tal_diff_e"] = f["tal_diff"].fillna(0) * early
    f["rtp_diff_e"] = f["rtp_diff"].fillna(0) * early
    return f


# ---------------- model ----------------

class Stack:
    def __init__(self, X, alpha=1.0):
        self.X, self.alpha = X, alpha

    def fit(self, df):
        d = df.dropna(subset=["am"])
        self.fill = d[self.X].mean()
        self.m = Ridge(alpha=self.alpha).fit(d[self.X].fillna(self.fill), d["am"])
        self.sigma = float(np.std(d["am"] - self.m.predict(d[self.X].fillna(self.fill))))
        # cover model: does the stacked edge over the line predict covers?
        v = d.dropna(subset=["line_home"]).copy()
        v["e"] = self.m.predict(v[self.X].fillna(self.fill)) - v["line_home"]
        v["c"] = np.sign(v["am"] - v["line_home"])
        v = v[v["c"] != 0]
        self.cover = None
        if len(v) > 200:
            # model P(home covers) from edge, no intercept bias beyond what data shows
            self.cover = LogisticRegression(C=0.5).fit(v[["e"]], (v["c"] > 0).astype(int))
        return self

    def predict(self, df):
        return self.m.predict(df[self.X].fillna(self.fill))

    def p_cover_home(self, edge):
        if self.cover is None:
            return 0.5
        return float(self.cover.predict_proba(pd.DataFrame({"e": [edge]}))[0, 1])

    def coefs(self):
        return {k: float(v) for k, v in zip(self.X, self.m.coef_)}


def season_eval(hist, X, seasons, alpha=1.0):
    """Train stage 2 on all seasons before s, test on s. Returns per-season rows."""
    rows = []
    for s in seasons:
        tr, te = hist[hist["season"] < s], hist[(hist["season"] == s) & hist["am"].notna()]
        if tr.empty or te.empty:
            continue
        st = Stack(X, alpha).fit(tr)
        p = st.predict(te)
        dec = te["am"] != 0
        r = {"season": int(s), "games": int(len(te)),
             "winner_pct": float(((p > 0) == (te["am"] > 0))[dec].mean()),
             "base_winner_pct": float(((te["pm"] > 0) == (te["am"] > 0))[dec].mean()),
             "margin_mae": float(np.abs(te["am"] - p).mean()),
             "base_mae": float(np.abs(te["am"] - te["pm"]).mean())}
        v = te["line_home"].notna()
        if v.sum() > 20:
            e = (p - te["line_home"])[v]
            c = np.sign(te["am"] - te["line_home"])[v]
            k = c != 0
            r["vegas_mae"] = float(np.abs(te["am"] - te["line_home"])[v].mean())
            r["ats_pct"] = float((np.sign(e[k]) == c[k]).mean())
            r["ats_bets"] = int(k.sum())
            big = k & (e.abs() >= 3)
            r["ats3_pct"] = float((np.sign(e[big]) == c[big]).mean()) if big.sum() >= 10 else None
            r["ats3_bets"] = int(big.sum())
        rows.append(r)
    return rows


def pooled(rows):
    """Combine per-season rows into one all-seasons line."""
    if not rows:
        return {}
    n = sum(r["games"] for r in rows)
    w = lambda k: sum(r[k] * r["games"] for r in rows if r.get(k) is not None) / n
    out = {"season": "All", "games": n, "winner_pct": w("winner_pct"), "base_winner_pct": w("base_winner_pct"),
           "margin_mae": w("margin_mae"), "base_mae": w("base_mae")}
    ats = [r for r in rows if r.get("ats_bets")]
    if ats:
        nb = sum(r["ats_bets"] for r in ats)
        out["ats_pct"] = sum(r["ats_pct"] * r["ats_bets"] for r in ats) / nb
        out["ats_bets"] = nb
        out["vegas_mae"] = sum(r["vegas_mae"] * r["games"] for r in ats) / sum(r["games"] for r in ats)
        b3 = [r for r in ats if r.get("ats3_pct") is not None]
        if b3:
            n3 = sum(r["ats3_bets"] for r in b3)
            out["ats3_pct"] = sum(r["ats3_pct"] * r["ats3_bets"] for r in b3) / n3
            out["ats3_bets"] = n3
    return out


def tier(p):
    p = max(p, 1 - p)
    return "Strong" if p >= 0.75 else "Solid" if p >= 0.65 else "Lean" if p >= 0.57 else "Toss-up"

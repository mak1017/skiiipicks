"""Second-stage ("stacked") model.

Stage 1 (core.py) rates teams. Stage 2 learns, from several seasons of
walk-forward predictions, how much to trust that rating and how much to adjust
for game context the ratings can't see:

  NFL  rest advantage, starting-QB changes, divisional games
  CFB  Elo, recruiting talent, returning production, returning QB, travel

When a Vegas spread is known, a second ridge model also uses it (the "market-aware" model). The spread
already prices in injuries, weather and news the ratings can't see, so the win chance comes from that
model; the spread lean still comes from the ratings-only margin, since that is the part that can
disagree with Vegas.

NBA and MLB use a logistic stage 2 (WinStack): NBA adds lineup strength changes from the player box
scores and injury report; MLB adds the two starting pitchers.

Stage 2 is always trained only on seasons before the one being predicted, so
the per-season results reported on the dashboard are genuinely out of sample.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
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
        # market-aware model: same features plus the closing spread (home expected margin)
        self.mk = None
        lv = d.dropna(subset=["line_home"])
        if len(lv) > 200:
            self.XM = self.X + ["line_home"]
            self.mfill = lv[self.XM].mean()
            self.mk = Ridge(alpha=self.alpha).fit(lv[self.XM].fillna(self.mfill), lv["am"])
            self.msigma = float(np.std(lv["am"] - self.mk.predict(lv[self.XM].fillna(self.mfill))))
        return self

    def predict_mkt(self, df):
        """(margin, sigma) per row: market-aware where a spread is known, ratings-only otherwise."""
        m = self.predict(df)
        sig = np.full(len(df), self.sigma)
        if self.mk is None or "line_home" not in df:
            return m, sig
        has = pd.to_numeric(df["line_home"], errors="coerce").notna().values
        if has.any():
            sub = df[has].copy()
            sub["line_home"] = pd.to_numeric(sub["line_home"])
            m = m.copy(); m[has] = self.mk.predict(sub[self.XM].fillna(self.mfill)); sig[has] = self.msigma
        return m, sig

    def win_prob(self, df):
        m, sig = self.predict_mkt(df)
        return norm.cdf(m / sig)

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
        wp = st.win_prob(te)
        dec = te["am"] != 0
        won = (te["am"] > 0).values
        lh = pd.to_numeric(te["line_home"], errors="coerce")
        vegas_side = np.where(lh.notna(), lh > 0, p > 0)  # pick'em lines fall back to the model
        vegas_side = np.where(lh == 0, p > 0, vegas_side)
        conf = np.maximum(wp, 1 - wp)
        hi = dec.values & (conf >= 0.70)
        r = {"season": int(s), "games": int(len(te)),
             "winner_pct": float(((wp > 0.5) == won)[dec].mean()),
             "model_winner_pct": float(((p > 0) == won)[dec].mean()),
             "vegas_winner_pct": float((vegas_side == won)[dec].mean()),
             "base_winner_pct": float(((te["pm"] > 0) == (te["am"] > 0))[dec].mean()),
             "brier": float(np.mean((wp[dec.values] - won[dec.values]) ** 2)),
             "green_pct": float(((wp > 0.5) == won)[hi].mean()) if hi.sum() else None, "green_n": int(hi.sum()),
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
    out = {"season": "All", "games": n, **{k: w(k) for k in rows[0] if k not in ("season", "games", "green_pct", "green_n")
                                           and not k.startswith("ats") and k != "vegas_mae" and rows[0][k] is not None}}
    gn = sum(r.get("green_n") or 0 for r in rows)
    if gn:
        out["green_pct"] = sum((r["green_pct"] or 0) * r["green_n"] for r in rows if r.get("green_n")) / gn
        out["green_n"] = gn
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


# ---------------- logistic stage 2 (NBA, MLB) ----------------

NBA_X = ["pm", "lineup_diff"]
MLB_X = ["pm", "sp_woba", "sp_fip"]


class WinStack:
    """P(home wins) from the ratings margin plus context, fit by logistic regression.

    Also fixes the ratings' calibration: a single sigma over- or under-states confidence, and the
    learned intercept carries the real home edge.
    """

    def __init__(self, X, C=10.0):
        self.X, self.C = X, C

    def fit(self, df):
        d = df.dropna(subset=["am"])
        d = d[d["am"] != 0]
        self.fill = d[self.X].mean()
        self.m = LogisticRegression(C=self.C, max_iter=2000).fit(d[self.X].fillna(self.fill), (d["am"] > 0).astype(int))
        return self

    def win_prob(self, df):
        return self.m.predict_proba(df[self.X].fillna(self.fill))[:, 1]

    def coefs(self):
        return {k: float(v) for k, v in zip(self.X, self.m.coef_[0])}


def win_eval(hist, X, periods, sigma, C=10.0, freq=None):
    """Walk-forward test of a WinStack: for each test season (refit monthly within it when freq='M'),
    train only on games before the test window. Returns per-season rows like season_eval."""
    rows, preds = [], []
    for s in periods:
        te_all = hist[(hist["season"] == s) & hist["am"].notna() & (hist["am"] != 0)]
        if te_all.empty or hist[hist["date"] < te_all["date"].min()].shape[0] < 300:
            continue
        groups = te_all.groupby(pd.to_datetime(te_all["date"]).dt.to_period("M")) if freq == "M" else [(s, te_all)]
        wp, y, pm = [], [], []
        for _, te in groups:
            tr = hist[hist["date"] < te["date"].min()]
            ws = WinStack(X, C).fit(tr)
            wp.append(ws.win_prob(te)); y.append((te["am"] > 0).values); pm.append(te["pm"].values)
            preds.append(te[["date", "home", "away", "season"]].assign(wp=wp[-1]))
        wp, y, pm = np.concatenate(wp), np.concatenate(y), np.concatenate(pm)
        conf = np.maximum(wp, 1 - wp); hi = conf >= 0.70
        base_p = norm.cdf(pm / sigma)
        rows.append({"season": int(s), "games": int(len(y)),
                     "winner_pct": float(((wp > 0.5) == y).mean()),
                     "base_winner_pct": float(((pm > 0) == y).mean()),
                     "brier": float(np.mean((wp - y) ** 2)), "base_brier": float(np.mean((base_p - y) ** 2)),
                     "green_pct": float(((wp > 0.5) == y)[hi].mean()) if hi.sum() else None, "green_n": int(hi.sum())})
    return rows, (pd.concat(preds) if preds else pd.DataFrame(columns=["date", "home", "away", "season", "wp"]))

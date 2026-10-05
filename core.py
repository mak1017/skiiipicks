"""Core rating engine shared by every sport.

The model: for every team-game we observe points scored (or another per-game
stat). We fit

    stat = league_avg + offense[team] - defense[opponent] + home_edge * loc

with ridge regression so each team's offense is measured against the defenses
it actually faced (and vice versa). Recent games count more (exponential time
decay) and last season's games count less, so early-season ratings lean on the
prior year and fade it out as new games arrive.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge


@dataclass
class Ratings:
    mu: float
    home: float
    off: dict
    dfn: dict  # positive = better defense (prevents points)
    teams: list = field(default_factory=list)

    def predict(self, home: str, away: str, neutral: bool = False):
        loc = 0.0 if neutral else 0.5
        h = self.mu + self.off.get(home, 0) - self.dfn.get(away, 0) + self.home * loc
        a = self.mu + self.off.get(away, 0) - self.dfn.get(home, 0) - self.home * loc
        return h, a


def team_game_rows(games: pd.DataFrame, stat_home: str, stat_away: str) -> pd.DataFrame:
    """Turn one-row-per-game into two rows (one per offense)."""
    neu = games.get("neutral", pd.Series(False, index=games.index)).fillna(False).astype(bool)
    h = pd.DataFrame({
        "date": games["date"], "team": games["home"], "opp": games["away"],
        "y": games[stat_home], "loc": np.where(neu, 0.0, 0.5), "season": games["season"],
    })
    a = pd.DataFrame({
        "date": games["date"], "team": games["away"], "opp": games["home"],
        "y": games[stat_away], "loc": np.where(neu, 0.0, -0.5), "season": games["season"],
    })
    return pd.concat([h, a], ignore_index=True).dropna(subset=["y"])


def fit_ratings(rows: pd.DataFrame, as_of, cur_season: int, alpha: float = 4.0,
                half_life_days: float = 120.0, prior_weight: float = 0.35,
                extra_weight: pd.Series | None = None) -> Ratings:
    """Opponent-adjusted offense/defense ratings with time decay."""
    rows = rows[rows["date"] < as_of]
    if rows.empty:
        return Ratings(0, 0, {}, {}, [])
    teams = sorted(set(rows["team"]) | set(rows["opp"]))
    idx = {t: i for i, t in enumerate(teams)}
    n, k = len(rows), len(teams)
    X = np.zeros((n, 2 * k + 1))
    r = np.arange(n)
    X[r, rows["team"].map(idx).values] = 1.0
    X[r, k + rows["opp"].map(idx).values] = 1.0
    X[:, -1] = rows["loc"].values
    age = (pd.Timestamp(as_of) - rows["date"]).dt.days.values
    w = np.power(0.5, age / half_life_days)
    w = w * np.where(rows["season"].values < cur_season, prior_weight, 1.0)
    if extra_weight is not None:
        w = w * extra_weight.loc[rows.index].values
    m = Ridge(alpha=alpha, fit_intercept=True)
    m.fit(X, rows["y"].values, sample_weight=w)
    c = m.coef_
    return Ratings(
        mu=float(m.intercept_), home=float(c[-1]),
        off={t: float(c[i]) for t, i in idx.items()},
        dfn={t: float(-c[k + i]) for t, i in idx.items()},
        teams=teams,
    )


def win_prob(margin: float, sigma: float) -> float:
    return 0.5 * (1 + math.erf(margin / (sigma * math.sqrt(2))))


def walk_forward(games: pd.DataFrame, fit_fn, start, step: str = "W") -> pd.DataFrame:
    """Re-fit before each period and predict only games the model hasn't seen."""
    test = games[(games["date"] >= start) & games["home_pts"].notna()].copy()
    out = []
    for _, chunk in test.groupby(test["date"].dt.to_period(step)):
        as_of = chunk["date"].min()
        model = fit_fn(as_of)
        for _, g in chunk.iterrows():
            ph, pa = model.predict(g["home"], g["away"], bool(g.get("neutral", False)))
            out.append({**g.to_dict(), "pred_home": ph, "pred_away": pa})
    return pd.DataFrame(out)


def backtest_summary(bt: pd.DataFrame, sigma: float | None = None) -> dict:
    pm = bt["pred_home"] - bt["pred_away"]
    am = bt["home_pts"] - bt["away_pts"]
    dec = am != 0
    sigma = sigma or float(np.std(am - pm))
    p = np.array([win_prob(x, sigma) for x in pm])
    out = {
        "games": int(len(bt)),
        "winner_pct": float(((pm > 0) == (am > 0))[dec].mean()),
        "margin_mae": float(np.abs(am - pm).mean()),
        "total_mae": float(np.abs((bt["home_pts"] + bt["away_pts"]) - (bt["pred_home"] + bt["pred_away"])).mean()),
        "brier": float(np.mean((p[dec] - (am[dec] > 0)) ** 2)),
        "sigma": sigma,
    }
    # Calibration buckets for the dashboard
    cal = []
    for lo in (0.5, 0.6, 0.7, 0.8, 0.9):
        fav_p = np.where(p >= 0.5, p, 1 - p)
        fav_won = np.where(p >= 0.5, am > 0, am < 0)
        msk = (fav_p >= lo) & (fav_p < lo + 0.1) & dec.values
        if msk.sum() >= 5:
            cal.append({"bucket": f"{int(lo*100)}-{int(lo*100)+10}%", "n": int(msk.sum()),
                        "predicted": float(fav_p[msk].mean()), "actual": float(fav_won[msk].mean())})
    out["calibration"] = cal
    if "line_home" in bt.columns and bt["line_home"].notna().any():
        out.update(ats_summary(bt))
    return out


def ats_summary(bt: pd.DataFrame) -> dict:
    """Against-the-spread record. line_home = Vegas expected home margin."""
    b = bt.dropna(subset=["line_home"]).copy()
    b["pm"] = b["pred_home"] - b["pred_away"]
    b["am"] = b["home_pts"] - b["away_pts"]
    b["edge"] = b["pm"] - b["line_home"]
    b["cover"] = np.sign(b["am"] - b["line_home"])  # +1 home covered
    res = {"vegas_margin_mae": float(np.abs(b["am"] - b["line_home"]).mean())}
    tiers = []
    for thr in (0, 2, 3, 5, 7):
        s = b[b["edge"].abs() >= thr]
        s = s[s["cover"] != 0]
        if len(s) < 10:
            continue
        wins = int((np.sign(s["edge"]) == s["cover"]).sum())
        n = len(s)
        tiers.append({"min_edge": thr, "bets": n, "wins": wins, "pct": wins / n,
                      "units": round(wins * (100 / 110) - (n - wins), 1)})
    res["ats"] = tiers
    if "total_line" in b.columns and b["total_line"].notna().any():
        t = b.dropna(subset=["total_line"]).copy()
        t["edge_t"] = (t["pred_home"] + t["pred_away"]) - t["total_line"]
        t["res_t"] = np.sign((t["home_pts"] + t["away_pts"]) - t["total_line"])
        ttiers = []
        for thr in (0, 2, 3, 5):
            s = t[(t["edge_t"].abs() >= thr) & (t["res_t"] != 0)]
            if len(s) < 10:
                continue
            wins = int((np.sign(s["edge_t"]) == s["res_t"]).sum())
            ttiers.append({"min_edge": thr, "bets": len(s), "wins": wins, "pct": wins / len(s),
                           "units": round(wins * (100 / 110) - (len(s) - wins), 1)})
        res["totals"] = ttiers
    return res


def rank_dict(d: dict, higher_better: bool = True) -> dict:
    s = sorted(d, key=lambda t: d[t], reverse=higher_better)
    return {t: i + 1 for i, t in enumerate(s)}


class Blend:
    """Blend a points-based model with an efficiency-based (EPA) model.

    margin = w * points_margin + (1 - w) * scale * efficiency_margin
    The total comes from the points model (efficiency says little about pace).
    """

    def __init__(self, pts: Ratings, eff: Ratings | None, scale: float, w: float = 0.6):
        self.pts, self.eff, self.scale, self.w = pts, eff, scale, w
        self.off, self.dfn, self.mu, self.home = pts.off, pts.dfn, pts.mu, pts.home

    def predict(self, home, away, neutral=False):
        h, a = self.pts.predict(home, away, neutral)
        if self.eff is None or home not in self.eff.off or away not in self.eff.off:
            return h, a
        eh, ea = self.eff.predict(home, away, neutral)
        m = self.w * (h - a) + (1 - self.w) * self.scale * (eh - ea)
        tot = h + a
        return tot / 2 + m / 2, tot / 2 - m / 2


def epa_rows(ge: pd.DataFrame) -> pd.DataFrame:
    """ge: one row per offense-game with columns date, team, opp, epa, is_home, season, neutral."""
    loc = np.where(ge["neutral"], 0.0, np.where(ge["is_home"], 0.5, -0.5))
    return pd.DataFrame({"date": pd.to_datetime(ge["date"]), "team": ge["team"], "opp": ge["opp"],
                         "y": ge["epa"], "loc": loc, "season": ge["season"]})

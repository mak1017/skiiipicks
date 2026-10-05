"""Injury reports for NFL and NBA.

Source order:
  1. ESPN's public injury page (updates through the day). Unofficial, so it can change or block requests.
  2. The sportsdataverse daily injury feed (refreshed about once a day).
Whichever works first is used, and the dashboard says which one and when.

What this does: flags injured players on picks, removes ruled-out players from player props, and tags
questionable ones. It does not change projected scores (except NFL QB changes, handled in stack.py).
"""
import json
import re
import urllib.request
from datetime import datetime, timezone

import pandas as pd

LIVE_URL = {"nfl": "https://site.api.espn.com/apis/site/v2/sports/football/nfl/injuries",
            "nba": "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"}
SHOW = {"nfl": {"Out", "Doubtful", "Questionable", "Suspension"},
        "nba": {"Out", "Day-To-Day", "Doubtful", "Questionable"}}
RULED_OUT = {"Out", "Injured Reserve", "Suspension", "Doubtful"}  # doubtful players rarely play

NFL_TEAMS = {
    "Arizona Cardinals": "ARI", "Atlanta Falcons": "ATL", "Baltimore Ravens": "BAL", "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR", "Chicago Bears": "CHI", "Cincinnati Bengals": "CIN", "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL", "Denver Broncos": "DEN", "Detroit Lions": "DET", "Green Bay Packers": "GB",
    "Houston Texans": "HOU", "Indianapolis Colts": "IND", "Jacksonville Jaguars": "JAX", "Kansas City Chiefs": "KC",
    "Las Vegas Raiders": "LV", "Los Angeles Chargers": "LAC", "Los Angeles Rams": "LA", "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN", "New England Patriots": "NE", "New Orleans Saints": "NO", "New York Giants": "NYG",
    "New York Jets": "NYJ", "Philadelphia Eagles": "PHI", "Pittsburgh Steelers": "PIT", "San Francisco 49ers": "SF",
    "Seattle Seahawks": "SEA", "Tampa Bay Buccaneers": "TB", "Tennessee Titans": "TEN", "Washington Commanders": "WAS",
}


def _fetch_live(sport, team_map, timeout=15):
    req = urllib.request.Request(LIVE_URL[sport], headers={"User-Agent": "Mozilla/5.0 skiiipicks"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        js = json.loads(r.read().decode("utf-8"))
    rows = []
    for t in js.get("injuries", []) or []:
        team = team_map.get(t.get("displayName") or t.get("name") or "")
        for e in t.get("injuries", []) or []:
            a = e.get("athlete") or {}
            pos = (a.get("position") or {}).get("abbreviation") if isinstance(a.get("position"), dict) else a.get("position")
            rows.append({"team": team, "name": a.get("displayName") or a.get("fullName"), "pos": pos,
                         "status": e.get("status") or (e.get("type") or {}).get("description"),
                         "comment": e.get("shortComment") or "", "date": e.get("date")})
    df = pd.DataFrame(rows)
    if len(df) < 10 or df["team"].isna().mean() > 0.5:
        raise ValueError("live injury page returned an unexpected format")
    return df.dropna(subset=["team", "name"])


def _load_daily(path, team_map):
    d = pd.read_parquet(path)
    d = d[d["as_of_date"] == d["as_of_date"].max()]
    return pd.DataFrame({"team": d["team_display_name"].map(team_map), "name": d["athlete_display_name"],
                         "pos": d["athlete_position"], "status": d["status"], "comment": d["short_comment"].fillna(""),
                         "date": d["injury_date"]}).dropna(subset=["team"]), str(d["as_of_date"].max())


def get(sport, daily_path, team_map):
    """Returns (dataframe of non-active players, info dict about the source)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%MZ")
    try:
        df = _fetch_live(sport, team_map)
        info = {"source": "ESPN live injury report", "as_of": now}
    except Exception as e:
        try:
            df, asof = _load_daily(daily_path, team_map)
            info = {"source": "daily injury feed (live page unavailable)", "as_of": asof, "live_error": type(e).__name__}
        except Exception as e2:
            return pd.DataFrame(columns=["team", "name", "pos", "status", "comment"]), {"source": None, "error": type(e2).__name__}
    df = df[df["status"].notna() & (df["status"] != "Active")]
    return df.reset_index(drop=True), info


# ---------- matching ----------

_SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\.?$", re.I)


def _norm(s):
    s = _SUFFIX.sub("", str(s).strip().lower())
    return re.sub(r"[^a-z ]", "", s.replace("-", " ")).strip()


def match(name, team, inj):
    """Find a player's injury row. Handles 'M.Penix' / 'Bi.Robinson' (NFL play-by-play) and full names (NBA)."""
    t = inj[inj["team"] == team]
    if t.empty:
        return None
    if "." in name and " " not in name.split(".", 1)[0]:
        pre, last = name.split(".", 1)
        pre, last = pre.lower(), _norm(last)
        for _, r in t.iterrows():
            parts = _norm(r["name"]).split()
            if len(parts) >= 2 and parts[-1] == last.split()[-1] and parts[0].startswith(pre):
                return r
        return None
    n = _norm(name)
    hit = t[t["name"].map(_norm) == n]
    return hit.iloc[0] if len(hit) else None


def attach(ups, inj, sport, key_players):
    """Add injury lists to each upcoming game, flag key players in notes, and clean up player props.

    key_players: {team: set(names)} for players worth calling out (starters / top usage).
    """
    show = SHOW[sport]
    for u in ups:
        props = u.get("props") or {}
        keys = {u["home"]: set(key_players.get(u["home"], set())), u["away"]: set(key_players.get(u["away"], set()))}
        for side in ("home", "away"):
            keep = []
            for p in props.get(side, []):
                r = match(p["name"], u[side], inj)
                if r is not None:
                    keys[u[side]].add(r["name"])       # anyone with a player prop counts as a key player
                    if r["status"] in RULED_OUT:
                        continue
                    p["status"] = r["status"]
                keep.append(p)
            if side in props:
                props[side] = keep
        u["injuries"] = {}
        for side in ("home", "away"):
            team = u[side]
            t = inj[(inj["team"] == team) & inj["status"].isin(show)]
            u["injuries"][side] = [{"name": r["name"], "pos": r["pos"], "status": r["status"],
                                    "comment": (r["comment"] or "")[:160]} for _, r in t.iterrows()]
            for _, r in t.iterrows():
                if r["name"] in keys[team]:
                    u.setdefault("notes", []).append(f"{team}: {r['name']} ({r['status']})")
    return ups

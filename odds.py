"""Sportsbook odds from The Odds API (https://the-odds-api.com).

Free tier = 500 credits a month. One call costs (markets x regions) = 3 credits per sport here, so by
default odds are fetched once a day (the 8 AM Eastern run) and cached in odds_cache.json; the hourly
runs reuse the cache. Set ODDS_HOURS (UTC hours, comma separated) to fetch more often on a paid plan.

For each game it reports: best moneyline price per side (and the book), the market's no-vig win chance,
the consensus spread and total, and the best spread/total numbers available.
"""
import json
import os
import re
import unicodedata
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from statistics import median
from zoneinfo import ZoneInfo

API = "https://api.the-odds-api.com/v4/sports/{sport}/odds"
SPORT_KEYS = {"nfl": "americanfootball_nfl", "cfb": "americanfootball_ncaaf", "nba": "basketball_nba",
              "mlb": "baseball_mlb"}
ET = ZoneInfo("America/New_York")


# ---------------- odds math ----------------

def implied(american):
    if american is None:
        return None
    a = float(american)
    return 100 / (a + 100) if a > 0 else -a / (-a + 100)


def payout(american):
    a = float(american)
    return a / 100 if a > 0 else 100 / -a


def ev(prob, american):
    """Expected profit per 1 unit staked."""
    return prob * payout(american) - (1 - prob)


# ---------------- fetching with a credit budget ----------------

def fetch(sport, key, timeout=20):
    q = urllib.parse.urlencode({"apiKey": key, "regions": "us", "markets": "h2h,spreads,totals",
                                "oddsFormat": "american", "dateFormat": "iso"})
    req = urllib.request.Request(API.format(sport=SPORT_KEYS[sport]) + "?" + q, headers={"User-Agent": "skiiipicks"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read().decode("utf-8"))
        remaining = r.headers.get("x-requests-remaining")
    return data, remaining


def update_cache(path, key, sports, hours=None, now=None, fetcher=fetch):
    """Refresh the cache for `sports` if this is a fetch hour (or the cache is missing/stale)."""
    now = now or datetime.now(timezone.utc)
    cache = {}
    if path and os.path.exists(path):
        try:
            cache = json.load(open(path))
        except Exception:
            cache = {}
    cache.setdefault("sports", {})
    hours = hours if hours is not None else [int(h) for h in os.environ.get("ODDS_HOURS", "12").split(",") if h.strip()]
    status = {"has_key": bool(key), "fetched": [], "errors": {}}
    if key:
        for sp in sports:
            last = cache["sports"].get(sp, {}).get("fetched_at")
            age_h = (now - datetime.fromisoformat(last)).total_seconds() / 3600 if last else 1e9
            if now.hour in hours or age_h > 26:
                try:
                    data, remaining = fetcher(sp, key)
                    cache["sports"][sp] = {"fetched_at": now.isoformat(), "events": data}
                    cache["remaining"] = remaining
                    status["fetched"].append(sp)
                except Exception as e:
                    status["errors"][sp] = f"{type(e).__name__}: {str(e)[:120]}"
        if path:
            json.dump(cache, open(path, "w"))
    status["remaining"] = cache.get("remaining")
    status["fetched_at"] = {sp: v.get("fetched_at") for sp, v in cache["sports"].items()}
    return cache, status


# ---------------- team matching ----------------

def _norm(s):
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9() ]", "", s.lower().replace("&", "and").replace("-", " ")).strip()


def _nick(full):
    w = _norm(full).split()
    return " ".join(w[-2:]) if w and w[-1] == "sox" else (w[-1] if w else "")


def make_matcher(sport, team_names):
    """team_names: {our_code: full display name} for pro leagues, or a list of our names for college."""
    if sport == "cfb":
        import difflib
        ours = sorted({_norm(t): t for t in team_names}.items(), key=lambda kv: -len(kv[0]))
        keys = [k for k, _ in ours]
        lookup = dict(ours)
        alias = {"southern mississippi": "southern miss", "louisiana monroe": "ul monroe", "ul lafayette": "louisiana",
                 "umass": "massachusetts", "uconn": "connecticut", "miami (fl)": "miami", "app state": "appalachian state",
                 "southern methodist": "smu", "central florida": "ucf", "texas el paso": "utep", "texas san antonio": "utsa",
                 "brigham young": "byu", "louisiana state": "lsu", "mississippi": "ole miss", "nc state": "nc state",
                 "north carolina state": "nc state", "florida intl": "florida international", "fiu": "florida international",
                 "southern california": "usc", "pittsburgh": "pittsburgh", "hawaii": "hawaii"}

        def m(odds_name):
            n = _norm(odds_name)
            for a, b in alias.items():
                if n == a or n.startswith(a + " "):
                    n = b + n[len(a):]
                    break
            for k, t in ours:  # longest school name that prefixes the odds name ("miami (oh)" before "miami")
                if n == k or n.startswith(k + " "):
                    return t
            w = n.split()
            for cut in (1, 2, 3):  # drop the mascot and try a close spelling match
                if len(w) > cut:
                    hit = difflib.get_close_matches(" ".join(w[:-cut]), keys, n=1, cutoff=0.88)
                    if hit:
                        return lookup[hit[0]]
            return None
        return m
    by_nick = {_nick(full): code for code, full in team_names.items()}
    by_full = {_norm(full): code for code, full in team_names.items()}
    return lambda odds_name: by_full.get(_norm(odds_name)) or by_nick.get(_nick(odds_name))


# ---------------- summarizing a game's market ----------------

def summarize_event(ev_, home, away, matcher):
    """Return market info oriented to OUR home/away (the feed may list them the other way round)."""
    books = ev_.get("bookmakers") or []
    flip = matcher(ev_.get("home_team")) == away
    ml = {"home": [], "away": []}
    spreads, totals = [], []
    for b in books:
        title = b.get("title") or b.get("key")
        for mk in b.get("markets") or []:
            outs = mk.get("outcomes") or []
            if mk.get("key") == "h2h":
                for o in outs:
                    side = matcher(o.get("name"))
                    if side in (home, away) and o.get("price") is not None:
                        ml["home" if side == home else "away"].append((o["price"], title))
            elif mk.get("key") == "spreads":
                for o in outs:
                    if matcher(o.get("name")) == home and o.get("point") is not None:
                        spreads.append((-float(o["point"]), o.get("price"), title))  # home expected margin
            elif mk.get("key") == "totals":
                for o in outs:
                    if o.get("point") is not None:
                        totals.append((o.get("name"), float(o["point"]), o.get("price"), title))
    out = {"books": len(books), "flipped": flip}
    if ml["home"] and ml["away"]:
        bh = max(ml["home"], key=lambda x: x[0])
        ba = max(ml["away"], key=lambda x: x[0])
        out["ml_best"] = {"home": bh[0], "home_book": bh[1], "away": ba[0], "away_book": ba[1]}
        # no-vig market chance: median over books of each book's normalized implied probabilities
        pairs = []
        for b in books:
            h = next((o["price"] for mk in b.get("markets", []) if mk.get("key") == "h2h"
                      for o in mk.get("outcomes", []) if matcher(o.get("name")) == home), None)
            a = next((o["price"] for mk in b.get("markets", []) if mk.get("key") == "h2h"
                      for o in mk.get("outcomes", []) if matcher(o.get("name")) == away), None)
            if h is not None and a is not None:
                ih, ia = implied(h), implied(a)
                pairs.append(ih / (ih + ia))
        if pairs:
            out["mkt_home"] = float(median(pairs))
        out["ml_median"] = {"home": float(median(p for p, _ in ml["home"])), "away": float(median(p for p, _ in ml["away"]))}
    if spreads:
        out["spread_home"] = float(median(s[0] for s in spreads))
        best_h = max(spreads, key=lambda s: (-s[0], s[1] or -999))  # home wants the most points (lowest expected margin)
        out["spread_best_home"] = {"line": -best_h[0], "price": best_h[1], "book": best_h[2]}
    overs = [t for t in totals if t[0] == "Over"]
    unders = [t for t in totals if t[0] == "Under"]
    if overs:
        out["total"] = float(median(t[1] for t in overs))
        bo = min(overs, key=lambda t: (t[1], -(t[2] or -999)))
        bu = max(unders, key=lambda t: (t[1], t[2] or -999)) if unders else None
        out["total_best"] = {"over": bo[1], "over_price": bo[2], "over_book": bo[3],
                             **({"under": bu[1], "under_price": bu[2], "under_book": bu[3]} if bu else {})}
    return out


def attach(ups, sport, cache, matcher):
    """Add a `market` block to each upcoming game that has odds. Returns number matched."""
    events = (cache.get("sports", {}).get(sport) or {}).get("events") or []
    idx = {}
    for e in events:
        try:
            d = datetime.fromisoformat(e["commence_time"].replace("Z", "+00:00")).astimezone(ET).strftime("%Y-%m-%d")
        except Exception:
            continue
        h, a = matcher(e.get("home_team")), matcher(e.get("away_team"))
        if h and a:
            idx[(d, h, a)] = e
            idx[(d, a, h)] = e
    n = 0
    for u in ups:
        e = idx.get((u["date"], u["home"], u["away"]))
        if e is None:  # dates can differ by a day for late games across time zones
            for dd in (-1, 1):
                d2 = (datetime.fromisoformat(u["date"]) + timedelta(days=dd)).strftime("%Y-%m-%d")
                e = idx.get((d2, u["home"], u["away"]))
                if e:
                    break
        if e is None:
            continue
        m = summarize_event(e, u["home"], u["away"], matcher)
        if "ml_best" not in m:
            continue
        pick_home = u["win_home"] >= 0.5
        side = "home" if pick_home else "away"
        p_model = u["win_home"] if pick_home else 1 - u["win_home"]
        price = m["ml_best"][side]
        m["pick_price"], m["pick_book"] = price, m["ml_best"][side + "_book"]
        if "mkt_home" in m:
            m["mkt_pick"] = m["mkt_home"] if pick_home else 1 - m["mkt_home"]
            m["edge"] = p_model - m["mkt_pick"]
        m["ev"] = ev(p_model, price)
        # value on the other side? (model thinks the dog is underpriced even if it's not the pick)
        other = "away" if pick_home else "home"
        m["ev_other"] = ev(1 - p_model, m["ml_best"][other])
        u["market"] = m
        if u.get("line_home") is None and "spread_home" in m:
            u["line_home"] = m["spread_home"]
        if u.get("total_line") is None and "total" in m:
            u["total_line"] = m["total"]
        n += 1
    return n


def value_rating(ev_):
    return None if ev_ is None else ("green" if ev_ >= 0.05 else "yellow" if ev_ >= 0 else "red")

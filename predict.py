"""Quick command-line projection from a built dashboard_data.json.

    python -m skiiipicks.predict nfl KC BUF            # away first, then home
    python -m skiiipicks.predict cfb "Texas" "Ohio State" --neutral
"""
import argparse
import json
import math


def project(d, sport, home, away, neutral=False):
    s, m = d[sport], d[sport]["model"]
    loc = 0 if neutral else 0.5
    g = lambda k, t: (m.get(k) or {}).get(t, 0) or 0
    ph = m["mu"] + g("off", home) - g("dfn", away) + m["home"] * loc
    pa = m["mu"] + g("off", away) - g("dfn", home) - m["home"] * loc
    if sport == "nba":
        pace = (2 * m["pace_mu"] + g("pace_off", home) - g("pace_dfn", away) + g("pace_off", away) - g("pace_dfn", home)) / 2
        ph, pa = ph * pace / 100, pa * pace / 100
    if sport == "nfl" and m.get("eff_off"):
        em = (g("eff_off", home) - g("eff_dfn", away) + m["eff_home"] * loc) - (g("eff_off", away) - g("eff_dfn", home) - m["eff_home"] * loc)
        mg, tot = m["w"] * (ph - pa) + (1 - m["w"]) * m["scale"] * em, ph + pa
        ph, pa = tot / 2 + mg / 2, tot / 2 - mg / 2
    wp = 0.5 * (1 + math.erf((ph - pa) / (s["sigma"] * math.sqrt(2))))
    return ph, pa, wp


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("sport", choices=["nfl", "cfb", "nba", "mlb"])
    ap.add_argument("away"); ap.add_argument("home")
    ap.add_argument("--neutral", action="store_true")
    ap.add_argument("--data", default="dashboard_data.json")
    a = ap.parse_args()
    d = json.load(open(a.data))
    teams = {t["team"] for t in d[a.sport]["teams"]}
    for t in (a.home, a.away):
        if t not in teams:
            raise SystemExit(f"Unknown team '{t}'. Options: {', '.join(sorted(teams))}")
    ph, pa, wp = project(d, a.sport, a.home, a.away, a.neutral)
    fav = a.home if ph >= pa else a.away
    print(f"{a.away} {pa:.1f}  at  {a.home} {ph:.1f}")
    print(f"Spread: {fav} -{abs(ph - pa):.1f}   Total: {ph + pa:.1f}   {a.home} win chance: {wp:.0%}")

"""#2184 review finding 1: at short time limits, does DISCOPT_GDP_CONFIG_ROOT_SKI lose
incumbents? Scores issue1917_gdp_flag_panel.py solve rows, OFF vs ON per cell.

    python score.py gdp_tl10.jsonl [gdp_tl20.jsonl ...]
"""

import json
import sys
from collections import defaultdict

REL = 1e-4
n = 0
for path in sys.argv[1:]:
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    rows = [r for r in rows if r.get("phase") == "solve"]
    cells = defaultdict(dict)
    for r in rows:
        cells[(r["instance"], r["method"])][r["arm"]] = r
    lost, gained, better, worse, same, solved_lost, solved_gained = [], [], [], [], 0, [], []
    for c, a in sorted(cells.items()):
        if set(a) != {"off", "on"}:
            continue
        n += 1
        o, w = a["off"], a["on"]
        io, iw = o.get("objective") is not None, w.get("objective") is not None
        if o.get("verdict") == "solved" and w.get("verdict") != "solved":
            solved_lost.append(c)
        if w.get("verdict") == "solved" and o.get("verdict") != "solved":
            solved_gained.append(c)
        if io and not iw:
            lost.append(c)
        elif iw and not io:
            gained.append(c)
        elif io and iw:
            mn = o.get("minimize", True)
            fo, fw = o["objective"], w["objective"]
            tol = REL * max(1.0, abs(fo))
            if (fw < fo - tol) if mn else (fw > fo + tol):
                better.append((c, fo, fw))
            elif (fw > fo + tol) if mn else (fw < fo - tol):
                worse.append((c, fo, fw))
            else:
                same += 1
    tl = rows[0].get("time_limit") if rows else None
    print(f"== {path}: time_limit={tl}, {len(cells)} cells")
    print(f"  incumbent lost ON: {len(lost)} {lost}")
    print(f"  incumbent gained ON: {len(gained)} {gained}")
    print(f"  both have one: same {same}, ON better {len(better)}, ON worse {len(worse)}")
    for x in better + worse:
        print("    ", x)
    print(f"  solved lost ON: {solved_lost}; solved gained ON: {solved_gained}")
    inc = [r for r in rows if r.get("verdict") == "incorrect"]
    print(f"  incorrect rows: {len(inc)}")
print(f"executed comparisons: {n}")
assert n > 0

"""#917 entry experiment driver.

For every in-scope instance in the in-repo corpus, run the worker in an isolated
subprocess and classify:

  * ``certified``      -- stage A certified inside 65% of the budget; no waste.
  * ``WASTE``          -- stage A hit its reduced deadline WITH an incumbent: today
                          the 35% reserve is forfeited. This is #917's class.
  * ``reserve-spent``  -- stage A found no incumbent; today's code already spends the
                          reserve here (#844).

For the WASTE rows it reports whether the reserve's dual bound would tighten the
merged certificate -- the issue's kill criterion for candidate 1.

Ends with an executed-comparison count and exits non-zero if nothing was measured
(CLAUDE.md §6).
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPORA = [ROOT / "python/tests/data/minlplib_nl", ROOT / "python/tests/data/minlplib"]


def find(name):
    for d in CORPORA:
        p = d / f"{name}.nl"
        if p.exists():
            return p
    raise SystemExit(f"instance not found: {name}")
WORKER = Path(__file__).with_name("issue917_worker.py")

INSTANCES = sys.argv[2:] or [
    "nvs03", "nvs07", "nvs10", "nvs11", "nvs12", "nvs13", "nvs15",
    "nvs17", "nvs18", "nvs19", "nvs23", "nvs24", "prob02", "prob03",
    "st_miqp1", "st_miqp2", "st_miqp3", "st_test1", "st_testgr3",
]
T = float(sys.argv[1]) if len(sys.argv) > 1 else 20.0

TIGHTER_EPS = 1e-6

rows = []
compared = 0
for name in INSTANCES:
    nl = find(name)
    print(f"--- {name} (T={T}) ...", flush=True)
    proc = subprocess.run(
        [sys.executable, "-u", str(WORKER), str(nl), str(T)],
        capture_output=True,
        text=True,
        timeout=6 * T + 600,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-4000:], file=sys.stderr)
        raise SystemExit(f"{name}: worker exited {proc.returncode}")
    row = json.loads(proc.stdout.strip().splitlines()[-1])

    if row["primary_gap_certified"]:
        cls = "certified"
    elif row["primary_objective"] is None:
        cls = "reserve-spent"
    else:
        cls = "WASTE"
    row["class"] = cls

    pb, rb = row["primary_bound"], row["reserve_bound"]
    if pb is None or rb is None:
        row["bound_tighter"] = None
        row["merged_bound"] = pb if rb is None else rb
    else:
        compared += 1
        if row["sense"] == "max":
            row["merged_bound"] = min(pb, rb)
            row["bound_tighter"] = rb < pb - TIGHTER_EPS
        else:
            row["merged_bound"] = max(pb, rb)
            row["bound_tighter"] = rb > pb + TIGHTER_EPS
    rows.append(row)
    print(
        f"    {cls:13s} primary(status={row['primary_status']}, "
        f"obj={row['primary_objective']}, bound={row['primary_bound']}, "
        f"wall={row['primary_wall']:.1f}) | reserve(status={row['reserve_status']}, "
        f"obj={row['reserve_objective']}, bound={row['reserve_bound']}, "
        f"wall={row['reserve_wall']:.1f}) | tighter={row['bound_tighter']}",
        flush=True,
    )

waste = [r for r in rows if r["class"] == "WASTE"]
print()
print(f"instances       : {len(rows)}")
print(f"  certified     : {sum(1 for r in rows if r['class'] == 'certified')}")
print(f"  reserve-spent : {sum(1 for r in rows if r['class'] == 'reserve-spent')}")
print(f"  WASTE (#917)  : {len(waste)}")
print(f"  of those, reserve bound tighter: {sum(1 for r in waste if r['bound_tighter'])}")
print(f"BOUND_COMPARISONS={compared}")

out = Path(__file__).with_name(f"issue917_entry_panel_T{int(T)}.json")
out.write_text(json.dumps(rows, indent=2))
print(f"wrote {out}")

if compared == 0:
    print("PROBE FIRED NOTHING: zero bound comparisons executed", file=sys.stderr)
    sys.exit(1)

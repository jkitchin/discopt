"""#1017 scout: which vendored instances actually reach the Farkas certificate?

The first cut of the A/B panel measured *nothing* — every Farkas counter was zero
on all 16 instances, because (a) the counters need `DISCOPT_PROFILE` and (b) the
MILP-ish instances in that list never produce an infeasible node LP. This finds the
instances that do, so the panel is not a no-op that reads as a pass (CLAUDE.md §6).

Prints one line per instance and a final count of Farkas-active instances; exits
non-zero if none is found.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
T = float(sys.argv[1]) if len(sys.argv) > 1 else 15.0

CANDIDATES = [
    ("kall_circles_c8a", "minlplib"),
    ("gear2", "minlplib"),
    ("gear3", "minlplib"),
    ("gear4", "minlplib"),
    ("nvs17", "minlplib"),
    ("nvs20", "minlplib"),
    ("nvs24", "minlplib"),
    ("ex8_1_1", "minlplib"),
    ("ex8_5_4", "minlplib"),
    ("ex1252a", "minlplib"),
    ("ex1263", "minlplib"),
    ("ex1264", "minlplib"),
    ("ex1266", "minlplib"),
    ("st_e07", "minlplib"),
    ("st_e15", "minlplib"),
    ("st_e31", "minlplib"),
    ("st_ph10", "minlplib"),
    ("meanvarx", "minlplib"),
    ("prob06", "minlplib"),
    ("prob10", "minlplib"),
    ("util", "minlplib"),
    ("trig", "minlplib"),
    ("mathopt3", "minlplib"),
    ("casctanks", "minlplib_nl"),
    ("tspn05", "minlplib_nl"),
    ("tspn08", "minlplib_nl"),
    ("clay0303hfsg", "minlplib_nl"),
    ("flay02m", "minlplib_nl"),
    ("flay03m", "minlplib_nl"),
    ("syn05hfsg", "minlplib_nl"),
    ("heatexch_gen1", "minlplib_nl"),
    ("ex1224", "minlplib_nl"),
    ("ex1225", "minlplib_nl"),
    ("ex1226", "minlplib_nl"),
    ("ex14_1_9", "minlplib_nl"),
    ("st_e29", "minlplib_nl"),
    ("st_e36", "minlplib_nl"),
    ("st_e38", "minlplib_nl"),
    ("gkocis", "minlplib_nl"),
    ("m3", "minlplib_nl"),
    ("fac2", "minlplib_nl"),
    ("oaer", "minlplib_nl"),
    ("tanksize", "minlplib_nl"),
    ("tls2", "minlplib_nl"),
    ("4stufen", "minlplib_nl"),
    ("beuster", "minlplib_nl"),
    ("contvar", "minlplib_nl"),
    ("hda", "minlplib_nl"),
]

SNIPPET = """
import json, os, sys, time
os.environ.setdefault("JAX_PLATFORMS", "cpu")
from discopt._rust import profile_counters_py, profile_reset_py
from discopt.modeling.core import from_nl
profile_reset_py()
m = from_nl(sys.argv[1])
t0 = time.perf_counter()
r = m.solve(time_limit=float(sys.argv[2]))
c = profile_counters_py()
print(json.dumps({
    "wall": time.perf_counter() - t0,
    "status": r.status,
    "objective": r.objective,
    "bound": r.bound,
    "nodes": int(r.node_count or 0),
    "cert": bool(r.gap_certified),
    "rej_margin": int(c.get("FarkasRejectMargin", 0)),
    "rej_open": int(c.get("FarkasRejectOpen", 0)),
    "rej_cancel": int(c.get("FarkasRejectCancellation", 0)),
    "lp_infeas": int(c.get("LpVerdictInfeasible", 0)),
    "warm_infeas": int(c.get("WarmVerdictInfeasible", 0)),
}))
"""

active, done = [], 0
for name, corpus in CANDIDATES:
    nl = ROOT / f"python/tests/data/{corpus}/{name}.nl"
    if not nl.exists():
        print(f"{name}: MISSING", flush=True)
        continue
    p = subprocess.run(
        [sys.executable, "-u", "-c", SNIPPET, str(nl), str(T)],
        capture_output=True,
        text=True,
        timeout=10 * T + 600,
        env={**os.environ, "DISCOPT_PROFILE": "1", "DISCOPT_LP_WARM_DEADLINE": "0"},
        cwd="/tmp",
    )
    if p.returncode != 0:
        print(f"{name}: FAILED {p.stderr.strip().splitlines()[-1][:160]}", flush=True)
        continue
    row = json.loads(p.stdout.strip().splitlines()[-1])
    done += 1
    hits = row["rej_margin"] + row["rej_open"] + row["lp_infeas"] + row["warm_infeas"]
    if hits:
        active.append((name, corpus, row))
    print(
        f"{name}: {row['status']} nodes={row['nodes']} cert={row['cert']} "
        f"wall={row['wall']:.1f}s farkas(margin={row['rej_margin']} open={row['rej_open']} "
        f"cancel={row['rej_cancel']}) infeas(cold={row['lp_infeas']} warm={row['warm_infeas']})",
        flush=True,
    )

print(f"\ninstances run: {done}; Farkas-active: {len(active)}")
for name, corpus, row in active:
    print(f"  ACTIVE {name} ({corpus}) cert={row['cert']} status={row['status']}")
if done == 0:
    print("SCOUT MEASURED NOTHING")
    sys.exit(1)
if not active:
    print("NO FARKAS-ACTIVE INSTANCE FOUND — the panel would be a no-op")
    sys.exit(2)

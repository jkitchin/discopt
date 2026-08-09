"""#928 flag-OFF bound-neutrality check (CLAUDE.md §5, regime 1).

With ``DISCOPT_LP_WARM_DEADLINE`` OFF (the default) the #928 change must be a
no-op: ``bank_deadline_duals`` is false on every constructed ``SimplexOptions``
except the pure-LP warm binding's, and that one only when its caller passed a
``time_limit`` — which the flag-OFF relaxation path never does. Assert exact
equality of ``node_count`` and the certified ``objective``/``bound`` on instances
that CERTIFY inside the budget (the only budget-independent, tree-comparable
results; same instance list as the #917 check this is adapted from).

Run once on the baseline tree with ``--write`` (marker absent), then on the
changed tree with ``--check`` (marker present).
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("issue928_neutral_worker.py")

CERTIFYING = [
    ("nvs03", "minlplib_nl"),
    ("nvs07", "minlplib_nl"),
    ("nvs10", "minlplib_nl"),
    ("nvs11", "minlplib_nl"),
    ("nvs12", "minlplib_nl"),
    ("nvs15", "minlplib_nl"),
    ("prob02", "minlplib"),
    ("prob03", "minlplib"),
    ("st_miqp1", "minlplib_nl"),
    ("st_miqp2", "minlplib_nl"),
    ("st_miqp3", "minlplib_nl"),
    ("st_test1", "minlplib_nl"),
    ("st_testgr3", "minlplib_nl"),
]
T = 20.0

ap = argparse.ArgumentParser()
ap.add_argument("--write", action="store_true")
ap.add_argument("--check", action="store_true")
ap.add_argument("--store", default=str(Path(__file__).with_name("issue928_neutrality.json")))
ap.add_argument("--expect-marker", choices=["0", "1"], required=True)
args = ap.parse_args()

rows = {}
for name, corpus in CERTIFYING:
    nl = ROOT / f"python/tests/data/{corpus}/{name}.nl"
    proc = subprocess.run(
        [sys.executable, "-u", str(WORKER), str(nl), str(T), args.expect_marker],
        capture_output=True,
        text=True,
        timeout=10 * T + 600,
        env={**os.environ, "DISCOPT_LP_WARM_DEADLINE": "0"},
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-2000:] + proc.stderr[-4000:])
        raise SystemExit(f"worker failed on {name}")
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    rows[name] = {
        "status": r["status"],
        "objective": r["objective"],
        "bound": r["bound"],
        "node_count": r["node_count"],
        "gap_certified": r["gap_certified"],
    }
    print(f"{name:11s} {rows[name]}", flush=True)

store = Path(args.store)
if args.write:
    store.write_text(json.dumps(rows, indent=2))
    print(f"wrote baseline: {store}")
    raise SystemExit(0)

if args.check:
    base = json.loads(store.read_text())
    compared = 0
    drift = []
    for name, cur in rows.items():
        ref = base.get(name)
        if ref is None:
            drift.append(f"{name}: absent from baseline")
            continue
        compared += 1
        if ref != cur:
            drift.append(f"{name}: baseline={ref} current={cur}")
    print(f"\nCOMPARED={compared}")
    for d in drift:
        print("DRIFT:", d)
    if compared == 0:
        print("PROBE FIRED NOTHING", file=sys.stderr)
        raise SystemExit(1)
    print("BOUND_NEUTRAL=" + str(not drift))
    raise SystemExit(1 if drift else 0)

raise SystemExit("pass --write or --check")

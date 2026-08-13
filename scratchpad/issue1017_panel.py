"""#1017 Farkas-margin A/B panel: the cost side of tightening the certificate.

The change is *monotone* by construction — the margin is `max(legacy, rigorous)`, so
it can only ever WITHDRAW a certificate, never issue a new one. Its whole risk is
therefore lost fathoming: a node LP that used to return `Infeasible` returning the
honest, non-fathoming `Numerical` instead, costing nodes.

This panel measures that on the vendored corpus, per CLAUDE.md §5's bound-neutral
regime: `node_count` and the certified `objective`/`bound` on instances that CERTIFY
inside the budget, plus the new `FarkasRejectCancellation` counter, which counts
exactly the certificates the change removed.

Run on the baseline tree with `--write --expect-marker 0`, then on the changed tree
with `--check --expect-marker 1`.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKER = Path(__file__).with_name("issue1017_panel_worker.py")

# The #966/#917 certifying list (same instances, so the two panels are comparable),
# extended with the corpus instances whose relaxations are the ones that reach the
# Farkas path most often (bchoco*, gear*, st_e*).
PANEL = [
    # Farkas-ACTIVE and terminating, per `issue1017_scout.py` (baseline, T=15): each
    # of these certifies inside the budget *and* reaches the certificate — a panel of
    # instances that never touch it would measure nothing and read as a pass (§6).
    ("gear4", "minlplib"),  # 2 cold + 2 warm infeasible verdicts
    ("ex8_1_1", "minlplib"),  # 1 warm infeasible, 1 margin reject
    ("ex1263", "minlplib"),  # 12 cold infeasible, 2134 margin rejects
    ("ex1264", "minlplib"),  # 14 cold infeasible, 2621 margin rejects
    ("ex1266", "minlplib"),  # 16 cold infeasible, 919 margin rejects
    ("st_e31", "minlplib"),  # 1 cold infeasible
    ("prob10", "minlplib"),  # 11 cold + 12 warm infeasible
    ("util", "minlplib"),  # 98 cold infeasible
    ("trig", "minlplib"),  # 8 cold + 8 warm infeasible
    ("tspn05", "minlplib_nl"),  # 2 margin rejects
    ("st_e36", "minlplib_nl"),  # 58 cold + 58 warm infeasible, 6 margin rejects
    ("st_e38", "minlplib_nl"),  # 15 cold + 15 warm infeasible
    ("ex14_1_9", "minlplib_nl"),  # 8 margin rejects
    # Farkas-inactive controls: these must be bit-identical no matter what.
    ("nvs12", "minlplib_nl"),
    ("st_testgr3", "minlplib_nl"),
    ("prob02", "minlplib"),
]
T = 60.0

ap = argparse.ArgumentParser()
ap.add_argument("--write", action="store_true")
ap.add_argument("--check", action="store_true")
ap.add_argument("--store", default=str(Path(__file__).with_name("issue1017_panel.json")))
ap.add_argument("--expect-marker", choices=["0", "1"], required=True)
args = ap.parse_args()

rows = {}
for name, corpus in PANEL:
    nl = ROOT / f"python/tests/data/{corpus}/{name}.nl"
    if not nl.exists():
        print(f"{name}: MISSING {nl}", flush=True)
        continue
    proc = subprocess.run(
        [sys.executable, "-u", str(WORKER), str(nl), str(T), args.expect_marker],
        capture_output=True,
        text=True,
        timeout=10 * T + 600,
        # The counters are no-ops unless profiling is on — the first cut of this
        # panel reported every Farkas counter as 0 for exactly that reason.
        env={**os.environ, "DISCOPT_PROFILE": "1", "DISCOPT_LP_WARM_DEADLINE": "0"},
    )
    if proc.returncode != 0:
        print(f"{name}: WORKER FAILED rc={proc.returncode}\n{proc.stderr[-2000:]}", flush=True)
        sys.exit(1)
    row = json.loads(proc.stdout.strip().splitlines()[-1])
    rows[name] = row
    print(
        f"{name}: {row['status']} obj={row['objective']} bound={row['bound']} "
        f"nodes={row['node_count']} cert={row['gap_certified']} "
        f"farkas_margin={row['farkas_reject_margin']} "
        f"cancellation={row['farkas_reject_cancellation']} wall={row['wall']:.1f}s",
        flush=True,
    )

if not rows:
    print("PANEL MEASURED NOTHING")
    sys.exit(1)

if args.write:
    Path(args.store).write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} rows to {args.store}")
    sys.exit(0)

base = json.loads(Path(args.store).read_text())
compared = 0
diffs = []
for name, row in rows.items():
    if name not in base:
        diffs.append(f"{name}: absent from the baseline store")
        continue
    b = base[name]
    compared += 1
    if row["status"] != b["status"]:
        diffs.append(f"{name}: status {b['status']} -> {row['status']}")
    if row["gap_certified"] != b["gap_certified"]:
        diffs.append(f"{name}: gap_certified {b['gap_certified']} -> {row['gap_certified']}")
    if row["node_count"] != b["node_count"]:
        diffs.append(f"{name}: node_count {b['node_count']} -> {row['node_count']}")
    for key in ("objective", "bound"):
        x, y = b[key], row[key]
        if (x is None) != (y is None):
            diffs.append(f"{name}: {key} {x} -> {y}")
        elif x is not None and abs(x - y) > 1e-9 * (1.0 + abs(x)):
            diffs.append(f"{name}: {key} {x} -> {y}")

# CLAUDE.md §6: the comparison count is printed and a zero-comparison run FAILS —
# and so does a run in which the panel never reached the certificate at all.
print(f"\ncomparisons executed: {compared}")
print(f"{'instance':<18}{'infeas cold/warm':>22}{'rej margin/open':>20}{'withdrawn':>12}")
touched = 0
for name, row in rows.items():
    b = base.get(name, {})
    touched += row["lp_infeasible"] + row["warm_infeasible"]
    print(
        f"{name:<18}"
        f"{b.get('lp_infeasible', '?')}/{b.get('warm_infeasible', '?')}"
        f" -> {row['lp_infeasible']}/{row['warm_infeasible']:<8}"
        f"{b.get('farkas_reject_margin', '?')}/{b.get('farkas_reject_open', '?')}"
        f" -> {row['farkas_reject_margin']}/{row['farkas_reject_open']:<6}"
        f"{row['farkas_reject_cancellation']:>10}"
    )
removed = sum(r["farkas_reject_cancellation"] for r in rows.values())
print(f"\nsuccessful Farkas certifications on this panel: {touched}")
print(f"certificates withdrawn by #1017 across the panel: {removed}")
if compared == 0:
    print("PANEL COMPARED NOTHING")
    sys.exit(1)
if touched == 0:
    print("PANEL NEVER REACHED THE FARKAS CERTIFICATE — it measures nothing")
    sys.exit(1)
if diffs:
    print("DIFFS:")
    for d in diffs:
        print(" ", d)
    sys.exit(1)
print("IDENTICAL: status, gap_certified, node_count, objective and bound all match")

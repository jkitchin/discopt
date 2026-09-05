"""#1154 panel, arm 2c — CERT-CLEAN scoring of the corpus arm against the oracle.

The A/B differential (arm 2) shows OFF == ON; that says the flag changes nothing,
not that what it does not change is sound. CLAUDE.md §5 also wants the ON arm
scored against the reference optimum: no dual bound above its oracle (for a MIN
instance; below, for a MAX one), and no ``optimal`` certificate whose objective
disagrees with the oracle.

Uses ``discopt_benchmarks.utils.reference_optima`` (the accessor that resolves
through ``known_optima.toml`` / ``cert-optima.json`` when no ``.solu`` snapshot is
installed, so it scores in CI too), and takes the sense from the loaded MODEL --
5 vendored instances are MAXIMIZE and assuming min manufactures false violations.

Prints per-instance progress (§10), and both a scored count and an unscored count
so a table that resolved no oracles cannot read as "no violations" (§6).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from discopt_benchmarks.utils.reference_optima import reference_oracle  # noqa: E402

CORPUS = Path("python/tests/data/minlplib_nl")
TIME_LIMIT = 10.0

CHILD = r'''
import json, os, sys
from discopt.modeling.core import from_nl
m = from_nl(sys.argv[1])
sense = str(getattr(m._objective, "sense", "min")).lower()
r = m.solve(time_limit=float(os.environ["PANEL_TL"]), deterministic=True)
print("RESULT" + json.dumps({
    "sense": sense,
    "status": str(r.status),
    "objective": None if r.objective is None else float(r.objective),
    "bound": None if r.bound is None else float(r.bound),
}))
'''


def run(path: Path) -> dict:
    env = dict(os.environ, DISCOPT_GDP_SUMOVER="1", PANEL_TL=str(TIME_LIMIT))
    proc = subprocess.run(
        [sys.executable, "-c", CHILD, str(path)],
        capture_output=True, text=True, env=env, timeout=TIME_LIMIT * 20 + 120,
    )
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT"):])
    raise RuntimeError(f"no RESULT for {path.name}: {(proc.stderr or proc.stdout)[-400:]}")


scored = 0
unscored: list[str] = []
unproven: list[str] = []
bound_violations: list[str] = []
primal_violations: list[str] = []

for path in sorted(CORPUS.glob("*.nl")):
    name = path.stem
    res = run(path)
    oracle = reference_oracle(name)
    if oracle is None:
        unscored.append(name)
        print(f"  {name}: UNSCORED (no oracle) status={res['status']}", flush=True)
        continue
    if not oracle.proven:
        unproven.append(name)
        print(f"  {name}: oracle is =best= (unproven), soundness not gated", flush=True)
        continue
    scored += 1
    ref = oracle.value
    tol = 1e-5 * max(1.0, abs(ref))
    is_min = "min" in res["sense"]
    verdict = "ok"
    if res["bound"] is not None:
        if is_min and res["bound"] > ref + tol:
            bound_violations.append(f"{name}: min dual bound {res['bound']} > oracle {ref}")
            verdict = "BOUND VIOLATION"
        if not is_min and res["bound"] < ref - tol:
            bound_violations.append(f"{name}: max dual bound {res['bound']} < oracle {ref}")
            verdict = "BOUND VIOLATION"
    if res["objective"] is not None:
        if is_min and res["objective"] < ref - tol:
            primal_violations.append(f"{name}: min incumbent {res['objective']} < oracle {ref}")
            verdict = "PRIMAL VIOLATION"
        if not is_min and res["objective"] > ref + tol:
            primal_violations.append(f"{name}: max incumbent {res['objective']} > oracle {ref}")
            verdict = "PRIMAL VIOLATION"
    print(
        f"  {name}: {verdict} sense={res['sense']} status={res['status']} "
        f"obj={res['objective']} bound={res['bound']} oracle={ref} ({oracle.source})",
        flush=True,
    )

print()
print(f"instances_scored={scored}")
print(f"instances_unscored={len(unscored)} {unscored}")
print(f"instances_with_unproven_oracle={len(unproven)} {unproven}")
print(f"bound_violations={len(bound_violations)}")
for line in bound_violations:
    print("  BOUND VIOLATION", line)
print(f"primal_violations={len(primal_violations)}")
for line in primal_violations:
    print("  PRIMAL VIOLATION", line)
print(f"executed_comparisons={scored}")
if scored == 0:
    print("PROBE DID NOT FIRE: no oracle resolved for any instance", file=sys.stderr)
    sys.exit(1)
sys.exit(1 if (bound_violations or primal_violations) else 0)

"""Census: what a substitution<->bound-tightening round loop could reach (#844).

This is the **entry experiment** for coupling `substitute_variables` to the existing
FBBT (CLAUDE.md §4: run the falsifying experiment on real corpus instances *before*
writing the implementation).

## The mechanism under test

SCIP's presolve alternates handlers across rounds: propagate bounds, fix variables
whose domain collapsed, aggregate, propagate again. We already have both halves —
`substitute_variables` (#844/#888) and `fbbt_fixed_point` — but they never run against
each other. This measures the ceiling of coupling them.

## What is measured, per instance

1. `substitute(4)` to its own fixpoint — the shipped behaviour.
2. `fbbt_fixed_point` on the *reduced* model, read to a fixed point.
3. How many **continuous** blocks that were not already points have collapsed to a
   point, at four widths: exactly 0, <=1e-12, <=1e-9, <=1e-6.

Step 3 is the only thing a sound round loop may act on. Note the constraint this
operates under: `presolve/orchestrator.rs:132-147` deliberately does **not** write
FBBT-tightened bounds back into the returned model, for stated correctness reasons
(an inactive bound flipped active changes LP duals; a cutoff-derived tightening can
manufacture a false infeasibility on re-solve). A round loop must therefore use the
tightened box *only* to detect exact domain collapse and fix, never to install general
tightened bounds. The width histogram says whether that restriction costs anything.

## Executed-assertion discipline

Prints the number of instances examined and exits non-zero at zero (§6). Every
instance runs in its own subprocess so one pathological FBBT cannot stall the census,
and a crash is recorded as data rather than swallowed (§7).

Wall times here are NOT timing claims — the census runs unshielded against other load.
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "discopt_benchmarks"))

from utils.corpus import corpus_is_synced, nl_dir  # noqa: E402

SEED = int(os.environ.get("CENSUS_SEED", "20260727"))
N = int(os.environ.get("CENSUS_N", "300"))
FBBT_MS = int(os.environ.get("CENSUS_FBBT_MS", "20000"))
CHILD_TIMEOUT_S = float(os.environ.get("CENSUS_CHILD_TIMEOUT", "180"))

CHILD = r"""
import json, sys, time
import numpy as np
import discopt, discopt.modeling as dm
from discopt._rust import model_to_repr

path, fbbt_ms, repo = sys.argv[1], int(sys.argv[2]), sys.argv[3]
# CLAUDE.md §8: assert WHICH code is loaded, and a marker unique to the version.
assert discopt.__file__.startswith(repo), discopt.__file__
import discopt._rust as R
assert hasattr(R, "SubstitutionChain"), "marker absent"

t0 = time.time()
m = dm.from_nl(path)
rep = model_to_repr(m, getattr(m, "_builder", None))
n0 = rep.n_vars
red, chain = rep.substitute(4)
t_sub = time.time() - t0

types = red.var_types()
lo0 = np.array([red.var_lb(i)[0] for i in range(red.n_var_blocks)], dtype=float)
hi0 = np.array([red.var_ub(i)[0] for i in range(red.n_var_blocks)], dtype=float)

t1 = time.time()
_out, st = red.presolve(passes=["fbbt_fixed_point"], max_iterations=1, time_limit_ms=fbbt_ms)
t_fbbt = time.time() - t1
lo1 = np.asarray(st["bounds_lo"], dtype=float)
hi1 = np.asarray(st["bounds_hi"], dtype=float)
assert lo1.shape == lo0.shape, (lo1.shape, lo0.shape)

cont = np.array([t == "continuous" for t in types])
was_point = (hi0 - lo0) == 0.0
w = hi1 - lo1
fin = np.isfinite(w)
base = cont & ~was_point & fin
out = {
    "n_vars": int(n0),
    "n_after_subst": int(red.n_vars),
    "blocks": int(red.n_var_blocks),
    "subst_sweeps": int(chain.n_sweeps),
    "collapse_exact": int(np.sum(base & (w == 0.0))),
    "collapse_1e12": int(np.sum(base & (w > 0.0) & (w <= 1e-12))),
    "collapse_1e9": int(np.sum(base & (w > 0.0) & (w <= 1e-9))),
    "collapse_1e6": int(np.sum(base & (w > 0.0) & (w <= 1e-6))),
    "bounds_tightened": int(np.sum(fin & ((lo1 > lo0 + 1e-12) | (hi1 < hi0 - 1e-12)))),
    "fbbt_terminated_by": str(st["terminated_by"]),
    "t_subst": t_sub,
    "t_fbbt": t_fbbt,
}
print("JSONRESULT " + json.dumps(out))
"""


def run_one(path: Path) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "python")
    try:
        p = subprocess.run(
            [sys.executable, "-u", "-c", CHILD, str(path), str(FBBT_MS), str(REPO)],
            capture_output=True,
            text=True,
            env=env,
            timeout=CHILD_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT"}
    for line in p.stdout.splitlines():
        if line.startswith("JSONRESULT "):
            d = json.loads(line[len("JSONRESULT ") :])
            d["status"] = "ok"
            return d
    return {"status": "CRASH", "stderr": p.stderr.strip()[-400:]}


def main() -> int:
    d = nl_dir()
    if d is None:
        print("FATAL: corpus did not resolve", file=sys.stderr)
        return 2
    if corpus_is_synced():
        print(f"WARNING: corpus resolves into a synced folder ({d})", flush=True)
    names = sorted(p.stem for p in d.glob("*.nl"))
    print(f"population: {len(names)} .nl instances at {d}", flush=True)
    sample = random.Random(SEED).sample(names, min(N, len(names)))
    print(f"sample: {len(sample)} instances, seed {SEED}", flush=True)

    results: dict[str, dict] = {}
    for i, name in enumerate(sample, 1):
        t0 = time.time()
        r = run_one(d / f"{name}.nl")
        results[name] = r
        if r["status"] == "ok":
            print(
                f"[{i:3d}/{len(sample)}] {name:32s} vars {r['n_vars']:7d}->{r['n_after_subst']:7d} "
                f"exact={r['collapse_exact']:5d} le1e-9={r['collapse_1e9']:4d} "
                f"tight={r['bounds_tightened']:6d} {time.time() - t0:6.1f}s",
                flush=True,
            )
        else:
            print(f"[{i:3d}/{len(sample)}] {name:32s} {r['status']}", flush=True)

    out_path = REPO / "presolve_roundloop_census.json"
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"\nwrote {out_path}")

    ok = [r for r in results.values() if r["status"] == "ok"]
    print(f"\nINSTANCES EXAMINED: {len(ok)} (of {len(sample)} sampled)")
    if not ok:
        print("CENSUS EXECUTED NOTHING")
        return 2

    def pct(pred) -> tuple[int, float]:
        n = sum(1 for r in ok if pred(r))
        return n, 100.0 * n / len(ok)

    subst = lambda r: r["n_after_subst"] < r["n_vars"]  # noqa: E731
    coll = lambda r: r["collapse_exact"] > 0  # noqa: E731

    for label, pred in (
        ("substitution reduces (shipped)", subst),
        ("FBBT tightens any bound", lambda r: r["bounds_tightened"] > 0),
        (">=1 EXACT new point-collapse", coll),
        (">=1 collapse within 1e-12", lambda r: r["collapse_exact"] + r["collapse_1e12"] > 0),
        (">=1 collapse within 1e-9", lambda r: r["collapse_exact"] + r["collapse_1e9"] > 0),
        (">=1 collapse within 1e-6", lambda r: r["collapse_exact"] + r["collapse_1e6"] > 0),
        ("CEILING: subst OR exact collapse", lambda r: subst(r) or coll(r)),
    ):
        n, p = pct(pred)
        print(f"  {label:36s} {n:4d}/{len(ok)}  {p:5.1f}%")

    n_ceiling, p_ceiling = pct(lambda r: subst(r) or coll(r))
    n_sub, p_sub = pct(subst)
    print(
        f"\n  collapse but NOT already reduced: "
        f"{sum(1 for r in ok if coll(r) and not subst(r))} instances "
        f"(the only ones a round loop can newly touch)"
    )
    print(f"  total blocks collapsed exactly: {sum(r['collapse_exact'] for r in ok)}")
    print(f"  total blocks collapsed >0 and <=1e-6: {sum(r['collapse_1e6'] for r in ok)}")
    print(
        f"\nKILL CRITERION: coupled loop must raise 'any reduction' to >= 45.0%. "
        f"Shipped = {p_sub:.1f}%, CEILING = {p_ceiling:.1f}%."
    )
    print("VERDICT:", "PROCEED" if p_ceiling >= 45.0 else "STOP — ceiling below the bar")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

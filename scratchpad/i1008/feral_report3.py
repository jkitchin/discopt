"""Three-arm report for the feral bump (#1008).

Arms, all over the same captured relaxation LPs, one process each:

  base  feral 0.15.1 (the shipped pin before this change)
  off   feral e00aa706, `DISCOPT_LU_TRIANGULARIZE` unset — the shipping default
  on    feral e00aa706, `DISCOPT_LU_TRIANGULARIZE=1` — peel + dense-bump route

Two questions, two standards of proof:

1. Is the BUMP neutral? `off` must equal `base` EXACTLY — status, bit-level
   objective, and every counter. This is the Cargo.toml pin comment's regime
   (CLAUDE.md §5 bound-neutral): any drift, even an improvement, fails.
2. Is the PEEL worth graduating? `on` vs `base` on fill and factor work, with
   objectives checked against HiGHS at the repo tolerance first (§1, zero slack).

Counters are exact integers and load-independent. Wall is reported because it is
the quantity #1008 is about, but the three arms ran CONCURRENTLY on one machine,
so wall here is directional only — a speed claim needs the interleaved,
load-gated run (§9), not this table.

§6: prints the executed comparison count and exits non-zero if it is zero.
§7: nothing is caught.
"""

from __future__ import annotations

import json
import os
import sys

D = os.environ.get("OUT_DIR", "/private/tmp/wt1008g/scratchpad/i1008/out2")
ATOL, RTOL = 1e-6, 1e-4
EXACT = ("status", "obj", "facs", "basis_nnz", "factor_nnz", "p1", "p2", "cold_fallback")


def load(name):
    p = os.path.join(D, f"{name}.jsonl")
    return {json.loads(ln)["tag"]: json.loads(ln) for ln in open(p) if ln.strip()}


base, off, on = load("base"), load("off"), load("on")
common = sorted(set(base) & set(off) & set(on))
assert common, "the three arms share no LPs"

print(f"# arms: base={len(base)} off={len(off)} on={len(on)}; comparable on {len(common)} LPs\n")

print("─── 1. is the bump neutral? (off vs base, EXACT) ───")
diffs = [(t, k, base[t][k], off[t][k]) for t in common for k in EXACT if base[t][k] != off[t][k]]
n_exact = len(common) * len(EXACT)
if diffs:
    print(f"  {len(diffs)} DIFFERENCE(S) of {n_exact} comparisons — the bump is NOT neutral:")
    for t, k, x, y in diffs[:20]:
        print(f"    {t:24s} {k:14s} {x!r} -> {y!r}")
else:
    print(
        f"  NEUTRAL — {n_exact} exact comparisons ({len(common)} LPs x {len(EXACT)} fields), "
        f"0 differences"
    )

print("\n─── correctness vs HiGHS (zero slack, all three arms) ───")
bad = []
for t in common:
    for nm, r in (("base", base[t]), ("off", off[t]), ("on", on[t])):
        if r["ref"] is None or r["status"] != "optimal":
            continue
        if abs(r["obj"] - r["ref"]) > ATOL + RTOL * abs(r["ref"]):
            bad.append((t, nm, r["obj"], r["ref"]))
if bad:
    for t, nm, o, r in bad:
        print(f"  FAIL {t:24s} [{nm}] obj={o!r} ref={r!r}")
else:
    print(f"  clean — {3 * len(common)} arm x LP objectives all at the HiGHS optimum")


def fill(r):
    return r["factor_nnz"] / r["basis_nnz"] if r["basis_nnz"] else float("nan")


print("\n─── 2. what the peel does (on vs base) ───")
print(f"{'LP':<24} {'fill base':>9} {'fill on':>9} {'facs b/on':>12} {'wall b/on':>14} {'cold':>6}")
tot = {k: [0, 0] for k in ("facs", "basis_nnz", "factor_nnz")}
wall = [0.0, 0.0]
better = worse = same = 0
for t in common:
    b, o = base[t], on[t]
    for k in tot:
        tot[k][0] += b[k]
        tot[k][1] += o[k]
    wall[0] += b["wall"]
    wall[1] += o["wall"]
    if b["factor_nnz"]:
        rel = (o["factor_nnz"] - b["factor_nnz"]) / b["factor_nnz"]
        better += rel < -0.01
        worse += rel > 0.01
        same += abs(rel) <= 0.01
    print(
        f"{t:<24} {fill(b):>8.2f}x {fill(o):>8.2f}x {b['facs']:>5}/{o['facs']:<6} "
        f"{b['wall']:>6.2f}/{o['wall']:<7.2f} {b['cold_fallback']}->{o['cold_fallback']:<4}"
    )

print("\n─── totals (on vs base) ───")
print(f"  factorizations   {tot['facs'][0]:>12,} -> {tot['facs'][1]:>12,}")
print(f"  basis nnz        {tot['basis_nnz'][0]:>12,} -> {tot['basis_nnz'][1]:>12,}")
print(
    f"  factor nnz       {tot['factor_nnz'][0]:>12,} -> {tot['factor_nnz'][1]:>12,}"
    f"   ({(tot['factor_nnz'][1] / tot['factor_nnz'][0] - 1) * 100:+.1f}%)"
)
print(
    f"  fill             {tot['factor_nnz'][0] / tot['basis_nnz'][0]:>11.2f}x -> "
    f"{tot['factor_nnz'][1] / tot['basis_nnz'][1]:>11.2f}x"
)
print(f"  wall (directional only, contended)  {wall[0]:.1f}s -> {wall[1]:.1f}s")
print(f"  per-LP factor nnz: better {better}, worse {worse}, within 1% {same}")
print(
    f"  cold fallbacks   base {sum(base[t]['cold_fallback'] for t in common)} -> "
    f"on {sum(on[t]['cold_fallback'] for t in common)}"
)

print(f"\nexecuted comparisons: {n_exact} exact + {len(common)} peel rows")
sys.exit(1 if (diffs or bad) else 0)

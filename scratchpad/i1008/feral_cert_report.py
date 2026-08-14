"""Diff two cert-panel arms for the feral bump (#1008).

The regime is `crates/discopt-core/Cargo.toml`'s: a dependency bump that moves LU
arithmetic is admissible only if `status`, `node_count` and the certified
`objective` are EXACTLY unchanged on the certifying panel. Any drift — even an
apparent improvement — fails.

§6: prints the executed comparison count and exits non-zero if it is zero.
§7: nothing is caught.
"""

from __future__ import annotations

import json
import os
import sys

A = os.environ.get("ARM_A", "/private/tmp/wt1008g/scratchpad/i1008/out/cert_old.jsonl")
B = os.environ.get("ARM_B", "/private/tmp/wt1008g/scratchpad/i1008/out/cert_new.jsonl")
FIELDS = ("status", "node_count", "objective", "bound")


def load(p):
    rows = {}
    for ln in open(p):
        if ln.strip():
            r = json.loads(ln)
            rows[r["instance"]] = r
    return rows


a, b = load(A), load(B)
common = sorted(set(a) & set(b))
assert common, "the two arms share no instances"

viol = []
for inst in common:
    for f in FIELDS:
        if a[inst].get(f) != b[inst].get(f):
            viol.append((inst, f, a[inst].get(f), b[inst].get(f)))

print(
    f"cert panel: {len(common)} instances x {len(FIELDS)} fields "
    f"= {len(common) * len(FIELDS)} exact comparisons"
)
print(f"  arm A = {os.path.basename(A)} ({len(a)} rows)")
print(f"  arm B = {os.path.basename(B)} ({len(b)} rows)")
only = set(a) ^ set(b)
if only:
    print(f"  NOTE: {len(only)} instance(s) in only one arm: {sorted(only)}")

print("\n─── bound-neutrality (exact) ───")
if not viol:
    print(f"  NEUTRAL — all {len(common)} rows bit-identical in {', '.join(FIELDS)}")
else:
    print(f"  {len(viol)} DIFFERENCE(S):")
    for inst, f, x, y in viol:
        print(f"    {inst:20s} {f:12s} {x!r} -> {y!r}")

print(f"\nexecuted comparisons: {len(common) * len(FIELDS)}")
sys.exit(1 if viol else 0)

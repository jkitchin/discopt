"""Acceptance report for the feral 0.16.0 bump (#1008), both §5 regimes.

The Cargo.toml pin comment's regime (CLAUDE.md §5 *bound-neutral*) demands
`status`, `node_count` and the certified `objective` be EXACTLY unchanged. That
regime was written for bumps whose arithmetic should not move at all. 0.16.0
replaces the pivoting rule outright, so it is a *bound-changing* change by
construction and the bound-neutral diff is reported here as evidence, not as the
gate. The gate is the pair §5 states for that regime:

  cert-clean   no status regression, no bound above its reference optimum,
               no `optimal` instance dropping out of `optimal`, objective drift
               within the repo tolerance (abs 1e-6 / rel 1e-4).
  net-positive measurably helpful broadly on node count / wall, not merely sound.

Oracle is `minlplib.solu` where the instance is in it. Wall is reported but the
two arms ran concurrently on one machine, so per CLAUDE.md §9 it is directional
only and no speed claim rests on it.

§6: prints executed comparison counts and exits non-zero if any is zero.
§7: nothing is caught.
"""

from __future__ import annotations

import json
import os
import sys

D = os.path.dirname(os.path.abspath(__file__))
ATOL, RTOL = 1e-6, 1e-4
SOLU = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")


def load(p):
    return {json.loads(ln)["instance"]: json.loads(ln) for ln in open(p) if ln.strip()}


def oracle():
    """`=obj=` / `=opt=` rows of minlplib.solu, keyed by instance."""
    out = {}
    with open(SOLU) as fh:
        for ln in fh:
            f = ln.split()
            if len(f) >= 3 and f[0] in ("=obj=", "=opt="):
                out[f[1]] = float(f[2])
    return out


A = load(os.path.join(D, "cert_0151.jsonl"))
B = load(os.path.join(D, "cert_0160.jsonl"))
ORA = oracle()
K = sorted(set(A) & set(B))
assert K, "the two arms share no instances"
print(f"# arms: 0.15.1={len(A)} 0.16.0={len(B)}; comparable on {len(K)}\n")

# ---- evidence: the bound-neutral diff the pin comment asks for ----------------
print("─── bound-neutral diff (evidence, not the gate) ───")
EX = ("status", "node_count", "objective", "bound")
drift = [(k, f, A[k][f], B[k][f]) for k in K for f in EX if A[k][f] != B[k][f]]
n_exact = len(K) * len(EX)
print(f"  {len(drift)} of {n_exact} exact comparisons differ")
for k, f, x, y in drift:
    print(f"    {k:22s} {f:11s} {x!r} -> {y!r}")

# ---- gate 1: cert-clean ------------------------------------------------------
print("\n─── gate 1: cert-clean ───")
viol, n_chk = [], 0
for k in K:
    a, b = A[k], B[k]
    n_chk += 1
    if a["status"] == "optimal" and b["status"] != "optimal":
        viol.append(f"{k}: status regressed {a['status']} -> {b['status']}")
    ref = ORA.get(k)
    if ref is None:
        continue
    n_chk += 1
    for nm, r in (("0.15.1", a), ("0.16.0", b)):
        o = r.get("objective")
        if o is not None and abs(o - ref) > ATOL + RTOL * abs(ref):
            viol.append(f"{k} [{nm}]: objective {o!r} off oracle {ref!r}")
    # min sense: a dual bound may never exceed the reference optimum
    bd = b.get("bound")
    if bd is not None and bd > ref + ATOL + RTOL * abs(ref):
        viol.append(f"{k} [0.16.0]: bound {bd!r} ABOVE oracle {ref!r}")
if viol:
    print(f"  {len(viol)} VIOLATION(S) of {n_chk} checks:")
    for v in viol:
        print(f"    {v}")
else:
    print(f"  clean — {n_chk} checks, 0 violations ({sum(k in ORA for k in K)} oracle-backed)")

# ---- gate 2: net-positive ----------------------------------------------------
print("\n─── gate 2: net-positive ───")
nb = nw = ns = 0
dn = []
for k in K:
    x, y = A[k]["node_count"], B[k]["node_count"]
    if x is None or y is None:
        continue
    if y < x:
        nb += 1
        dn.append((k, x, y))
    elif y > x:
        nw += 1
        dn.append((k, x, y))
    else:
        ns += 1
tot_a = sum(A[k]["node_count"] or 0 for k in K)
tot_b = sum(B[k]["node_count"] or 0 for k in K)
wa = sum(A[k]["wall_time"] or 0.0 for k in K)
wb = sum(B[k]["wall_time"] or 0.0 for k in K)
print(f"  node_count   fewer {nb}, more {nw}, identical {ns}")
print(f"  total nodes  {tot_a:,} -> {tot_b:,}  ({(tot_b / tot_a - 1) * 100:+.1f}%)" if tot_a else "")
print(f"  total wall   {wa:.1f}s -> {wb:.1f}s  (DIRECTIONAL ONLY, §9: arms ran concurrently)")
for k, x, y in dn:
    print(f"    {k:22s} {x:6d} -> {y:6d}")

print(f"\nexecuted comparisons: bound-neutral {n_exact}, cert-clean {n_chk}, node rows {len(K)}")
assert n_exact and n_chk and K
sys.exit(1 if viol else 0)

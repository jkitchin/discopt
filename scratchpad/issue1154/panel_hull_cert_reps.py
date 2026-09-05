"""#1154 — repetition control on the ONE net-positive claim that is timing-shaped.

The v2 capability panel measured the ``hull`` certification rate at 81/108
(chain arm) vs 99/108 (Σ arm). That is measured under a 15 s wall limit, so it is
a timing claim and CLAUDE.md §9 requires an interleaved A/B, a load gate and a
spread rather than a single draw.

This probe re-runs only the subset that carries the difference -- the ``hull``
route on the 54 nonlinear cases -- three times, arms INTERLEAVED per case, and
reports the per-rep certification counts and their standard deviation.

Prints per-rep progress (§10) and an executed-comparison count (§6).
"""

from __future__ import annotations

import itertools
import statistics
import sys

import discopt.modeling as dm

REPS = 3
ARMS = ("chain", "sumover")


def _chain(terms):
    acc = terms[0]
    for t in terms[1:]:
        acc = acc + t
    return acc


def build(n_terms, n_disj, sense, coefs, *, arm):
    m = dm.Model("rep")
    x = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(n_terms)]
    rhs = float(n_terms) + 1.0
    disjuncts = []
    for k in range(n_disj):
        scale = 1.0 + k
        parts = [dm.exp(coefs[i % len(coefs)] * scale * x[i] / 10.0) for i in range(n_terms)]
        body = dm.sum(p for p in parts) if arm == "sumover" else _chain(parts)
        if sense == "<=":
            disjuncts.append([body <= rhs])
        elif sense == ">=":
            disjuncts.append([body >= -rhs])
        else:
            disjuncts.append([body == rhs])
    m.either_or(disjuncts)
    m.minimize(-sum(x[i] for i in range(n_terms)))
    return m


CASES = list(
    itertools.product(
        (2, 3, 5), (2, 3), ("<=", ">=", "=="),
        ((1.0,), (1.0, -1.0), (0.5, 2.0, -1.5)),
    )
)

print(f"load at start: {open('/proc/loadavg').read().split()[0]}", flush=True)
print(f"cases per rep: {len(CASES)}", flush=True)

per_rep = {arm: [] for arm in ARMS}
comparisons = 0

for rep in range(REPS):
    counts = dict.fromkeys(ARMS, 0)
    for case in CASES:
        for arm in ARMS:            # interleaved within the case, not arm-by-arm
            r = build(*case, arm=arm).solve(gdp_method="hull", time_limit=15)
            comparisons += 1
            if str(r.status) == "optimal":
                counts[arm] += 1
    for arm in ARMS:
        per_rep[arm].append(counts[arm])
    print(f"  rep {rep}: " + "  ".join(f"{a}={counts[a]}/{len(CASES)}" for a in ARMS), flush=True)

print()
for arm in ARMS:
    vals = per_rep[arm]
    sd = statistics.stdev(vals) if len(vals) > 1 else 0.0
    print(f"{arm:8s}: certified {vals}  mean={statistics.mean(vals):.2f}  sd={sd:.2f}")
print(f"executed_comparisons={comparisons}")
if comparisons == 0:
    print("PROBE DID NOT FIRE", file=sys.stderr)
    sys.exit(1)

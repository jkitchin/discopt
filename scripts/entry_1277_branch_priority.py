"""#1277 G entry experiment: is a user-declared SPATIAL BRANCHING ORDER worth it?

#1277 G proposes ``m.branch_priority({var: int})`` so a plugin can tell the
solver which continuous variables to bisect first. Before building it, this
measures whether the branching order has any leverage to give -- on the CALPHAD
class the issue was opened about.

Two probes, both run before any implementation:

1. **The oracle over declaration orders.** Reading
   ``bnb/branching.rs::select_spatial_branch_variable``, the default rule is
   longest-edge in NORMALIZED coordinates (``width / global_width``). At the root
   every continuous column sits at exactly 1.0, so the strict ``>`` with an
   ascending scan makes the root branch on the LOWEST-INDEX column -- declaration
   order. Permuting that order over 8 permutations gave a node-count spread of
   1.12x on a 4-variable CEF, **1.11x** on the 9-variable/27-trilinear CEF below
   (3461..3859 nodes, sd 122 on a mean of 3561) and **1.00x** on two pooling
   models (identical node counts every way round). An informed hint cannot beat
   the oracle, so that is the ceiling.

2. **A per-node override** (this script). Permutation only tips the root tie,
   which is a weak proxy for a priority that overrides the width rule at EVERY
   node. ``tree.set_branch_deprioritized(cols)`` is exactly a two-level priority
   applied at every spatial node, so the two seams it is fed from are patched
   here to carry an ARBITRARY column set -- the mechanism G would generalise.

**A trap worth recording.** The first version of probe 2 compared NODE COUNTS at
a fixed time limit and appeared to show the override nearly halving them (5613 ->
2037..2955). Every arm was hitting the time limit, so that column was measuring
throughput, not work: an arm that got through fewer nodes was slower per node,
not better. Re-run at a fixed NODE BUDGET and compared on the DUAL BOUND, the
result reverses completely and the override is 2x-5x worse.

RESULT (equal node budget, ~2000 nodes, gap to the same incumbent):

    default (no override)          8.10%
    branch species 0 first        16.04%
    branch sublattice 2 first     29.71%
    branch sublattice 1 first     32.51%
    branch sublattice 0 first     41.29%

Every hand-specified order loses to the default, and the oracle over orders is
worth 11%. G is declined on this measurement: a public API that steers a
soundness-critical selector (the completeness argument in ``branching.rs`` is
what stops a false certificate) is not justified by a lever this small and this
often negative.
"""

import discopt._relax.dependent_vars as dv
import discopt.modeling as dm
import numpy as np

R_T = 8.314 * 1000.0
COEF = np.random.default_rng(3).normal(0.0, 8000.0, size=(3, 3, 3))
NAMES = [f"s{a}_{b}" for a in range(3) for b in range(3)]

_calls = {"names": 0, "cols": 0}
_forced: set = set()
_orig_names = dv.find_functionally_dependent_names
_orig_cols = dv.dependent_columns_for_model


def _patched_names(model):
    _calls["names"] += 1
    return set(_forced) if _forced else _orig_names(model)


def _patched_cols(model, names):
    _calls["cols"] += 1
    if not _forced:
        return _orig_cols(model, names)
    idx = {n: i for i, n in enumerate(NAMES)}
    return {idx[n] for n in names if n in idx}


dv.find_functionally_dependent_names = _patched_names
dv.dependent_columns_for_model = _patched_cols


def cef3():
    m = dm.Model("cef3")
    v = {n: m.continuous(n, lb=1e-6, ub=1.0) for n in NAMES}
    subs = [[v[f"s{a}_{b}"] for b in range(3)] for a in range(3)]
    for s in subs:
        m.subject_to(s[0] + s[1] + s[2] == 1.0)
    g = R_T * sum(dm.xlogx(x) for s in subs for x in s)
    for i in range(3):
        for j in range(3):
            for k in range(3):
                g = g + float(COEF[i, j, k]) * subs[0][i] * subs[1][j] * subs[2][k]
    m.minimize(g)
    return m


NODE_BUDGET = 2000

ARMS = {
    "default (no override)": set(),
    "branch sublattice 0 first": {f"s{a}_{b}" for a in (1, 2) for b in range(3)},
    "branch sublattice 1 first": {f"s{a}_{b}" for a in (0, 2) for b in range(3)},
    "branch sublattice 2 first": {f"s{a}_{b}" for a in (0, 1) for b in range(3)},
    "branch species 0 first": {n for n in NAMES if not n.endswith("_0")},
}

base = None
n = 0
for label, forced in ARMS.items():
    _forced = forced
    before = dict(_calls)
    r = cef3().solve(time_limit=600, max_nodes=NODE_BUDGET)
    fired = _calls["cols"] - before["cols"]
    if forced:
        assert fired > 0, f"{label}: the deprioritization seam never fired"
    if base is None:
        base = r.objective
    assert abs(r.objective - base) <= 1e-3 * max(1.0, abs(base)), (label, r.objective, base)
    gap = abs(r.objective - r.bound) / max(1.0, abs(r.objective))
    print(
        f"{label:<28} nodes={r.node_count:7d} bound={r.bound:14.4f} "
        f"gap={gap:8.4%} {r.status:>9} seam_calls={fired}",
        flush=True,
    )
    assert r.bound <= r.objective + 1e-6 * max(1.0, abs(r.objective)), (label, r.bound)
    n += 1

assert n == len(ARMS) and _calls["cols"] > 0, (n, _calls)
print(f"EXECUTED_ARMS={n} seam_invocations={_calls['cols']}")

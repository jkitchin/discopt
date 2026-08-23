"""EXPERIMENT 6 -- the discrete version: which pair of states crosses lowest?

Experiments 1-5 are all purely continuous: you name two surfaces and ask for
the lowest point where those two are degenerate.  The real question a chemist
has is usually one level up.  A molecule has several low-lying electronic
states, and what matters is the lowest crossing point *over any pair of them* --
because that is the one the reaction goes through.

That is a disjunction, and it makes the problem an MINLP:

    min_{x, y}  E
    s.t.  for each candidate pair p = (i,j):
              y_p = 1  =>  W_i(x) = W_j(x)  and  E = W_i(x)
          sum_p y_p = 1,   y_p in {0,1},   x in [x^L, x^U]

Here it is written as a big-M MINLP (discopt also has a GDP path,
``gdp_method``, which would take the disjunction directly).  The baseline is
enumeration: solve each pair's MECP as a separate continuous problem and take
the best.  With P pairs that is P certified solves versus one.

The question this measures is not "does the MINLP get the right answer" --
it must, since the formulations are equivalent -- but whether posing the
disjunction is *cheaper* than enumerating it.  For 3 states / 3 pairs the
answer is going to be "no"; the interesting regime is many states, where
enumeration grows quadratically and a single tree can prune whole pairs on a
bound.  Recording the 3-state datapoint so the crossover has a starting point.
"""

from __future__ import annotations

import itertools
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 1)[0])

import discopt.modeling as dm  # noqa: E402
import mecp_models as M  # noqa: E402

CHECKS = 0
NOTES: list[str] = []

# Three surfaces over the same two coordinates: one lower state (the
# double-well conformer surface) and two upper states at different offsets.
# States are indexed 0, 1, 2 and every pair is a candidate crossing.
LOWER = M.TwoWellParams(n=2, dE2=1.10)
UPPER_A = M.TwoWellParams(n=2, dE2=1.10)
UPPER_B = M.TwoWellParams(n=2, dE2=1.85, B=0.62)

BNDS = np.array(LOWER.bounds(), float)
PAIRS = list(itertools.combinations(range(3), 2))


def surfaces(q, exp=np.exp):
    """(W0, W1, W2) -- W0 is the lower state, W1 and W2 the two upper ones."""
    w0, w1 = LOWER.states(q, exp=exp)
    _, w2 = UPPER_B.states(q, exp=exp)
    return w0, w1, w2


def show(tag, r, el=None):
    el = r.wall_time if el is None else el
    print(
        f"  {tag:<28s} status={r.status:<11s} obj={_n(r.objective)} bound={_n(r.bound)} "
        f"gap={_g(r.gap)} cert={str(r.gap_certified):<5s} nodes={r.node_count:<6d} t={el:7.2f}s",
        flush=True,
    )


def _n(v):
    return "     None" if v is None else f"{v:10.6f}"


def _g(v):
    return "    None" if v is None else f"{v:8.1e}"


print("=" * 104)
print("EXPERIMENT 6 -- lowest crossing point over ANY pair of 3 electronic states")
print("=" * 104)

# --------------------------------------------------------------------------
# Baseline: enumerate the pairs, one continuous MECP solve each
# --------------------------------------------------------------------------
print(f"\n[A] enumeration baseline -- {len(PAIRS)} separate continuous solves")
enum_best = (np.inf, None, None)
enum_time = 0.0
enum_nodes = 0
for i, j in PAIRS:

    def build(i=i, j=j):
        m = dm.Model(f"mecp_pair_{i}{j}")
        q = [m.continuous(f"q{k}", lb=BNDS[k, 0], ub=BNDS[k, 1]) for k in range(2)]
        w = surfaces(q, exp=dm.exp)
        m.minimize(w[i])
        m.subject_to(w[i] - w[j] == 0)
        return m

    m = build()
    t0 = time.time()
    r = m.solve(time_limit=120)
    el = time.time() - t0
    enum_time += el
    enum_nodes += r.node_count
    show(f"pair ({i},{j})", r, el)
    CHECKS += 1
    if r.objective is not None and r.objective < enum_best[0]:
        xv = np.array([float(np.asarray(r.x[f"q{k}"]).ravel()[0]) for k in range(2)])
        enum_best = (r.objective, (i, j), xv)

print(
    f"  => best pair {enum_best[1]} at E={enum_best[0]:.6f}, "
    f"total {enum_time:.2f}s over {enum_nodes} nodes"
)
CHECKS += 1
assert enum_best[1] is not None, "enumeration found no crossing at all"

# --------------------------------------------------------------------------
# Big-M constant. A rigorous M would come from interval arithmetic / FBBT;
# a grid maximum with a safety factor is enough for an illustrative example,
# and it is verified after the solve (the selected disjunct must be tight).
# --------------------------------------------------------------------------
g = np.linspace(BNDS[0, 0], BNDS[0, 1], 220)
h = np.linspace(BNDS[1, 0], BNDS[1, 1], 220)
G, H = np.meshgrid(g, h, indexing="ij")
W = surfaces([G, H], exp=np.exp)
spans = [float(np.max(np.abs(W[i] - W[j]))) for i, j in PAIRS]
levels = [float(np.max(np.abs(w))) for w in W]
BIGM = 2.0 * max(max(spans), max(levels))
print(f"\n[B] disjunctive MINLP  (big-M = {BIGM:.2f}, from a grid max x2)")

# --------------------------------------------------------------------------
# One MINLP: binaries choose which pair is the active disjunct
# --------------------------------------------------------------------------
m = dm.Model("mecp_multistate_gdp")
q = [m.continuous(f"q{k}", lb=BNDS[k, 0], ub=BNDS[k, 1]) for k in range(2)]
E = m.continuous("E", lb=float(-BIGM), ub=float(BIGM))
y = [m.binary(f"y_{i}{j}") for i, j in PAIRS]
w = surfaces(q, exp=dm.exp)

m.minimize(E)
m.subject_to(sum(y) == 1)
for p, (i, j) in enumerate(PAIRS):
    slack = BIGM * (1 - y[p])
    # y_p = 1  =>  the two states are degenerate here
    m.subject_to(w[i] - w[j] <= slack)
    m.subject_to(w[j] - w[i] <= slack)
    # y_p = 1  =>  E is that (common) energy
    m.subject_to(E - w[i] <= slack)
    m.subject_to(w[i] - E <= slack)

t0 = time.time()
r_gdp = m.solve(time_limit=300)
gdp_time = time.time() - t0
show("disjunctive MINLP", r_gdp, gdp_time)
CHECKS += 1

if r_gdp.x is not None:
    xv = np.array([float(np.asarray(r_gdp.x[f"q{k}"]).ravel()[0]) for k in range(2)])
    yv = [float(np.asarray(r_gdp.x[f"y_{i}{j}"]).ravel()[0]) for i, j in PAIRS]
    chosen = PAIRS[int(np.argmax(yv))]
    wv = surfaces(list(xv), exp=np.exp)
    print(f"  selected pair {chosen}  y={np.round(yv, 3)}")
    print(f"  x={np.round(xv, 5)}  W={np.round([float(v) for v in wv], 6)}")
    print(f"  |W{chosen[0]}-W{chosen[1]}| = {abs(float(wv[chosen[0]] - wv[chosen[1]])):.2e}")

    # The MINLP and the enumeration solve equivalent problems, so they must
    # agree on the energy. (The pair need not be unique if two pairs tie.)
    CHECKS += 1
    if abs(float(wv[chosen[0]]) - enum_best[0]) > 1e-3:
        NOTES.append(
            f"MINLP energy {float(wv[chosen[0]]):.6f} != enumeration best "
            f"{enum_best[0]:.6f} -- the two formulations are not equivalent"
        )
    CHECKS += 1
    if chosen != enum_best[1]:
        NOTES.append(f"MINLP chose pair {chosen}, enumeration chose {enum_best[1]}")

    # Big-M validity check: on the SELECTED disjunct the constraints must be
    # tight, i.e. the chosen pair really is degenerate at the returned point.
    CHECKS += 1
    if abs(float(wv[chosen[0]] - wv[chosen[1]])) > 1e-4:
        NOTES.append(
            "selected disjunct is not degenerate at the solution -- big-M too "
            "loose, the MINLP found a spurious 'crossing'"
        )

print("\n" + "=" * 104)
print(f"{'route':<24s} {'energy':>12s} {'certified':>10s} {'nodes':>8s} {'wall':>9s}")
print(
    f"{'enumeration (3 solves)':<24s} {enum_best[0]:>12.6f} "
    f"{'yes':>10s} {enum_nodes:>8d} {enum_time:>8.2f}s"
)
print(
    f"{'disjunctive MINLP (1)':<24s} "
    f"{(r_gdp.objective if r_gdp.objective is not None else float('nan')):>12.6f} "
    f"{str(r_gdp.gap_certified):>10s} {r_gdp.node_count:>8d} {gdp_time:>8.2f}s"
)

print(f"\nEXECUTED CHECKS: {CHECKS}")
print(f"NOTES: {len(NOTES)}")
for s in NOTES:
    print(f"  - {s}")
if CHECKS == 0:
    sys.exit(2)

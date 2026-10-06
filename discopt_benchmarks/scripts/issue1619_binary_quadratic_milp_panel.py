"""#1619 C-01b graduation panel: ``DISCOPT_BINARY_QUADRATIC_MILP`` OFF vs ON.

ON sends a model whose only nonlinear terms are products of binary-valued
variables of degree 2 (a binary QP, possibly with continuous variables appearing
linearly) to the MILP route as its exact Fortet linearization
(``binary_multilinear_reform``); OFF keeps it on spatial branch and bound.

Instances: the in-repo corpus members the flag changes (``st_miqp1``,
``st_test1`` -- found by scanning all 66 ``python/tests/data/minlplib_nl`` files
for ``has_binary_multilinear_work`` at degree 2 but not 3) plus seeded generated
families: the issue's 4x4 dopant lattice (and 5x5 / random-coupling variants),
max-cut, unconstrained BQP, quadratic knapsack, k-cardinality BQP, and a mixed
binary-product / continuous-linear facility model. Each pure-binary instance
with n <= 20 carries a brute-force oracle (exhaustive enumeration).

Per pair: OFF and ON run back to back on fresh models (interleaved), same
``time_limit``. CLAUDE.md §5 bars:
 * cert-clean: no false certificate (a certified objective off the oracle, or a
   bound past it), no certified->uncertified regression, the two arms'
   certified objectives agree within tolerance, every ON incumbent re-verified
   feasible on the ORIGINAL model;
 * net-positive: certificates / wall / nodes.

Prints an executed-comparison count and exits non-zero when it is zero (§6).
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
FLAG = "DISCOPT_BINARY_QUADRATIC_MILP"


# ── instance builders: return (model, oracle_or_None, sense) ─────────────────


def _bqp_oracle(Qm, h, sense, feasible=None, n=None):
    """Exhaustive optimum of x'Qx + h.x over {0,1}^n (upper-triangular Q)."""
    n = len(h) if n is None else n
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.float64)
    vals = np.einsum("ki,ij,kj->k", X, Qm, X) + X @ h
    if feasible is not None:
        vals = np.where(feasible(X), vals, np.inf if sense == "min" else -np.inf)
    return float(vals.min() if sense == "min" else vals.max())


def dopants(L, k, seed=None):
    import discopt.modeling as dm

    N = L * L
    J1, J2, J3, hh = 0.40, -0.10, -0.25, -0.08
    rng = np.random.default_rng(seed) if seed is not None else None
    Jm = np.zeros((N, N))
    for s, t in itertools.combinations(range(N), 2):
        (r1, c1), (r2, c2) = divmod(s, L), divmod(t, L)
        dr, dc = min(abs(r1 - r2), L - abs(r1 - r2)), min(abs(c1 - c2), L - abs(c1 - c2))
        v = {(0, 1): J1, (1, 0): J1, (1, 1): J2, (0, 2): J3, (2, 0): J3}.get((dr, dc), 0.0)
        if rng is not None and v != 0.0:
            v = round(v * rng.uniform(0.5, 1.5), 3)
        Jm[s, t] = v
    hv = np.array([hh if s < L else 0.0 for s in range(N)])
    pairs = [(s, t) for s, t in itertools.combinations(range(N), 2) if Jm[s, t] != 0]
    m = dm.Model(f"dopants{L}_{k}_{seed}")
    x = m.binary("x", shape=(N,))
    m.minimize(
        dm.sum(
            lambda p: float(Jm[pairs[p]]) * x[pairs[p][0]] * x[pairs[p][1]], over=range(len(pairs))
        )
        + dm.sum(lambda s: hv[s] * x[s], over=range(N))
    )
    m.subject_to(dm.sum(lambda s: x[s], over=range(N)) == k)
    oracle = None
    if N <= 20:
        oracle = _bqp_oracle(Jm, hv, "min", feasible=lambda X: X.sum(1) == k)
    return m, oracle, "min"


def maxcut(n, p, seed):
    import discopt.modeling as dm

    rng = np.random.default_rng(seed)
    edges = [
        (i, j, float(rng.integers(1, 10)))
        for i, j in itertools.combinations(range(n), 2)
        if rng.uniform() < p
    ]
    m = dm.Model(f"maxcut{n}_{seed}")
    x = m.binary("x", shape=(n,))
    m.maximize(
        dm.sum(
            lambda e: edges[e][2]
            * (x[edges[e][0]] + x[edges[e][1]] - 2 * x[edges[e][0]] * x[edges[e][1]]),
            over=range(len(edges)),
        )
    )
    Qm, h = np.zeros((n, n)), np.zeros(n)
    for i, j, w in edges:
        Qm[i, j] -= 2 * w
        h[i] += w
        h[j] += w
    oracle = _bqp_oracle(Qm, h, "max") if n <= 20 else None
    return m, oracle, "max"


def bqp(n, density, seed):
    import discopt.modeling as dm

    rng = np.random.default_rng(seed)
    Qm = np.triu(rng.integers(-100, 101, (n, n)).astype(float), 1)
    Qm *= rng.uniform(size=(n, n)) < density
    h = rng.integers(-100, 101, n).astype(float)
    terms = [(i, j, Qm[i, j]) for i in range(n) for j in range(i + 1, n) if Qm[i, j] != 0]
    m = dm.Model(f"bqp{n}_{seed}")
    x = m.binary("x", shape=(n,))
    m.minimize(
        dm.sum(lambda t: terms[t][2] * x[terms[t][0]] * x[terms[t][1]], over=range(len(terms)))
        + dm.sum(lambda i: h[i] * x[i], over=range(n))
    )
    oracle = _bqp_oracle(Qm, h, "min") if n <= 20 else None
    return m, oracle, "min"


def qknap(n, seed):
    import discopt.modeling as dm

    rng = np.random.default_rng(seed)
    P = np.triu(rng.integers(0, 50, (n, n)).astype(float) * (rng.uniform(size=(n, n)) < 0.5), 1)
    w = rng.integers(1, 30, n).astype(float)
    cap = float(np.floor(0.4 * w.sum()))
    pr = rng.integers(0, 50, n).astype(float)
    terms = [(i, j, P[i, j]) for i in range(n) for j in range(i + 1, n) if P[i, j] != 0]
    m = dm.Model(f"qknap{n}_{seed}")
    x = m.binary("x", shape=(n,))
    m.maximize(
        dm.sum(lambda t: terms[t][2] * x[terms[t][0]] * x[terms[t][1]], over=range(len(terms)))
        + dm.sum(lambda i: pr[i] * x[i], over=range(n))
    )
    m.subject_to(dm.sum(lambda i: w[i] * x[i], over=range(n)) <= cap)
    oracle = _bqp_oracle(P, pr, "max", feasible=lambda X: X @ w <= cap) if n <= 20 else None
    return m, oracle, "max"


def kcard(n, k, seed):
    import discopt.modeling as dm

    rng = np.random.default_rng(seed)
    Qm = np.triu(rng.normal(0, 1, (n, n)).round(3), 1)
    h = rng.normal(0, 1, n).round(3)
    terms = [(i, j, Qm[i, j]) for i in range(n) for j in range(i + 1, n)]
    m = dm.Model(f"kcard{n}_{k}_{seed}")
    x = m.binary("x", shape=(n,))
    m.minimize(
        dm.sum(
            lambda t: float(terms[t][2]) * x[terms[t][0]] * x[terms[t][1]], over=range(len(terms))
        )
        + dm.sum(lambda i: float(h[i]) * x[i], over=range(n))
    )
    m.subject_to(dm.sum(lambda i: x[i], over=range(n)) == k)
    oracle = _bqp_oracle(Qm, h, "min", feasible=lambda X: X.sum(1) == k) if n <= 20 else None
    return m, oracle, "min"


def facility(nf, nc, seed):
    """Open facilities (binary) with pairwise interaction costs y_i*y_j, and
    continuous flows that may only use open facilities. No oracle."""
    import discopt.modeling as dm

    rng = np.random.default_rng(seed)
    f = rng.integers(20, 60, nf).astype(float)
    I = np.triu(rng.integers(0, 15, (nf, nf)).astype(float), 1)
    cost = rng.integers(1, 20, (nf, nc)).astype(float)
    d = rng.integers(5, 15, nc).astype(float)
    capf = float(d.sum() / 2)
    m = dm.Model(f"fac{nf}_{nc}_{seed}")
    y = m.binary("y", shape=(nf,))
    z = m.continuous("z", shape=(nf, nc), lb=0, ub=float(d.max()))
    pairs = [(i, j) for i in range(nf) for j in range(i + 1, nf) if I[i, j] != 0]
    m.minimize(
        dm.sum(lambda i: f[i] * y[i], over=range(nf))
        + dm.sum(lambda p: I[pairs[p]] * y[pairs[p][0]] * y[pairs[p][1]], over=range(len(pairs)))
        + dm.sum(lambda k: cost[k // nc, k % nc] * z[k // nc, k % nc], over=range(nf * nc))
    )
    for j in range(nc):
        m.subject_to(dm.sum(lambda i: z[i, j], over=range(nf)) == d[j])
    for i in range(nf):
        m.subject_to(dm.sum(lambda j: z[i, j], over=range(nc)) <= capf * y[i])
    return m, None, "min"


def autocorr(n, K, objvar):
    """Degree-4 Bernasconi instance (the #187 class the pass was built for), in
    the expression form and the `.nl` objvar form; brute-force oracle."""
    from discopt.modeling.core import Model

    m = Model(name=f"autocorr{n}_{K}_{objvar}")
    b = [m.integer(f"b{i}", lb=0, ub=1) for i in range(n)]
    s = [2 * bi - 1 for bi in b]
    E = None
    for k in range(1, K + 1):
        Ck = None
        for i in range(n - k):
            t = s[i] * s[i + k]
            Ck = t if Ck is None else Ck + t
        E = Ck * Ck if E is None else E + Ck * Ck
    if objvar:
        tau = m.continuous("objvar", lb=-1e20, ub=1e20)
        m.subject_to(tau >= E)
        m.minimize(tau)
    else:
        m.minimize(E)
    best = min(
        sum(
            sum((2 * bits[i] - 1) * (2 * bits[i + k] - 1) for i in range(n - k)) ** 2
            for k in range(1, K + 1)
        )
        for bits in itertools.product([0, 1], repeat=n)
    )
    return m, float(best), "min"


def cubic(n, seed):
    """Random cubic + quadratic pseudo-Boolean minimization with a cardinality row."""
    import discopt.modeling as dm

    rng = np.random.default_rng(seed)
    tri = [tuple(sorted(rng.choice(n, 3, replace=False))) for _ in range(2 * n)]
    tw = rng.integers(-20, 21, len(tri)).astype(float)
    Qm = np.triu(rng.integers(-10, 11, (n, n)).astype(float) * (rng.uniform(size=(n, n)) < 0.3), 1)
    h = rng.integers(-10, 11, n).astype(float)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n) if Qm[i, j] != 0]
    m = dm.Model(f"cubic{n}_{seed}")
    x = m.binary("x", shape=(n,))
    m.minimize(
        dm.sum(lambda t: tw[t] * x[tri[t][0]] * x[tri[t][1]] * x[tri[t][2]], over=range(len(tri)))
        + dm.sum(lambda p: Qm[pairs[p]] * x[pairs[p][0]] * x[pairs[p][1]], over=range(len(pairs)))
        + dm.sum(lambda i: h[i] * x[i], over=range(n))
    )
    m.subject_to(dm.sum(lambda i: x[i], over=range(n)) <= n // 2)
    X = np.array(list(itertools.product([0, 1], repeat=n)), dtype=np.float64)
    vals = np.einsum("ki,ij,kj->k", X, Qm, X) + X @ h
    for (a, b_, c), w in zip(tri, tw):
        vals += w * X[:, a] * X[:, b_] * X[:, c]
    vals = np.where(X.sum(1) <= n // 2, vals, np.inf)
    return m, float(vals.min()), "min"


def corpus(name):
    from discopt.modeling.core import from_nl

    m = from_nl(os.path.join(REPO, "python/tests/data/minlplib_nl", name + ".nl"))
    return m, None, "max" if "max" in str(m._objective.sense).lower() else "min"


def instances():
    out = [
        ("corpus", "st_miqp1", lambda: corpus("st_miqp1")),
        ("corpus", "st_test1", lambda: corpus("st_test1")),
    ]
    out.append(("dopants", "4x4_k4", lambda: dopants(4, 4)))
    for s in range(3):
        out.append(("dopants", f"4x4_k5_s{s}", lambda s=s: dopants(4, 5, seed=s)))
    out.append(("dopants", "5x5_k6", lambda: dopants(5, 6)))
    for s in range(4):
        out.append(("maxcut", f"16_s{s}", lambda s=s: maxcut(16, 0.4, s)))
    for s in range(2):
        out.append(("maxcut", f"30_s{s}", lambda s=s: maxcut(30, 0.2, s)))
    for s in range(4):
        out.append(("bqp", f"18_s{s}", lambda s=s: bqp(18, 0.3, s)))
    for s in range(2):
        out.append(("bqp", f"30_s{s}", lambda s=s: bqp(30, 0.2, s)))
    for s in range(4):
        out.append(("qknap", f"16_s{s}", lambda s=s: qknap(16, s)))
    for s in range(3):
        out.append(("kcard", f"18_s{s}", lambda s=s: kcard(18, 6, s)))
    for s in range(3):
        out.append(("facility", f"6x10_s{s}", lambda s=s: facility(6, 10, s)))
    # Degree >= 3 (the pre-existing class): under the flag only its MILP engine
    # changes (the default MILP route instead of the forced Rust engine).
    for n, K in ((10, 3), (12, 4), (14, 5)):
        for objvar in (False, True):
            out.append(
                (
                    "autocorr",
                    f"{n}_{K}_{'obj' if objvar else 'expr'}",
                    lambda n=n, K=K, o=objvar: autocorr(n, K, o),
                )
            )
    for s in range(4):
        out.append(("cubic", f"14_s{s}", lambda s=s: cubic(14, s)))
    return out


# ── one arm ──────────────────────────────────────────────────────────────────


def run_arm(build, flag_on, tl):
    os.environ[FLAG] = "1" if flag_on else "0"
    m, oracle, sense = build()
    t0 = time.perf_counter()
    r = m.solve(time_limit=tl)
    wall = time.perf_counter() - t0
    return dict(
        status=r.status,
        objective=r.objective,
        bound=r.bound,
        certified=bool(r.gap_certified),
        nodes=int(r.node_count or 0),
        wall=wall,
        route=(r.algorithm_route or "")[:60],
        oracle=oracle,
        sense=sense,
        feasible=_verify(m, r),
    )


def _verify(m, r):
    """Independent check of the incumbent on the ORIGINAL model: box,
    integrality, every row (``_infer_constraint_bounds``, 1e-6 relative), and the
    reported objective re-evaluated. ``None`` when there is no incumbent."""
    if r.x is None:
        return None
    from discopt._tape_nlp_evaluator import make_evaluator
    from discopt.solvers.nlp_ipopt import _infer_constraint_bounds

    x = np.concatenate([np.asarray(r.x[v.name], dtype=float).ravel() for v in m._variables])
    off = 0
    for v in m._variables:
        xs = x[off : off + v.size]
        if np.any(xs < np.ravel(v.lb) - 1e-6) or np.any(xs > np.ravel(v.ub) + 1e-6):
            return False
        if v.var_type.value in ("binary", "integer") and np.any(np.abs(xs - np.round(xs)) > 1e-5):
            return False
        off += v.size
    ev = make_evaluator(m)
    if ev.n_constraints:
        g = np.asarray(ev.evaluate_constraints(x), dtype=float)
        lo, hi = _infer_constraint_bounds(ev)
        tol = 1e-6 * (1.0 + np.abs(g))
        if np.any(g < lo - tol) or np.any(g > hi + tol):
            return False
    # The evaluator works in minimization sense (a maximize objective negated).
    from discopt.modeling.core import ObjectiveSense

    obj = float(ev.evaluate_objective(x))
    if m._objective.sense == ObjectiveSense.MAXIMIZE:
        obj = -obj
    return abs(obj - float(r.objective)) <= 1e-6 * (1.0 + abs(obj))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import discopt

    print("discopt", discopt.__file__, flush=True)
    print("load", os.getloadavg(), flush=True)
    rows, executed = [], 0
    for fam, name, build in instances():
        for flag_on in (False, True):
            rec = run_arm(build, flag_on, args.time_limit)
            rec.update(family=fam, name=name, arm="ON" if flag_on else "OFF")
            rows.append(rec)
            print(json.dumps(rec), flush=True)
        executed += 1
    tally(rows)
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1)
    print(f"executed comparisons: {executed}", flush=True)
    if executed == 0:
        sys.exit(1)


def tally(rows):
    by = {}
    for r in rows:
        by.setdefault((r["family"], r["name"]), {})[r["arm"]] = r
    false_cert = lost = gained = disagree = bad_points = 0
    w = {"OFF": 0.0, "ON": 0.0}
    nodes = {"OFF": 0, "ON": 0}
    cert = {"OFF": 0, "ON": 0}
    for key, arms in by.items():
        for a in ("OFF", "ON"):
            r = arms[a]
            w[a] += r["wall"]
            nodes[a] += r["nodes"]
            cert[a] += r["certified"]
            if r["feasible"] is False:
                bad_points += 1
            o = r["oracle"]
            if o is not None and r["certified"]:
                tol = 1e-6 + 1e-4 * abs(o)
                if abs(r["objective"] - o) > tol:
                    false_cert += 1
            if o is not None and r["bound"] is not None:
                tol = 1e-6 + 1e-4 * abs(o)
                if (r["sense"] == "min" and r["bound"] > o + tol) or (
                    r["sense"] == "max" and r["bound"] < o - tol
                ):
                    false_cert += 1
        off, on = arms["OFF"], arms["ON"]
        if off["certified"] and not on["certified"]:
            lost += 1
        if on["certified"] and not off["certified"]:
            gained += 1
        if off["certified"] and on["certified"]:
            o1, o2 = off["objective"], on["objective"]
            if abs(o1 - o2) > 1e-6 + 1e-4 * max(abs(o1), abs(o2)):
                disagree += 1
    print(
        f"TALLY pairs={len(by)} false_cert={false_cert} bad_points={bad_points} "
        f"lost={lost} gained={gained} disagree={disagree} "
        f"certified OFF/ON={cert['OFF']}/{cert['ON']} wall OFF/ON={w['OFF']:.1f}/{w['ON']:.1f} "
        f"nodes OFF/ON={nodes['OFF']}/{nodes['ON']}",
        flush=True,
    )


if __name__ == "__main__":
    main()

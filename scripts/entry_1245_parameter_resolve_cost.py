#!/usr/bin/env python
"""Entry experiment for #1245 — what does re-solving across ``Parameter`` value
changes actually cost, and how much of it is reusable?

#1245 asks for compiled relaxations to be reused across ``Parameter`` value
changes, citing ``_relax/relaxation_compiler.py`` (which bakes values in as
constants) and asking for "relaxation compile time only on the first solve" over
100 re-solves.

This script measures the five things that decide whether that is worth building,
on a CALPHAD-shaped phase-pricing model — the workload behind the request:

    minimize  G(y) - mu^T y   over the site-fraction simplex
    G(y) = sum_i G_i0 y_i + R T sum_i y_i ln y_i
           + sum_{i<j} y_i y_j sum_k L_k (y_i - y_j)^k

  A. is ``relaxation_compiler`` on the default solve path at all?
  B. how big is the per-solve relaxation-analysis rebuild, by model size, and
     what is it made of?
  C. how much of the canonical DAG survives a ``mu`` change by CONTENT — the
     ceiling on any content-addressed reuse scheme?
  D. does a solve with a parameter change cost more than one without
     (interleaved A/B, the control the issue's premise implies)?
  E. what does the tape evaluator's per-evaluation staleness check cost?

Every arm prints an executed-assertion/comparison count and the script exits
non-zero if any arm measured nothing (CLAUDE.md §6).

Usage:  python -u scripts/entry_1245_parameter_resolve_cost.py [--quick]
"""

from __future__ import annotations

import argparse
import collections
import sys
import time

import discopt.modeling as dm
import numpy as np

R_GAS = 8.314
EXECUTED: collections.Counter = collections.Counter()


def build_pricing_model(n=3, rk_order=1, seed=0):
    """A CALPHAD phase-pricing model with ``mu`` and ``T`` as parameters."""
    rng = np.random.default_rng(seed)
    m = dm.Model(f"calphad_pricing_n{n}_rk{rk_order}")
    y = [m.continuous(f"y{i}", lb=1e-6, ub=1.0) for i in range(n)]
    mu = [m.parameter(f"mu{i}", 0.0) for i in range(n)]
    T = m.parameter("T", 1000.0)
    G0 = rng.normal(0.0, 5000.0, size=n)
    L = rng.normal(0.0, 20000.0, size=(n, n, rk_order + 1))

    G = sum(float(G0[i]) * y[i] for i in range(n))
    G = G + R_GAS * T * sum(dm.xlogx(y[i]) for i in range(n))
    for i in range(n):
        for j in range(i + 1, n):
            excess = sum(float(L[i, j, k]) * (y[i] - y[j]) ** k for k in range(rk_order + 1))
            G = G + y[i] * y[j] * excess

    m.minimize(G - sum(mu[i] * y[i] for i in range(n)))
    m.subject_to(sum(y) == 1.0)
    return m, y, mu, T


def _timer(mod, name, label, timers, counts):
    orig = getattr(mod, name)
    assert callable(orig), f"{mod}.{name} is not callable"

    def timed(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig(*a, **kw)
        finally:
            timers[label] += time.perf_counter() - t0
            counts[label] += 1

    setattr(mod, name, timed)


# --------------------------------------------------------------------------- #
# A. Is the relaxation compiler the issue names on the default solve path?
# --------------------------------------------------------------------------- #
def arm_compiler_on_path(reps=3):
    import discopt._relax.relaxation_compiler as rc

    calls = collections.Counter()
    for name in ("compile_relaxation", "_compile_relax_node"):
        orig = getattr(rc, name)

        def make(orig=orig, name=name):
            def counted(*a, **kw):
                calls[name] += 1
                return orig(*a, **kw)

            return counted

        setattr(rc, name, make())

    m, _y, mu, _T = build_pricing_model(n=3, rk_order=1)
    rng = np.random.default_rng(4)
    for _ in range(reps):
        for p in mu:
            p.value = float(rng.normal(0, 5000))
        m.solve(max_nodes=60, time_limit=60)
        EXECUTED["A_solves"] += 1

    print(f"[A] default solves: {reps}; relaxation_compiler calls: {dict(calls)}")
    return dict(calls)


# --------------------------------------------------------------------------- #
# B. Per-solve relaxation-analysis rebuild, by model size
# --------------------------------------------------------------------------- #
def arm_cold_fill(sizes=(2, 4, 6, 8), reps=2):
    import discopt._relax.convexity.interval_ad as iad
    import discopt._relax.convexity.rules as cvr
    import discopt._relax.uniform_relax as ur

    timers: collections.Counter = collections.Counter()
    counts: collections.Counter = collections.Counter()
    # NOTE: patch the name ``uniform_relax`` itself calls — it imports
    # ``canonicalize`` into its own namespace, so patching canonical_expr
    # measures nothing (a probe that silently measured nothing, CLAUDE.md §6).
    _timer(ur, "canonicalize", "canonicalize (structure)", timers, counts)
    _timer(cvr, "classify_expr", "classify_expr (declared-box dependent)", timers, counts)
    _timer(iad, "interval_hessian", "interval_hessian (declared-box dependent)", timers, counts)

    builds: list[float] = []
    orig_build = ur.build_uniform_relaxation

    def timed_build(*a, **kw):
        t0 = time.perf_counter()
        try:
            return orig_build(*a, **kw)
        finally:
            builds.append(time.perf_counter() - t0)

    ur.build_uniform_relaxation = timed_build

    rows = []
    print(f"[B] {'n':>3} {'wall/solve':>11} {'first build':>12} {'median':>9} {'cold excess':>12}")
    for n in sizes:
        m, _y, mu, _T = build_pricing_model(n=n, rk_order=2)
        rng = np.random.default_rng(11)
        m.solve(max_nodes=40, time_limit=90)  # warm the process
        wall = first = med = 0.0
        for _ in range(reps):
            for p in mu:
                p.value = float(rng.normal(0, 5000))
            builds.clear()
            timers.clear()
            counts.clear()
            t0 = time.time()
            m.solve(max_nodes=40, time_limit=90)
            wall = time.time() - t0
            EXECUTED["B_solves"] += 1
            if len(builds) >= 3:
                first, med = builds[0], float(np.median(builds[1:]))
        if first <= 0.0:
            print(f"[B] n={n}: no multi-build solve; skipped")
            continue
        EXECUTED["B_size_points"] += 1
        rows.append((n, wall, first, med, first - med, dict(timers)))
        print(
            f"[B] {n:>3} {wall:>10.3f}s {first * 1e3:>11.1f}ms {med * 1e3:>8.1f}ms "
            f"{(first - med) * 1e3:>11.1f}ms  ({100 * (first - med) / wall:.2f}% of the solve)"
        )
        for label, tt in sorted(timers.items(), key=lambda kv: -kv[1]):
            print(f"        {label:44s} {tt * 1e3:7.1f}ms  calls={counts[label]}")
    return rows


# --------------------------------------------------------------------------- #
# C. Content survival across a parameter change (the reuse ceiling)
# --------------------------------------------------------------------------- #
def arm_content_survival(sizes=(3, 6, 8)):
    from discopt._relax.canonical_expr import canonicalize

    def all_nodes(dag):
        seen, out = set(), []
        stack = [n for n in (getattr(dag, "roots", None) or []) if hasattr(n, "key")]
        if not stack:
            for attr in ("nodes", "_nodes", "objective", "constraints"):
                v = getattr(dag, attr, None)
                if isinstance(v, (list, tuple)):
                    stack.extend([n for n in v if hasattr(n, "key")])
                elif hasattr(v, "key"):
                    stack.append(v)
        while stack:
            n = stack.pop()
            if id(n) in seen:
                continue
            seen.add(id(n))
            out.append(n)
            stack.extend(n.children)
        return out

    out = []
    for n in sizes:
        m, _y, mu, T = build_pricing_model(n=n, rk_order=2)
        rng = np.random.default_rng(5)
        for p in mu:
            p.value = float(rng.normal(0, 5000))
        before = {nd.key for nd in all_nodes(canonicalize(m))}
        assert before, "DAG walk found no nodes — the probe is broken"
        for p in mu:
            p.value = float(rng.normal(0, 5000))
        after = all_nodes(canonicalize(m))
        survivors = sum(1 for nd in after if nd.key in before)
        EXECUTED["C_comparisons"] += len(after)
        pct = 100.0 * survivors / len(after)
        out.append((n, len(after), survivors, pct))
        print(
            f"[C] n={n}: {survivors}/{len(after)} canonical nodes survive a mu change ({pct:.1f}%)"
        )

        T.value = float(T.value) + 100.0
        after_T = all_nodes(canonicalize(m))
        surv_T = sum(1 for nd in after_T if nd.key in before)
        EXECUTED["C_comparisons"] += len(after_T)
        pct_T = 100.0 * surv_T / len(after_T)
        print(f"[C] n={n}: {surv_T}/{len(after_T)} survive a T change ({pct_T:.1f}%)")
    return out


# --------------------------------------------------------------------------- #
# D. Does a parameter change cost anything? (interleaved A/B)
# --------------------------------------------------------------------------- #
def arm_ab_change_vs_same(reps=6, n=2, max_nodes=400):
    m, _y, mu, _T = build_pricing_model(n=n, rk_order=0)
    rng = np.random.default_rng(7)
    kw = dict(max_nodes=max_nodes, time_limit=60)
    m.solve(**kw)
    base = [float(rng.normal(0, 5000)) for _ in mu]
    for p, v in zip(mu, base):
        p.value = v
    m.solve(**kw)

    arms: dict[str, list[float]] = {"changed": [], "unchanged": []}
    for _ in range(reps):
        for arm in ("changed", "unchanged"):  # interleaved, never sequential
            if arm == "changed":
                for p in mu:
                    p.value = float(rng.normal(0, 5000))
            else:
                for p, v in zip(mu, base):
                    p.value = v
            t0 = time.time()
            m.solve(**kw)
            arms[arm].append(time.time() - t0)
            EXECUTED["D_solves"] += 1
    for arm, ws in arms.items():
        print(f"[D] {arm:>10}: mean {np.mean(ws):.3f}s  sd {np.std(ws):.3f}  n={len(ws)}")
    return arms


# --------------------------------------------------------------------------- #
# E. The per-evaluation parameter staleness check
# --------------------------------------------------------------------------- #
def arm_staleness_check(calls=20000):
    from discopt._tape_nlp_evaluator import make_evaluator

    m, _y, _mu, _T = build_pricing_model(n=4, rk_order=2)
    ev = make_evaluator(m)
    params = ev._parameters
    snapshot = ev._param_snapshot

    def historical():
        """The pre-#1245 form: rebuild the snapshot tuple, then compare."""
        current = tuple(np.asarray(p.value, dtype=float).copy() for p in params)
        if len(current) != len(snapshot):
            return True
        return any(
            a.shape != b.shape or not np.array_equal(a, b) for a, b in zip(current, snapshot)
        )

    def bench(fn):
        fn()
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        EXECUTED["E_calls"] += calls
        return (time.perf_counter() - t0) / calls

    now = bench(ev._params_changed)
    was = bench(historical)
    assert ev._params_changed() == historical(), "the rewrite changed the verdict"
    EXECUTED["E_equivalence_checks"] += 1
    print(
        f"[E] staleness check with {len(params)} parameters: "
        f"pre-#1245 {was * 1e6:.2f} us/call -> now {now * 1e6:.2f} us/call ({was / now:.1f}x)"
    )
    return was, now


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="smaller grids (~1 min)")
    args = ap.parse_args()

    t0 = time.time()
    arm_compiler_on_path(reps=2 if args.quick else 3)
    arm_cold_fill(sizes=(2, 4) if args.quick else (2, 4, 6, 8), reps=1 if args.quick else 2)
    arm_content_survival(sizes=(3,) if args.quick else (3, 6, 8))
    arm_ab_change_vs_same(reps=2 if args.quick else 6)
    arm_staleness_check(calls=5000 if args.quick else 20000)

    print(f"\nelapsed {time.time() - t0:.1f}s")
    print("EXECUTED:", dict(EXECUTED))
    missing = [
        k
        for k in ("A_solves", "B_size_points", "C_comparisons", "D_solves", "E_calls")
        if EXECUTED[k] == 0
    ]
    if missing:
        print(f"FAIL: these arms measured nothing: {missing}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

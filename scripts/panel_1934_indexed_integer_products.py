"""#1934 graduation panel: ``DISCOPT_INTEGER_PRODUCT_INDEXED`` OFF vs ON.

The in-repo corpus is ``.nl`` (scalar variables), where the flag cannot fire, so
the panel has two parts:

1. **Corpus neutrality** -- for every ``python/tests/data/minlplib_nl/*.nl``,
   the integer-product detectors (bilinear witness, product work, multilinear
   work) give the same answer OFF and ON. Any difference is a failure: the flag
   must be invisible where no array variable exists.
2. **Array-variable panel** -- generated models written with integer array
   variables, the class the flag targets: integer bilinear (pure and with
   continuous partners), integer squares mixed with cross terms, 2-D variables.
   Each instance is solved OFF and ON, interleaved, and checked for
   *cert-clean* (no wrong ``optimal``; dual bound never above the oracle;
   incumbent independently feasible on the ORIGINAL model; no certified-OFF
   instance uncertified ON) and *net-positive* (nodes / wall / bound).

The oracle is brute-force enumeration over the integer box for pure-integer
instances; for mixed instances it is each arm's own certified optimum,
cross-checked between arms (a certified disagreement is a failure).

Usage: ``python -u scripts/panel_1934_indexed_integer_products.py [--time-limit 30]``.
Prints per-instance progress and an executed-assertion count; exits non-zero on
any failure or when nothing was checked (CLAUDE.md §6).
"""

from __future__ import annotations

import argparse
import glob
import itertools
import json
import os
import statistics
import sys
import time

import discopt.modeling as dm
import numpy as np
from discopt._point_eval import compile_point_fn
from discopt._relax import integer_product_reform as ipr

FLAG = "DISCOPT_INTEGER_PRODUCT_INDEXED"
MARKER = "_indexed_refs_enabled"  # CLAUDE.md §8: the version under test has the flag
assert hasattr(ipr, MARKER), ipr.__file__
print("ipr loaded from", ipr.__file__, flush=True)

CHECKS = 0
FAILS: list[str] = []


def check(cond: bool, msg: str) -> None:
    global CHECKS
    CHECKS += 1
    if not cond:
        FAILS.append(msg)
        print("  FAIL:", msg, flush=True)


def set_flag(on: bool) -> None:
    os.environ[FLAG] = "1" if on else "0"


# ---------------------------------------------------------------- part 1


def corpus_neutrality() -> None:
    from discopt.modeling import from_nl

    files = sorted(glob.glob("python/tests/data/minlplib_nl/*.nl"))
    print(f"[corpus] {len(files)} .nl files", flush=True)
    for f in files:
        name = os.path.basename(f)
        try:
            m = from_nl(f)
        except Exception as exc:  # noqa: BLE001 - a parse failure is reported, not hidden
            print(f"  {name}: parse error {type(exc).__name__}: {exc}", flush=True)
            continue
        answers = []
        for on in (False, True):
            set_flag(on)
            answers.append(
                (
                    ipr.has_nonconvex_integer_bilinear(m),
                    ipr.has_reformulation_work(m),
                    ipr.has_integer_multilinear_reformulation_work(m),
                )
            )
        check(answers[0] == answers[1], f"corpus {name}: OFF {answers[0]} != ON {answers[1]}")
        print(f"  {name}: {answers[0]} == {answers[1]}", flush=True)


# ---------------------------------------------------------------- part 2


def _pairs(n, rng, k):
    allp = [(i, j) for i in range(n) for j in range(i + 1, n)]
    idx = rng.choice(len(allp), size=min(k, len(allp)), replace=False)
    return [allp[t] for t in idx]


def gen_int_bilinear(seed, n, ub):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"intbil_n{n}_u{ub}_s{seed}")
    x = m.integer("x", shape=(n,), lb=0, ub=ub)
    for r in range(2):
        body = sum(float(rng.uniform(0.5, 2.0)) * x[i] * x[j] for i, j in _pairs(n, rng, n)) + sum(
            float(rng.uniform(-1, 1)) * x[i] for i in range(n)
        )
        m.subject_to(body <= float(rng.uniform(0.3, 0.6) * n * ub * ub))
    c = rng.uniform(0.5, 2.0, size=n)
    m.minimize(-sum(float(c[i]) * x[i] for i in range(n)))
    return m, [x], True


def gen_mixed(seed, n, ub):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"mixed_n{n}_u{ub}_s{seed}")
    x = m.integer("x", shape=(n,), lb=0, ub=ub)
    y = m.continuous("y", shape=(n,), lb=0.0, ub=5.0)
    for i in range(n):
        m.subject_to(x[i] * y[i] >= float(rng.uniform(1.0, 4.0)))
    m.subject_to(sum(x[i] * x[(i + 1) % n] for i in range(n)) <= float(n * ub))
    cx = rng.uniform(0.5, 3.0, size=n)
    cy = rng.uniform(0.5, 3.0, size=n)
    m.minimize(sum(float(cx[i]) * x[i] + float(cy[i]) * y[i] for i in range(n)))
    return m, [x, y], False


def gen_square_cross(seed, n, ub):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"sqx_n{n}_u{ub}_s{seed}")
    x = m.integer("x", shape=(n,), lb=-ub, ub=ub)
    t = rng.uniform(-ub, ub, size=n)
    obj = sum((x[i] - float(t[i])) ** 2 for i in range(n))
    obj = obj - sum(float(rng.uniform(0.5, 1.5)) * x[i] * x[j] for i, j in _pairs(n, rng, n))
    m.subject_to(sum(x[i] for i in range(n)) <= float(ub))
    m.minimize(obj)
    return m, [x], True


def gen_two_dim(seed, k, ub):
    rng = np.random.default_rng(seed)
    m = dm.Model(f"twod_k{k}_u{ub}_s{seed}")
    w = m.integer("w", shape=(k, k), lb=0, ub=ub)
    for i in range(k):
        m.subject_to(
            sum(w[i, j] * w[j, i] for j in range(k) if j != i) >= float(rng.integers(1, ub + 1))
        )
    c = rng.uniform(0.5, 2.0, size=(k, k))
    m.minimize(sum(float(c[i, j]) * w[i, j] for i in range(k) for j in range(k)))
    return m, [w], True


def gen_dense_qp(seed, n):
    """The #1940 dense integer QP with an array variable -- the class that blocked
    this flag's graduation: ``sum_{i<j} Q_ij x_i x_j + sum x``, ``x in [0,10]^n``,
    ``Q ~ N(0,1)``, ``sum x >= n``. Too large to brute-force; its oracle is the
    certified optimum, cross-checked between arms."""
    rng = np.random.default_rng(seed)
    m = dm.Model(f"dense_n{n}_s{seed}")
    x = m.integer("x", shape=(n,), lb=0, ub=10)
    m.minimize(
        sum(float(rng.normal()) * x[i] * x[j] for i in range(n) for j in range(i + 1, n))
        + sum(x[i] for i in range(n))
    )
    m.subject_to(sum(x[i] for i in range(n)) >= n)
    return m, [x], False


def dense_instances():
    return [gen_dense_qp(s, n) for n in (12, 15, 18, 20) for s in (0, 1)]


def instances():
    out = []
    for s in range(3):
        for n, ub in ((3, 7), (4, 5), (5, 4), (6, 3)):
            out.append(gen_int_bilinear(100 + s, n, ub))
        for n, ub in ((3, 5), (4, 4), (5, 3)):
            out.append(gen_mixed(200 + s, n, ub))
        for n, ub in ((3, 4), (4, 3)):
            out.append(gen_square_cross(300 + s, n, ub))
        for k, ub in ((2, 4), (3, 2)):
            out.append(gen_two_dim(400 + s, k, ub))
    return out


def _flat_box(vars_):
    axes = []
    for v in vars_:
        lb = np.asarray(v.lb, dtype=float).ravel() * np.ones(v.size)
        ub = np.asarray(v.ub, dtype=float).ravel() * np.ones(v.size)
        axes += [range(int(lo), int(hi) + 1) for lo, hi in zip(lb, ub)]
    return axes


def _evaluators(m):
    rows = [(compile_point_fn(c.body, m), c) for c in m._constraints]
    return rows, compile_point_fn(m._objective.expression, m)


def _viol(rows, x, tol):
    for fn, c in rows:
        v = float(np.asarray(fn(x)))
        rhs = float(np.asarray(c.rhs)) if c.rhs is not None else 0.0
        s = c.sense
        if (s == "<=" and v - rhs > tol) or (s == ">=" and rhs - v > tol):
            return True
        if s == "==" and abs(v - rhs) > tol:
            return True
    return False


def brute_force(m, vars_):
    rows, obj = _evaluators(m)
    best = np.inf
    for p in itertools.product(*_flat_box(vars_)):
        x = np.asarray(p, dtype=float)
        if not _viol(rows, x, 1e-9):
            best = min(best, float(np.asarray(obj(x))))
    return best


def x_flat(m, r):
    return np.concatenate([np.asarray(r.x[v.name], dtype=float).ravel() for v in m._variables])


def solve_arm(m, on, tl):
    set_flag(on)
    t0 = time.perf_counter()
    r = m.solve(time_limit=tl)
    wall = time.perf_counter() - t0
    return dict(
        status=r.status,
        obj=r.objective,
        bound=r.bound,
        nodes=int(r.node_count or 0),
        wall=wall,
        cert=bool(getattr(r, "gap_certified", False)),
        x=None if r.x is None else x_flat(m, r),
    )


def panel(tl, dense=False):
    rows_out = []
    for m, vars_, pure in instances() + (dense_instances() if dense else []):
        set_flag(True)
        fires = ipr.has_nonconvex_integer_bilinear(m)
        set_flag(False)
        silent = not ipr.has_nonconvex_integer_bilinear(m)
        check(fires, f"{m.name}: flag ON does not detect the indexed bilinear term")
        check(silent, f"{m.name}: flag OFF detects an indexed term (legacy changed)")
        oracle = brute_force(m, vars_) if pure else None
        # Interleave: OFF, ON, ON, OFF -> two of each, order-balanced.
        res = {False: [], True: []}
        for on in (False, True, True, False):
            res[on].append(solve_arm(m, on, tl))
        rows, _ = _evaluators(m)
        for on in (False, True):
            for a in res[on]:
                tag = f"{m.name} {'ON' if on else 'OFF'}"
                if a["x"] is not None:
                    check(not _viol(rows, a["x"], 1e-6), f"{tag}: incumbent infeasible")
                if oracle is not None and np.isfinite(oracle):
                    if a["bound"] is not None:
                        check(a["bound"] <= oracle + 1e-6, f"{tag}: bound {a['bound']} > {oracle}")
                    if a["status"] == "optimal":
                        check(
                            abs(a["obj"] - oracle) <= 1e-5 * max(1, abs(oracle)),
                            f"{tag}: optimal {a['obj']} != oracle {oracle}",
                        )
        off_opt = [a for a in res[False] if a["status"] == "optimal"]
        on_opt = [a for a in res[True] if a["status"] == "optimal"]
        if off_opt and on_opt:
            check(
                abs(off_opt[0]["obj"] - on_opt[0]["obj"]) <= 1e-5 * max(1, abs(off_opt[0]["obj"])),
                f"{m.name}: OFF {off_opt[0]['obj']} vs ON {on_opt[0]['obj']}",
            )
        if any(a["cert"] for a in res[False]):
            check(any(a["cert"] for a in res[True]), f"{m.name}: certification regression")
        row = dict(
            name=m.name,
            oracle=oracle,
            off_status=[a["status"] for a in res[False]],
            on_status=[a["status"] for a in res[True]],
            off_nodes=statistics.mean(a["nodes"] for a in res[False]),
            on_nodes=statistics.mean(a["nodes"] for a in res[True]),
            off_wall=statistics.mean(a["wall"] for a in res[False]),
            on_wall=statistics.mean(a["wall"] for a in res[True]),
            off_wall_sd=statistics.pstdev(a["wall"] for a in res[False]),
            on_wall_sd=statistics.pstdev(a["wall"] for a in res[True]),
            off_bound=res[False][0]["bound"],
            on_bound=res[True][0]["bound"],
        )
        rows_out.append(row)
        print(
            f"  {m.name:22s} oracle={oracle!s:>10.10} "
            f"OFF {row['off_status'][0]:>8s} n={row['off_nodes']:7.1f} "
            f"t={row['off_wall']:6.2f}+-{row['off_wall_sd']:.2f} | "
            f"ON {row['on_status'][0]:>8s} n={row['on_nodes']:7.1f} "
            f"t={row['on_wall']:6.2f}+-{row['on_wall_sd']:.2f}",
            flush=True,
        )
    return rows_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--skip-corpus", action="store_true")
    ap.add_argument(
        "--dense",
        action="store_true",
        help="also run the dense integer array QPs (n = 12..20) that blocked graduation",
    )
    args = ap.parse_args()
    print("load:", os.getloadavg(), flush=True)
    if not args.skip_corpus:
        corpus_neutrality()
    rows = panel(args.time_limit, dense=args.dense)
    n = len(rows)
    off_solved = sum(r["off_status"][0] == "optimal" for r in rows)
    on_solved = sum(r["on_status"][0] == "optimal" for r in rows)
    fewer_nodes = sum(r["on_nodes"] < r["off_nodes"] for r in rows)
    more_nodes = sum(r["on_nodes"] > r["off_nodes"] for r in rows)
    faster = sum(
        r["on_wall"] < r["off_wall"] - 2 * max(r["off_wall_sd"], r["on_wall_sd"]) for r in rows
    )
    slower = sum(
        r["on_wall"] > r["off_wall"] + 2 * max(r["off_wall_sd"], r["on_wall_sd"]) for r in rows
    )
    tot_off = sum(r["off_wall"] for r in rows)
    tot_on = sum(r["on_wall"] for r in rows)
    print(
        f"\nSUMMARY instances={n} solved OFF {off_solved} ON {on_solved}; "
        f"nodes ON<OFF {fewer_nodes} ON>OFF {more_nodes}; "
        f"wall faster {faster} slower {slower} (beyond 2 sd); "
        f"total wall OFF {tot_off:.1f}s ON {tot_on:.1f}s",
        flush=True,
    )
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(rows, fh, indent=1, default=str)
    print(f"executed checks: {CHECKS}; failures: {len(FAILS)}", flush=True)
    for f in FAILS:
        print("  -", f)
    if CHECKS == 0 or FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()

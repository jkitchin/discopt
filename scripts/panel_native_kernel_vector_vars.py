"""Graduation panel for ``DISCOPT_NATIVE_KERNEL_VECTOR_VARS`` (OFF vs ON).

The flag lets the native Rust spatial kernel take models written with array
variables, which ``spatial_producer.build_spatial_kernel_spec`` used to decline
outright, sending every such model to the Python spatial tree. The panel has three
parts:

1. **Corpus neutrality** -- every ``python/tests/data/minlplib{,_nl}/*.nl`` is
   scalar-variable, so the spec must be identical OFF and ON (same arrays, or
   declined both ways). Any difference is a failure.
2. **Twin identity** -- array models and their scalar twins (the same model written
   with scalar variables) must produce byte-identical kernel specs under the flag.
   This is what makes the array path sound: it IS the scalar path.
3. **Array-variable solve panel** -- the 33 generated models of the #1934 panel
   (brute-force oracle), dense integer array QPs n = 10..20, and the twin families.
   Each is solved OFF and ON, interleaved ABBA, and checked for *cert-clean* (no
   wrong ``optimal``; dual bound never above the oracle / best certified value;
   incumbent independently feasible on the ORIGINAL model; no certified-OFF
   instance uncertified ON) and *net-positive* (wall / certified count).

Usage: ``python -u scripts/panel_native_kernel_vector_vars.py [--time-limit 30]``.
Prints per-solve progress and an executed-check count; exits non-zero on any
failure or when nothing was checked (CLAUDE.md §6).
"""

from __future__ import annotations

import argparse
import glob
import os
import statistics
import sys
import time

import discopt.modeling as dm
import numpy as np
from discopt._point_eval import compile_point_fn
from discopt._relax import spatial_producer as sp
from discopt.modeling import from_nl

sys.path.insert(0, os.path.dirname(__file__))
import panel_1934_indexed_integer_products as P1934  # noqa: E402

# CLAUDE.md §8: the version under test carries the flag.
assert hasattr(sp, "_vector_vars_enabled"), sp.__file__
print("spatial_producer loaded from", sp.__file__, flush=True)

FLAG = "DISCOPT_NATIVE_KERNEL_VECTOR_VARS"
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


def _spec_equal(a, b) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return set(a) == set(b) and all(np.array_equal(np.asarray(a[k]), np.asarray(b[k])) for k in a)


# ---------------------------------------------------------------- twins


def _vars(m, arr, name, n, lb, ub, integer=True):
    """n scalar-variable handles, as one array variable or n scalar ones."""
    make = m.integer if integer else m.continuous
    if arr:
        X = make(name, shape=(n,), lb=lb, ub=ub)
        return [X[i] for i in range(n)]
    lo = np.broadcast_to(np.asarray(lb, dtype=float), (n,))
    hi = np.broadcast_to(np.asarray(ub, dtype=float), (n,))
    return [make(f"{name}{i}", lb=float(lo[i]), ub=float(hi[i])) for i in range(n)]


def qp(n, seed):
    def build(arr):
        rng = np.random.default_rng(seed)
        m = dm.Model(f"qp_n{n}_s{seed}")
        x = _vars(m, arr, "x", n, 0, 10)
        m.minimize(
            sum(float(rng.normal()) * x[i] * x[j] for i in range(n) for j in range(i + 1, n))
            + sum(x)
        )
        m.subject_to(sum(x) >= n)
        return m

    return build


def qp_nonuniform(n, seed):
    def build(arr):
        rng = np.random.default_rng(seed)
        m = dm.Model(f"qpnu_n{n}_s{seed}")
        lo = np.arange(n, dtype=float) % 3
        hi = lo + 2 + (np.arange(n) % 5)
        x = _vars(m, arr, "x", n, lo, hi)
        c = _vars(m, arr, "c", 2, np.array([0.0, 1.0]), np.array([4.0, 9.0]), integer=False)
        m.minimize(
            sum(float(rng.normal()) * x[i] * x[j] for i in range(n) for j in range(i + 1, n))
            + c[0] * x[0]
            - c[1]
        )
        m.subject_to(sum(x) + c[0] + c[1] >= n)
        return m

    return build


def two_d(k, seed):
    def build(arr):
        rng = np.random.default_rng(seed)
        m = dm.Model(f"twod_k{k}_s{seed}")
        if arr:
            W = m.integer("w", shape=(k, k), lb=0, ub=4)
            w = [[W[i, j] for j in range(k)] for i in range(k)]
        else:
            w = [[m.integer(f"w{i}_{j}", lb=0, ub=4) for j in range(k)] for i in range(k)]
        for i in range(k):
            m.subject_to(
                sum(w[i][j] * w[j][i] for j in range(k) if j != i) >= float(rng.integers(1, 5))
            )
        m.minimize(sum(float(rng.uniform(0.5, 2)) * w[i][j] for i in range(k) for j in range(k)))
        return m

    return build


def scalar_then_array(n, seed):
    """A scalar variable before the array and one after it: the offsets must shift."""

    def build(arr):
        rng = np.random.default_rng(seed)
        m = dm.Model(f"sa_n{n}_s{seed}")
        z = m.continuous("z", lb=0.0, ub=3.0)
        x = _vars(m, arr, "x", n, 1, 6)
        t = m.continuous("t", lb=-5.0, ub=5.0)
        m.subject_to(z * x[0] + t * x[1] >= 2.0)
        m.subject_to(sum(x) <= 3 * n)
        m.minimize(sum(float(rng.normal()) * x[i] * x[(i + 1) % n] for i in range(n)) + z - t)
        return m

    return build


TWINS = {
    **{f"qp_n{n}_s{s}": qp(n, s) for n in (10, 12, 15, 18, 20) for s in (0, 1)},
    **{f"qpnu_n{n}": qp_nonuniform(n, 7) for n in (8, 10, 12)},
    **{f"twod_k{k}": two_d(k, 4) for k in (3, 4)},
    **{f"sa_n{n}": scalar_then_array(n, 5) for n in (6, 9)},
}


# ---------------------------------------------------------------- part 1, 2


def corpus_neutrality() -> None:
    files = sorted(glob.glob("python/tests/data/minlplib*/*.nl"))
    print(f"[corpus] {len(files)} .nl files", flush=True)
    for f in files:
        try:
            m = from_nl(f)
        except Exception as exc:  # noqa: BLE001 - a parse failure is reported, not hidden
            print(f"  {os.path.basename(f)}: parse error {type(exc).__name__}", flush=True)
            continue
        check(
            all(int(getattr(v, "size", 1)) == 1 for v in m._variables),
            f"{f}: .nl model has an array variable",
        )
        specs = []
        for on in (False, True):
            set_flag(on)
            specs.append(sp.build_spatial_kernel_spec(m))
        check(_spec_equal(*specs), f"{f}: spec differs OFF vs ON")


def twin_identity() -> None:
    set_flag(True)
    for name, build in TWINS.items():
        a, s = sp.build_spatial_kernel_spec(build(True)), sp.build_spatial_kernel_spec(build(False))
        check(_spec_equal(a, s), f"{name}: array spec != scalar-twin spec")
        print(f"  twin {name}: {'declined both' if a is None else 'identical'}", flush=True)


# ---------------------------------------------------------------- part 3


def _flat_x(m, r):
    return np.concatenate([np.asarray(r.x[v.name], dtype=float).ravel() for v in m._variables])


def _infeasibility(m, x):
    worst = 0.0
    off = 0
    for v in m._variables:
        lo = np.broadcast_to(np.asarray(v.lb, dtype=float), v.shape).reshape(-1)
        hi = np.broadcast_to(np.asarray(v.ub, dtype=float), v.shape).reshape(-1)
        xv = x[off : off + v.size]
        off += v.size
        worst = max(worst, float(np.max(lo - xv, initial=0)), float(np.max(xv - hi, initial=0)))
        if v.var_type.name in ("INTEGER", "BINARY"):
            worst = max(worst, float(np.max(np.abs(xv - np.round(xv)), initial=0)))
    for c in m._constraints:
        val = np.asarray(compile_point_fn(c.body, m)(x), dtype=float).reshape(-1)
        rhs = float(np.asarray(c.rhs)) if c.rhs is not None else 0.0
        scale = max(1.0, abs(rhs))
        if c.sense == "<=":
            worst = max(worst, float(np.max(val - rhs)) / scale)
        elif c.sense == ">=":
            worst = max(worst, float(np.max(rhs - val)) / scale)
        else:
            worst = max(worst, float(np.max(np.abs(val - rhs))) / scale)
    return worst


def panel_instances():
    """(name, builder returning a fresh array-variable model, brute-force oracle or None)."""
    out = []
    for m, vars_, pure in P1934.instances():
        name = m.name
        oracle = P1934.brute_force(m, vars_) if pure else None

        def build(name=name):
            return next(mm for mm, _, _ in P1934.instances() if mm.name == name)

        out.append((name, build, oracle))
    for name, b in TWINS.items():
        out.append((name, (lambda b=b: b(True)), None))
    return out


def solve(build, on, tl):
    set_flag(on)
    m = build()
    t0 = time.perf_counter()
    r = m.solve(time_limit=tl)
    wall = time.perf_counter() - t0
    viol = (
        _infeasibility(m, _flat_x(m, r)) if (r.x is not None and r.objective is not None) else None
    )
    return dict(
        wall=wall,
        status=r.status,
        obj=r.objective,
        bound=r.bound,
        cert=bool(r.gap_certified) and r.status == "optimal",
        nodes=int(r.node_count or 0),
        viol=viol,
        route=(r.algorithm_route or "").split(":")[0],
    )


def panel(tl):
    rows = []
    for name, build, oracle in panel_instances():
        res = {False: [], True: []}
        for on in (True, False, False, True):
            res[on].append(solve(build, on, tl))
        allr = res[False] + res[True]
        cert_objs = [a["obj"] for a in allr if a["cert"]]
        ref = (
            oracle
            if oracle is not None and np.isfinite(oracle)
            else (
                min(cert_objs)
                if cert_objs
                else min((a["obj"] for a in allr if a["obj"] is not None), default=None)
            )
        )
        for on in (False, True):
            for a in res[on]:
                tag = f"{name} {'ON' if on else 'OFF'}"
                if a["viol"] is not None:
                    check(
                        a["viol"] <= 1e-5, f"{tag}: incumbent violates the model by {a['viol']:.3g}"
                    )
                if ref is not None and a["bound"] is not None:
                    check(
                        a["bound"] <= ref + 1e-6 * max(1.0, abs(ref)),
                        f"{tag}: bound {a['bound']} > {ref}",
                    )
                if a["cert"] and ref is not None:
                    check(
                        abs(a["obj"] - ref) <= 1e-4 * max(1.0, abs(ref)) + 1e-6,
                        f"{tag}: certified {a['obj']} != reference {ref}",
                    )
        if any(a["cert"] for a in res[False]):
            check(all(a["cert"] for a in res[True]), f"{name}: certification regression")
        row = dict(
            name=name,
            off=statistics.mean(a["wall"] for a in res[False]),
            on=statistics.mean(a["wall"] for a in res[True]),
            off_sd=statistics.pstdev(a["wall"] for a in res[False]),
            on_sd=statistics.pstdev(a["wall"] for a in res[True]),
            off_cert=sum(a["cert"] for a in res[False]),
            on_cert=sum(a["cert"] for a in res[True]),
            on_route=res[True][0]["route"],
            off_route=res[False][0]["route"],
        )
        rows.append(row)
        print(
            f"  {name:22s} OFF {row['off']:6.2f}+-{row['off_sd']:.2f} cert {row['off_cert']} "
            f"[{row['off_route']}] | ON {row['on']:6.2f}+-{row['on_sd']:.2f} cert {row['on_cert']} "
            f"[{row['on_route']}]",
            flush=True,
        )
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--skip-corpus", action="store_true")
    args = ap.parse_args()
    print("load:", os.getloadavg(), flush=True)
    if not args.skip_corpus:
        corpus_neutrality()
    twin_identity()
    rows = panel(args.time_limit)
    tot_off = sum(r["off"] for r in rows)
    tot_on = sum(r["on"] for r in rows)
    faster = [
        r["name"]
        for r in rows
        if r["on"] < r["off"] - 2 * max(r["on_sd"], r["off_sd"], 0.05) and r["on"] < r["off"] / 1.2
    ]
    slower = [
        r["name"]
        for r in rows
        if r["on"] > r["off"] + 2 * max(r["on_sd"], r["off_sd"], 0.05) and r["on"] > 1.2 * r["off"]
    ]
    cert_off = sum(r["off_cert"] > 0 for r in rows)
    cert_on = sum(r["on_cert"] > 0 for r in rows)
    print(
        f"\nSUMMARY instances={len(rows)} total wall OFF {tot_off:.1f}s ON {tot_on:.1f}s; "
        f"certified OFF {cert_off} ON {cert_on}; "
        f"faster {len(faster)} slower {len(slower)} {slower}",
        flush=True,
    )
    print(f"executed checks: {CHECKS}; failures: {len(FAILS)}", flush=True)
    for f in FAILS:
        print("  -", f)
    if CHECKS == 0 or FAILS:
        sys.exit(1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""E0 (scip-parity-kernel-plan): export real node LPs for the Rust warm-bench.

Exports standard-form LPs `min c'x s.t. Ax = b, l <= x <= u` (dense row-major)
in the E0LPBIN1 binary format consumed by
`crates/discopt-core/src/bin/e0_warm_bench.rs`:

    magic  8 bytes  b"E0LPBIN1"
    u64    m, n
    f64[n] c ; f64[m*n] a ; f64[m] b ; f64[n] l ; f64[n] u
    u64    n_cand ; u64[n_cand] cand   (integer/branchable structural columns)

Instances (the kernel-realistic node LPs):
  * rsyn0805m, syn40m — the convex NLP-BB root LP after OA convergence + the
    #781 GMI/c-MIR/cover cut loop (replicates `_root_cuts.generate_root_cuts`
    with the same components; the LP a P1 kernel node would re-solve).
  * tanksize, nvs09 — the spatial McCormick node LP over the FBBT root box,
    pulled from the production relaxer's own arrays (`_c/_A_ub/_b_ub/_bounds`).

Also runs an optional Python-binding breadth bench (--pybench) through
`solve_lp_warm_std` for the FFI-overhead comparison arm.

Measurement harness only — no solver code changes.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np

BENCH = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/")
VEND = "python/tests/data/minlplib_nl/"

MAGIC = b"E0LPBIN1"


def write_e0(path, c, a, b, lo, up, cand):
    # normalize the ±1e20 effective-infinity sentinels to real ±inf — the
    # simplex treats huge finite bounds as data and its scaling blows up
    # (syn40m cold exit `Numerical` with -1e20 lower bounds; matches the
    # `milp_simplex.py` `_INF` convention)
    lo = np.where(np.asarray(lo, float) <= -1e20, -np.inf, lo)
    up = np.where(np.asarray(up, float) >= 1e20, np.inf, up)
    m, n = a.shape
    assert c.shape == (n,) and b.shape == (m,) and lo.shape == (n,) and up.shape == (n,)
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<QQ", m, n))
        for arr in (c, a.ravel(), b, lo, up):
            f.write(np.ascontiguousarray(arr, dtype="<f8").tobytes())
        f.write(struct.pack("<Q", len(cand)))
        f.write(np.ascontiguousarray(np.asarray(cand, dtype="<u8")).tobytes())
    print(f"  wrote {path}  (m={m}, n={n}, cand={len(cand)})")


def std_form(a_le, b_le, a_eq, b_eq, c_struct, lo_struct, up_struct):
    """[A_le | I ; A_eq | 0] x = [b_le ; b_eq], slack in [0, inf)."""
    m_le, n = a_le.shape
    m_eq = a_eq.shape[0]
    a = np.zeros((m_le + m_eq, n + m_le))
    a[:m_le, :n] = a_le
    a[:m_le, n:] = np.eye(m_le)
    if m_eq:
        a[m_le:, :n] = a_eq
    b = np.concatenate([b_le, b_eq])
    c = np.concatenate([c_struct, np.zeros(m_le)])
    lo = np.concatenate([lo_struct, np.zeros(m_le)])
    up = np.concatenate([up_struct, np.full(m_le, np.inf)])
    return c, a, b, lo, up


def export_convex(name, out):
    """OA-converged root LP + #781 cut loop -> standard form."""
    import discopt.modeling as dm
    from discopt._jax.gdp_reformulate import reformulate_gdp
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt._rust import model_to_repr
    from discopt.modeling.core import ObjectiveSense, VarType
    from discopt.solvers import _root_cuts as rc

    m = reformulate_gdp(dm.from_nl(BENCH + name + ".nl"), method="big-m")
    ev = NLPEvaluator(m)
    repr_ = model_to_repr(m, getattr(m, "_builder", None))
    lb, ub = (np.asarray(x, float) for x in repr_.fbbt(max_iter=20, tol=1e-9))
    is_int = np.array([v.var_type in (VarType.BINARY, VarType.INTEGER) for v in m._variables])
    is_bin = np.array([v.var_type == VarType.BINARY for v in m._variables])
    sense_max = m._objective.sense == ObjectiveSense.MAXIMIZE
    root = rc._RootLP(m, ev, lb, ub, is_int, is_bin, sense_max)

    # OA convergence + the #781 cut loop (same components as generate_root_cuts)
    from discopt._jax.cmir_cuts import separate_cmir
    from discopt._jax.cover_cuts import separate_cover_cuts

    cuts_a, cuts_b = [], []
    pool = rc._CutPool()

    def add_oa(x):
        added = 0
        for i, v in root.nonlinear_violations(x).items():
            if v > rc.OA_TOL:
                t = root.oa_tangent(i, x)
                if t is not None:
                    cuts_a.append(t[0])
                    cuts_b.append(t[1])
                    added += 1
        return added

    def oa_converge():
        obj, x, duals, h = rc._solve_lp(root, cuts_a, cuts_b)
        for _ in range(rc.OA_MAX_ITERS):
            if x is None or add_oa(x) == 0:
                break
            obj, x, duals, h = rc._solve_lp(root, cuts_a, cuts_b)
        return obj, x, duals, h

    obj, x, duals, h = oa_converge()
    lb_s = np.where(np.isfinite(root.lb_sep), root.lb_sep, 0.0)
    ub_s = np.where(
        np.isfinite(np.minimum(root.ub, root.ub_sep)), np.minimum(root.ub, root.ub_sep), 1e5
    )
    for _ in range(rc.ROUNDS):
        if x is None:
            break
        a_all = np.vstack([root.A_le] + ([np.array(cuts_a)] if cuts_a else []))
        b_all = np.concatenate([root.b_le] + ([np.array(cuts_b)] if cuts_b else []))
        cands = list(separate_cmir(a_all, b_all, x, lb_s, ub_s, is_int, max_cuts=24, duals=duals))
        for cov, rhs in separate_cover_cuts(a_all, b_all, x, is_bin, max_cuts=32):
            arr = np.zeros(root.n)
            arr[list(cov)] = 1.0
            cands.append((arr, float(rhs)))
        if h is not None:
            cands += rc.separate_gmi(root, h, x, a_all, b_all)
        for arr, r in cands:
            arr = np.asarray(arr, float)
            if arr @ x - float(r) > rc.CUT_VIOL_TOL:
                pool.offer(arr, float(r))
        chosen = rc._select_cuts(pool.violated(x), x)
        if not chosen:
            break
        for arr, r in chosen:
            cuts_a.append(arr)
            cuts_b.append(r)
        obj, x, duals, h = oa_converge()

    a_le = np.vstack([root.A_le] + ([np.array(cuts_a)] if cuts_a else []))
    b_le = np.concatenate([root.b_le] + ([np.array(cuts_b)] if cuts_b else []))
    c_struct = -root.c if sense_max else root.c  # min form
    lo = np.where(np.isfinite(root.lb), root.lb, -1e20)
    up = np.where(np.isfinite(root.ub), root.ub, np.inf)
    c, a, b, lo2, up2 = std_form(a_le, b_le, root.A_eq, root.b_eq, c_struct, lo, up)
    cand = np.nonzero(is_int)[0]
    print(f"{name}: root LP bound {obj:.4f}, rows le={a_le.shape[0]} eq={root.A_eq.shape[0]}")
    write_e0(out, c, a, b, lo2, up2, cand)


def export_spatial(name, out):
    """McCormick node LP over the FBBT root box, from the production relaxer."""
    import discopt.modeling as dm
    import scipy.sparse as sp
    from discopt._jax.mccormick_lp import MccormickLPRelaxer
    from discopt._rust import model_to_repr
    from discopt.modeling.core import VarType

    m = dm.from_nl(VEND + name + ".nl")
    repr_ = model_to_repr(m, getattr(m, "_builder", None))
    lb, ub = (np.asarray(x, float) for x in repr_.fbbt(max_iter=20, tol=1e-9))
    rel = MccormickLPRelaxer(m)
    # Build the node LP the way the relaxer does (uniform factorable engine),
    # over the FBBT box — the base lifted relaxation a kernel node re-solves.
    from discopt._jax.milp_relaxation import build_milp_relaxation

    mm, _varmap = build_milp_relaxation(m, rel._terms, rel._disc, bound_override=(lb, ub))
    c_struct = np.asarray(mm._c, float).ravel()
    a_ub = (
        sp.csr_matrix(mm._A_ub).toarray() if mm._A_ub is not None else np.zeros((0, len(c_struct)))
    )
    b_ub = np.asarray(mm._b_ub, float).ravel() if mm._b_ub is not None else np.zeros(0)
    bounds = mm._bounds
    lo = np.array([(-1e20 if bl is None else float(bl)) for bl, _ in bounds])
    up = np.array([(np.inf if bu is None else float(bu)) for _, bu in bounds])
    c, a, b, lo2, up2 = std_form(
        a_ub, b_ub, np.zeros((0, len(c_struct))), np.zeros(0), c_struct, lo, up
    )
    # branch candidates: integer structural columns (spatial vars branch too, but
    # E0's bound-flip pattern uses the integer ones; spatial refresh is E1)
    n_orig = len(m._variables)
    is_int = np.array([v.var_type in (VarType.BINARY, VarType.INTEGER) for v in m._variables])
    cand = np.nonzero(is_int)[0] if is_int.any() else np.arange(min(8, n_orig))
    print(f"{name}: lifted LP rows={a_ub.shape[0]} cols={len(c_struct)} (struct orig {n_orig})")
    write_e0(out, c, a, b, lo2, up2, cand)


def pybench(path, trials=500):
    """Python-binding breadth bench (FFI arm): same flips, per-call from Python."""
    from discopt._rust import solve_lp_warm_py  # type: ignore

    with open(path, "rb") as f:
        assert f.read(8) == MAGIC
        m, n = struct.unpack("<QQ", f.read(16))
        c = np.frombuffer(f.read(8 * n))
        a = np.frombuffer(f.read(8 * m * n)).reshape(m, n).copy()
        b = np.frombuffer(f.read(8 * m))
        lo = np.frombuffer(f.read(8 * n)).copy()
        up = np.frombuffer(f.read(8 * n)).copy()
        (n_cand,) = struct.unpack("<Q", f.read(8))
        cand = np.frombuffer(f.read(8 * n_cand), dtype="<u8").astype(int)
    status, x, obj, iters, col_status, basic, _y, _rc = solve_lp_warm_py(c, a, b, lo, up)
    print(f"  py cold: {status} obj={obj:.6g} iters={iters}")
    x = np.asarray(x)
    frac = [j for j in cand if 1e-6 < x[j] - np.floor(x[j]) < 1 - 1e-6] or list(cand)
    t0 = time.perf_counter()
    done = 0
    for t in range(trials):
        j = frac[t % len(frac)]
        keep_l, keep_u = lo[j], up[j]
        if t % 2 == 0:
            up[j] = np.floor(x[j])
        else:
            lo[j] = np.ceil(x[j])
        solve_lp_warm_py(c, a, b, lo, up, col_status, basic)
        lo[j], up[j] = keep_l, keep_u
        done += 1
    dt = time.perf_counter() - t0
    rate = done / dt
    print(f"  py warm breadth: {done} solves, {dt:.2f}s -> {rate:.0f}/s, {1e6 / rate:.0f} us")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/tmp/e0_lps")
    ap.add_argument("--pybench", action="store_true")
    ap.add_argument("--trials", type=int, default=500)
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    jobs = [
        ("rsyn0805m", export_convex),
        ("syn40m", export_convex),
        ("tanksize", export_spatial),
        ("nvs09", export_spatial),
    ]
    outs = []
    for name, fn in jobs:
        out = os.path.join(args.outdir, f"{name}.e0lp")
        try:
            fn(name, out)
            outs.append(out)
        except Exception as e:
            print(f"{name}: EXPORT FAILED — {e}", file=sys.stderr)
    if args.pybench:
        for out in outs:
            print(f"pybench {os.path.basename(out)}:")
            try:
                pybench(out, trials=args.trials)
            except Exception as e:
                print(f"  pybench failed: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()

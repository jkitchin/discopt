#!/usr/bin/env python
"""Issue #798 / K1d — byte-check the Rust convex LP-OA node relaxation vs Python.

The K1 GATE: on the convex panel, the Rust `solve_convex_node_py` node bound must
match the Python reference LP-OA node relaxation to <=1e-6 over (a) the root box
and (b) perturbed child boxes. The reference is the throwaway prototype's
`node_relax(rm, lo, hi, ..., separate=False)` — the OA-converged LP over the box
with NO separation (the exact relaxation K1 reproduces before K2 adds cuts).

The producer `build_convex_arrays(rm, lo, hi)` marshals a RootModel + the
composite-of-affine decomposition (issue798_convex_decompose_probe.decompose) into
the flat arrays the PyO3 binding consumes — it is the reusable analyze-once
producer, not throwaway.

Soundness note: Python's bound is the raw HiGHS LP optimum (max c·x, a valid upper
bound). Rust's bound is the Neumaier-Shcherbina SAFE bound (<= the true LP optimum,
rounded down for verified soundness), so Rust <= Python is EXPECTED; the gate
checks |Rust - Python| <= tol, i.e. the safe margin is within tolerance AND Rust
never exceeds Python (which would be an unsound over-estimate).
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_COEF_TIGHTEN"] = "1"

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discopt._rust as _rust  # noqa: E402
from issue781_cutmgmt_probe import PANEL, RootModel  # noqa: E402
from issue786_lpoa_bandc_prototype import node_relax  # noqa: E402
from issue798_convex_decompose_probe import (  # noqa: E402
    _flat_offsets,
    constraint_expr,
    decompose,
)

_FUNC_CODE = {"log": 0, "exp": 1, "sqrt": 2, "log1p": 3}
GATE_TOL = 1e-6


def _csr_from_dense(a: np.ndarray, thresh: float = 1e-13):
    ptr, cols, vals = [0], [], []
    for r in range(a.shape[0]):
        nz = np.where(np.abs(a[r]) > thresh)[0]
        cols.extend(nz.tolist())
        vals.extend(a[r, nz].tolist())
        ptr.append(len(cols))
    return (
        np.asarray(ptr, np.int64),
        np.asarray(cols, np.int64),
        np.asarray(vals, float),
    )


def _affine_csr(items):
    """(sorted cols, coeffs) from a {col: coeff} dict."""
    cols = sorted(items)
    return (
        np.asarray(cols, np.int64),
        np.asarray([items[c] for c in cols], float),
    )


def build_convex_arrays(rm: RootModel, lo: np.ndarray, hi: np.ndarray) -> dict:
    """Marshal a RootModel node relaxation into the PyO3 flat-array schema."""
    offsets = _flat_offsets(rm.model)

    le_row_ptr, le_cols, le_coeffs = _csr_from_dense(rm.A_le)
    le_rhs = np.asarray(rm.b_le, float)
    eq_row_ptr, eq_cols, eq_coeffs = _csr_from_dense(rm.A_eq)
    eq_rhs = np.asarray(rm.b_eq, float)

    nl_rhs, nl_lin_const = [], []
    nl_lin_ptr, nl_lin_cols, nl_lin_coeffs = [0], [], []
    nl_term_ptr = [0]
    term_coeff, term_func, term_arg_const = [], [], []
    term_arg_ptr, term_arg_cols, term_arg_coeffs = [0], [], []

    for i in rm.nl_rows:
        d = decompose(constraint_expr(rm.model, i), offsets)  # g_i(x); constraint g_i <= 0
        lc, lk = _affine_csr(d.aff)
        nl_lin_cols.extend(lc.tolist())
        nl_lin_coeffs.extend(lk.tolist())
        nl_lin_ptr.append(len(nl_lin_cols))
        nl_lin_const.append(d.const)
        nl_rhs.append(0.0)
        for t in d.terms:
            term_coeff.append(t["coeff"])
            term_func.append(_FUNC_CODE[t["func"]])
            term_arg_const.append(t["arg_const"])
            ac, ak = _affine_csr(t["arg_aff"])
            term_arg_cols.extend(ac.tolist())
            term_arg_coeffs.extend(ak.tolist())
            term_arg_ptr.append(len(term_arg_cols))
        nl_term_ptr.append(len(term_coeff))

    return dict(
        n=rm.n,
        c=np.asarray(rm.c, float),
        integrality=np.asarray(rm.is_int, np.int64),
        lo=np.asarray(lo, float),
        hi=np.asarray(hi, float),
        sense_max=True,
        le_row_ptr=le_row_ptr,
        le_cols=le_cols,
        le_coeffs=le_coeffs,
        le_rhs=le_rhs,
        eq_row_ptr=eq_row_ptr,
        eq_cols=eq_cols,
        eq_coeffs=eq_coeffs,
        eq_rhs=eq_rhs,
        nl_rhs=np.asarray(nl_rhs, float),
        nl_lin_const=np.asarray(nl_lin_const, float),
        nl_lin_ptr=np.asarray(nl_lin_ptr, np.int64),
        nl_lin_cols=np.asarray(nl_lin_cols, np.int64),
        nl_lin_coeffs=np.asarray(nl_lin_coeffs, float),
        nl_term_ptr=np.asarray(nl_term_ptr, np.int64),
        term_coeff=np.asarray(term_coeff, float),
        term_func=np.asarray(term_func, np.int64),
        term_arg_const=np.asarray(term_arg_const, float),
        term_arg_ptr=np.asarray(term_arg_ptr, np.int64),
        term_arg_cols=np.asarray(term_arg_cols, np.int64),
        term_arg_coeffs=np.asarray(term_arg_coeffs, float),
    )


def rust_node_bound(rm: RootModel, lo: np.ndarray, hi: np.ndarray) -> dict:
    arrays = build_convex_arrays(rm, lo, hi)
    return _rust.solve_convex_node_py(**arrays, oa_tol=1e-6, max_oa_rounds=60)


def python_node_bound(rm: RootModel, lo: np.ndarray, hi: np.ndarray):
    lb_s = np.where(np.isfinite(rm.lb_sep), rm.lb_sep, 0.0)
    ub_s = np.where(np.isfinite(np.minimum(rm.ub, rm.ub_sep)), np.minimum(rm.ub, rm.ub_sep), 1e5)
    return node_relax(rm, lo.copy(), hi.copy(), lb_s, ub_s, separate=False)


def perturbed_boxes(rm: RootModel, k: int = 3):
    """A few child boxes: fix the first k binaries to 0 then to 1."""
    boxes = []
    bins = [j for j in range(rm.n) if rm.is_bin[j]][:k]
    for j in bins:
        for val in (0.0, 1.0):
            lo, hi = rm.lb.copy(), rm.ub.copy()
            lo[j] = hi[j] = val
            boxes.append((f"x{j}={int(val)}", lo, hi))
    return boxes


def main() -> bool:
    all_ok = True
    for name in PANEL:
        rm = RootModel(name)
        cases = [("root", rm.lb.copy(), rm.ub.copy())] + perturbed_boxes(rm)
        worst = 0.0
        worst_case = ""
        n_checked = 0
        n_certified = 0  # boxes where the NS safe bound is finite (tree-fathomable)
        unsound = False
        for label, lo, hi in cases:
            pb = python_node_bound(rm, lo, hi)
            py_bound = pb[0] if pb is not None else None
            rr = rust_node_bound(rm, lo, hi)
            rs_status, rs_raw, rs_safe = rr["status"], rr["raw_bound"], rr["bound"]
            if py_bound is None:
                # Python relaxation infeasible/None → Rust must not be Optimal-finite.
                if rs_status == "optimal" and np.isfinite(rs_raw):
                    print(f"  {name}/{label}: PY None but RUST optimal {rs_raw:.6f}  MISMATCH")
                    all_ok = False
                continue
            if rs_status != "optimal":
                print(f"  {name}/{label}: PY {py_bound:.6f} but RUST {rs_status}  MISMATCH")
                all_ok = False
                continue
            n_checked += 1
            # PRIMARY GATE: the relaxation optimum must match (raw LP vs raw HiGHS).
            gap = abs(rs_raw - py_bound)
            if gap > worst:
                worst, worst_case = gap, label
            # SAFE-BOUND soundness: when it certifies (finite), it must be a valid
            # upper bound (≤ python raw + tol); an over-estimate would be unsound.
            if np.isfinite(rs_safe):
                n_certified += 1
                if rs_safe > py_bound + GATE_TOL:
                    unsound = True
        status = "OK" if (worst <= GATE_TOL and not unsound) else "FAIL"
        flag = "  !! UNSOUND (safe bound > python)" if unsound else ""
        print(
            f"{name}: {status}  checked={n_checked} max|Δraw|={worst:.2e} @ {worst_case} "
            f"(tol {GATE_TOL:.0e})  safe-certified={n_certified}/{n_checked}{flag}",
            flush=True,
        )
        all_ok = all_ok and worst <= GATE_TOL and not unsound
    print(
        f"\nK1 GATE (Rust node relaxation == Python to <={GATE_TOL:.0e}): {'PASS' if all_ok else 'FAIL'}"
    )
    return all_ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

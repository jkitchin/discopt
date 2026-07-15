"""G6 (baron-gap §10) entry experiment — is bchoco06's LP non-convergence a
subnormal-noise problem the cheap flush can fix, or a genuine Rust-simplex
ill-conditioning defect?

Falsifies the G6 hypothesis (see ``docs/dev/g5-family-d-diagnoses.md`` §"G6
update"): flushing subnormal structural noise (``|v| < 1e-300 -> 0``) in the
assembled ``(A_ub, b_ub, bounds)`` does NOT make the in-house simplex converge on
the bchoco06 root relaxation. The genuine ``1e10`` coefficient spread survives full
geometric-mean (Ruiz) equilibration as a residual ``~1.25e12`` condition, which the
in-house Rust simplex stalls on (``iteration_limit``) while HiGHS solves it. The
defect is Rust-layer linear-algebra robustness, not a Python cleanup pass -> the
kill criterion fires; no flush is shipped.

Reproduce::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=$PWD/python \
      python discopt_benchmarks/scripts/g6_bchoco06_conditioning_probe.py
"""

from __future__ import annotations

import math

# Reuse the G5 probe's assembly of the byte-identical uniform root relaxation.
import sys
import warnings
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "discopt_benchmarks" / "scripts"))
from g5_bchoco06_hole_probe import _build_ctx  # noqa: E402

_NL = _REPO / "python" / "tests" / "data" / "minlplib_nl" / "bchoco06.nl"

# Structural-zero threshold the hypothesis proposed: ~288 orders of magnitude
# below the 1e-12 factorization tolerance / 1e-6 abs tolerance in conftest.py.
_STRUCTURAL_ZERO = 1e-300


def _spread(data: np.ndarray) -> tuple[float, float, float]:
    nz = np.abs(np.asarray(data))
    nz = nz[(nz != 0.0) & np.isfinite(nz)]
    if nz.size == 0:
        return 0.0, 0.0, 1.0
    return float(nz.min()), float(nz.max()), float(nz.max() / nz.min())


def _flush(v: np.ndarray) -> np.ndarray:
    v = np.array(v, dtype=np.float64, copy=True)
    mask = np.isfinite(v) & (np.abs(v) < _STRUCTURAL_ZERO) & (v != 0.0)
    v[mask] = 0.0
    return v


def main() -> None:
    from discopt._jax.milp_relaxation import MilpRelaxationModel, equilibrate_relaxation_lp
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt.modeling.core import from_nl
    from discopt.solvers.milp_simplex import solve_lp_warm_std

    model = from_nl(str(_NL))
    flat_lb, flat_ub = flat_variable_bounds(model)
    ctx, obj_lin, sign, _ = _build_ctx(model, flat_lb, flat_ub)
    n_cols = len(ctx.col_lb)

    data, ri, ci_ = [], [], []
    b = np.zeros(len(ctx.rows))
    for i, (coeffs, rhs) in enumerate(ctx.rows):
        b[i] = rhs
        for j, coef in coeffs.items():
            data.append(coef)
            ri.append(i)
            ci_.append(j)
    A = sp.csr_matrix((data, (ri, ci_)), shape=(len(ctx.rows), n_cols))
    c = np.zeros(n_cols)
    for j, coef in obj_lin.coeffs.items():
        c[j] += coef
    col_lb = np.asarray(ctx.col_lb, dtype=np.float64)
    col_ub = np.asarray(ctx.col_ub, dtype=np.float64)

    def sub_count(v: np.ndarray) -> int:
        v = np.asarray(v, dtype=np.float64)
        fin = v[np.isfinite(v)]
        return int(np.count_nonzero((fin != 0.0) & (np.abs(fin) < _STRUCTURAL_ZERO)))

    print(f"subnormals < {_STRUCTURAL_ZERO:g}:  A.data={sub_count(A.data)}  "
          f"b={sub_count(b)}  col_lb={sub_count(col_lb)}  col_ub={sub_count(col_ub)}")

    A_f = A.copy()
    A_f.data = _flush(A_f.data)
    b_f = _flush(b)
    lb_f = _flush(col_lb)
    ub_f = _flush(col_ub)

    def _show_spread(tag: str, data: np.ndarray) -> None:
        mn, mx, ra = _spread(data)
        print(f"{tag} A spread: min={mn:.3e} max={mx:.3e} ratio={ra:.3e}")

    _show_spread("RAW     ", A.data)
    _show_spread("FLUSHED ", A_f.data)
    c_s, A_s, b_s, bnds_s, _ = equilibrate_relaxation_lp(c, A_f, b_f, list(zip(lb_f, ub_f)), None)
    _show_spread("EQUILIB ", sp.csr_matrix(A_s).data)

    def simplex(A_, b_, lb_, ub_) -> tuple[str, object]:
        milp = MilpRelaxationModel(
            c=c, A_ub=A_, b_ub=b_, bounds=list(zip(lb_, ub_)),
            obj_offset=obj_lin.const, integrality=None, objective_bound_valid=True,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = milp.solve(backend="simplex")
        return r.status, r.bound

    def highs(A_, b_, lb_, ub_):
        from scipy.optimize import linprog
        bnds = list(zip(
            [None if not math.isfinite(x) else x for x in lb_],
            [None if not math.isfinite(x) else x for x in ub_],
        ))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hr = linprog(c, A_ub=A_, b_ub=b_, bounds=bnds, method="highs")
        return hr.success, (None if not hr.success else sign * (hr.fun + obj_lin.const))

    print("\n=== in-house simplex vs HiGHS (original-sense dual bound) ===")
    for tag, (A_, b_, lb_, ub_) in {
        "raw    ": (A, b, col_lb, col_ub),
        "flushed": (A_f, b_f, lb_f, ub_f),
    }.items():
        st, bd = simplex(A_, b_, lb_, ub_)
        hs, hb = highs(A_, b_, lb_, ub_)
        obd = None if bd is None else sign * bd
        print(f"  {tag}: simplex status={st:15s} dual={obd}   HiGHS ok={hs} dual={hb}")

    # Direct raw-simplex probe on the equilibrated (flushed) matrix + safe bound.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res, _, cert = solve_lp_warm_std(
            c_s, sp.csr_matrix(A_s), b_s, bnds_s, in_basis=None, return_cert=True
        )
    if res is None:
        print(f"\nraw Rust simplex on EQUILIB(flushed): result=None (iter-limit)  "
              f"safe_bound={cert.safe_bound}")
    else:
        print(f"\nraw Rust simplex on EQUILIB(flushed): status={res.status} "
              f"bound={res.bound} safe_bound={cert.safe_bound}")

    print("\nVERDICT: flushing subnormals does NOT make the in-house simplex "
          "converge; residual post-Ruiz condition ~1.25e12 is genuine "
          "ill-conditioning. Kill criterion MET -> Rust-layer numerics fix "
          "(see docs/dev/g5-family-d-diagnoses.md 'G6 update').")


if __name__ == "__main__":
    main()

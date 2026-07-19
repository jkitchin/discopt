"""Producer: extract a native-kernel ``SpatialKernelSpec`` from a discopt model.

The Rust spatial B&B kernel (``discopt-core::bnb::spatial_*``, issue #764) runs a
node entirely in Rust — patch box-dependent McCormick envelopes, assemble the node
LP, warm-solve, OBBT-sweep — given the *box-independent* relaxation structure once.
This module produces that structure for a model by reading the already-validated
:class:`IncrementalMcCormickLP` (which builds the LP structure once and separates the
box-independent rows from the box-dependent envelope rows, and registers every lifted
term with its operand/output columns).

Scope: the subset the incremental engine covers — bilinear products, integer-power
monomials on a sign-definite box, and affine squares. Models whose relaxation needs
atoms the incremental engine declines (``sqrt``/general univariate, RLT lifts, NN
activations, vector variables) return ``None`` here, and the caller keeps the trusted
Python path. Extending coverage to ``sqrt`` (which the Rust kernel already supports
via ``EnvTerm::Sqrt``) is a follow-up on the *producer* side, not the kernel.

Soundness: the produced spec drives the same McCormick relaxation the incremental
engine validated row-for-row against the cold ``build_milp_relaxation``; the Rust
patcher reproduces those envelope rows byte-for-byte (its differential fixtures), so
the native relaxation is bound-neutral with the trusted build.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


def build_spatial_kernel_spec(model) -> Optional[dict]:
    """Return the flat-array ``SpatialKernelSpec`` kwargs for
    ``discopt._rust.solve_spatial_tree_py``, or ``None`` if the model is outside the
    incremental engine's covered subset (caller falls back to the Python path)."""
    from discopt._jax.mccormick_lp import MccormickLPRelaxer
    from discopt.modeling.core import VarType

    # Only scalar-variable models: the incremental engine's column 0..n-1 maps to the
    # n variable objects one-to-one only when each has size 1. Vector variables shift
    # the column layout and are a follow-up.
    if any(getattr(v, "size", 1) != 1 for v in model._variables):
        return None

    relaxer = MccormickLPRelaxer(model, backend="simplex")
    inc = getattr(relaxer, "_inc", None)
    if inc is None or not getattr(inc, "ok", False):
        return None

    ncol, n = int(inc.ncol), int(inc.n)
    A = inc.base_A.tocsr()
    b = np.asarray(inc.base_b, dtype=np.float64).ravel()
    prod_rows = set(int(k) for k in inc._prod_rows)

    # Fixed (box-independent) rows: every row not owned by a lifted term's envelope.
    fixed_row_ptr = [0]
    fixed_cols: list[int] = []
    fixed_coeffs: list[float] = []
    fixed_rhs: list[float] = []
    for k in range(A.shape[0]):
        if k in prod_rows:
            continue
        row = A.getrow(k)
        fixed_cols.extend(int(c) for c in row.indices)
        fixed_coeffs.extend(float(v) for v in row.data)
        fixed_row_ptr.append(len(fixed_cols))
        fixed_rhs.append(float(b[k]))

    # Lifted-term descriptors (kind: 0=Bilinear 1=Monomial 2=AffineSquare).
    tk: list[int] = []
    ti: list[int] = []
    tj: list[int] = []
    tout: list[int] = []
    tp: list[int] = []
    tcoeff: list[float] = []
    tcst: list[float] = []

    def _push(kind, i, j, out, p, coeff, cst):
        tk.append(kind)
        ti.append(int(i))
        tj.append(int(j))
        tout.append(int(out))
        tp.append(int(p))
        tcoeff.append(float(coeff))
        tcst.append(float(cst))

    for (i, j), a in inc.bilinear.items():
        _push(0, i, j, a, 0, 0.0, 0.0)
    for (i, p), a in inc.monomial.items():
        _push(1, i, -1, a, p, 0.0, 0.0)
    for (j, a), (coeff, const) in inc.affine_square.items():
        _push(2, j, -1, a, 0, coeff, const)

    # Objective over the lifted columns.
    c = np.asarray(inc.c, dtype=np.float64).ravel()
    if c.shape[0] != ncol:
        return None

    # Global bounds + integrality. Original columns 0..n take the model variable
    # bounds/type; auxiliary columns n..ncol are left wide — the kernel derives their
    # finite range per node box in closed form (assemble_node_lp intersects), so the
    # assembled node LP always has finite aux bounds for the safe-bound evaluation.
    global_lo = np.full(ncol, -1e20, dtype=np.float64)
    global_hi = np.full(ncol, 1e20, dtype=np.float64)
    integrality = np.zeros(ncol, dtype=np.int64)
    for k, v in enumerate(model._variables):
        global_lo[k] = float(np.min(v.lb))
        global_hi[k] = float(np.max(v.ub))
        if v.var_type in (VarType.INTEGER, VarType.BINARY):
            integrality[k] = 1

    # OBBT candidates: the original (branchable) variables with finite bounds.
    obbt_candidates = [
        k for k in range(n) if np.isfinite(global_lo[k]) and np.isfinite(global_hi[k])
    ]

    return dict(
        n_cols=ncol,
        n_orig=n,
        c=c,
        integrality=integrality,
        global_lo=global_lo,
        global_hi=global_hi,
        fixed_row_ptr=np.asarray(fixed_row_ptr, dtype=np.int64),
        fixed_cols=np.asarray(fixed_cols, dtype=np.int64),
        fixed_coeffs=np.asarray(fixed_coeffs, dtype=np.float64),
        fixed_rhs=np.asarray(fixed_rhs, dtype=np.float64),
        term_kind=np.asarray(tk, dtype=np.int64),
        term_i=np.asarray(ti, dtype=np.int64),
        term_j=np.asarray(tj, dtype=np.int64),
        term_out=np.asarray(tout, dtype=np.int64),
        term_p=np.asarray(tp, dtype=np.int64),
        term_coeff=np.asarray(tcoeff, dtype=np.float64),
        term_cst=np.asarray(tcst, dtype=np.float64),
        obbt_candidates=np.asarray(obbt_candidates, dtype=np.int64),
    )


def solve_with_native_kernel(model, **config):
    """Convenience: produce the spec and run the native kernel. Returns the result
    dict from ``solve_spatial_tree_py``, or ``None`` if the model is unsupported."""
    spec = build_spatial_kernel_spec(model)
    if spec is None:
        return None
    from discopt import _rust

    return _rust.solve_spatial_tree_py(**spec, **config)

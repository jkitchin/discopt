"""G5 (baron-gap) diagnosis probe — bchoco06 unbounded-relaxation hole.

Answers: *which atom class of which constraint leaves the root LP unbounded, and
why?*  (baron-gap-plan.md §7, task 1.)

We rebuild the uniform factorable root relaxation (``build_uniform_relaxation``'s
own assembly, mirrored here with ``track_aux_exprs=True`` so every aux column
carries the modeling sub-expression it represents), then:

  1. reconstruct the minimize-equivalent objective ``obj_lin`` and every row;
  2. run the ``objective_bound_valid`` machinery (uniform_relax.py ~2237-2271)
     verbatim, printing WHICH column / atom / constraint produced each free
     cost column that is unbounded on its cost-relevant side;
  3. confirm the LP-unbounded direction empirically: clamp the free aux columns
     to a symmetric big-M box and re-solve; if the bound scales linearly with M,
     the columns are genuinely free in the cost direction (an unbounded ray),
     not merely wide.

Reproduce::

    cd <worktree>
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=$PWD/python \
      python discopt_benchmarks/scripts/g5_bchoco06_hole_probe.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_NL = _REPO / "python" / "tests" / "data" / "minlplib_nl" / "bchoco06.nl"


def _build_ctx(model, flat_lb, flat_ub):
    """Mirror build_uniform_relaxation's assembly, keeping ctx + obj_lin."""
    from discopt._jax.uniform_relax import LinForm, _Builder
    from discopt._jax.canonical_expr import canonicalize
    from discopt.modeling.core import ObjectiveSense

    dag = canonicalize(model)
    ctx = _Builder(model, flat_lb, flat_ub, track_aux_exprs=True)

    sign = 1.0
    obj_lin = LinForm()
    if dag.objective is not None:
        obj_lin = ctx.rep(dag.objective)
        if model._objective is not None and model._objective.sense == ObjectiveSense.MAXIMIZE:
            sign = -1.0
            obj_lin = obj_lin.scaled(-1.0)

    # constraint bodies -> rows (index-aligned so we can name the owning constraint)
    row_owner: list[int] = []
    for ci, (con, cnode) in enumerate(zip(model._constraints, dag.constraints)):
        lc = ctx.rep(cnode)
        before = len(ctx.rows)
        sense = con.sense
        rhs_shift = float(con.rhs)
        if sense == "<=":
            ctx.add_row(lc.coeffs, rhs_shift - lc.const)
        elif sense == ">=":
            ctx.add_row(lc.scaled(-1.0).coeffs, -(rhs_shift) + lc.const)
        elif sense == "==":
            ctx.add_row(lc.coeffs, rhs_shift - lc.const)
            ctx.add_row(lc.scaled(-1.0).coeffs, -(rhs_shift) + lc.const)
        row_owner.extend([ci] * (len(ctx.rows) - before))
    return ctx, obj_lin, sign, row_owner


def main() -> None:
    from discopt.modeling.core import from_nl
    from discopt._jax.model_utils import flat_variable_bounds
    from discopt._jax.milp_relaxation import MilpRelaxationModel

    model = from_nl(str(_NL))
    flat_lb, flat_ub = flat_variable_bounds(model)
    ctx, obj_lin, sign, row_owner = _build_ctx(model, flat_lb, flat_ub)

    n_cols = len(ctx.col_lb)
    print(f"bchoco06: {len(model._variables)} vars, {len(model._constraints)} cons, "
          f"sense={model._objective.sense.name}")
    print(f"LP: {n_cols} cols ({ctx.n_orig} orig + {n_cols - ctx.n_orig} aux), "
          f"{len(ctx.rows)} rows")

    # -- which columns appear in at least one row -------------------------- #
    row_cols: set[int] = set()
    for coeffs, _ in ctx.rows:
        row_cols.update(coeffs)

    # -- objective_bound_valid machinery, verbatim + instrumented ---------- #
    print("\n=== objective cost columns (minimize-equivalent) ===")
    obj_bound_valid = True
    obj_box_lb = obj_lin.const
    free_cost_cols = []
    for j, coef in sorted(obj_lin.coeffs.items()):
        edge = ctx.col_lb[j] if coef > 0 else ctx.col_ub[j]
        contrib = coef * edge
        lo, hi = ctx.col_lb[j], ctx.col_ub[j]
        kind = "orig" if j < ctx.n_orig else "aux"
        expr = ctx.aux_expr.get(j)
        note = ""
        if not math.isfinite(contrib):
            in_row = j in row_cols
            note = f"  <-- INFINITE cost-edge; in_row={in_row}"
            if not in_row:
                obj_bound_valid = False
            obj_box_lb = -math.inf
            free_cost_cols.append((j, coef, in_row, expr))
        else:
            obj_box_lb += contrib
        exprs = f"  expr={expr!s:.90}" if expr is not None else ""
        print(f"  col {j:3d} ({kind}) coef={coef:+.4g} box=[{lo:.4g},{hi:.4g}]{note}{exprs}")
    if not math.isfinite(obj_box_lb):
        obj_bound_valid = False
    print(f"\nobj_bound_valid = {obj_bound_valid}   obj_box_lb = {obj_box_lb}")

    # -- the LP solve as shipped ------------------------------------------- #
    import scipy.sparse as sp

    data, ri, ci_ = [], [], []
    b = np.zeros(len(ctx.rows))
    for i, (coeffs, rhs) in enumerate(ctx.rows):
        b[i] = rhs
        for j, coef in coeffs.items():
            data.append(coef); ri.append(i); ci_.append(j)
    A = sp.csr_matrix((data, (ri, ci_)), shape=(len(ctx.rows), n_cols))
    c = np.zeros(n_cols)
    for j, coef in obj_lin.coeffs.items():
        c[j] += coef

    def solve(col_lb, col_ub, valid=obj_bound_valid):
        milp = MilpRelaxationModel(
            c=c, A_ub=A, b_ub=b, bounds=list(zip(col_lb, col_ub)),
            obj_offset=obj_lin.const, integrality=None, objective_bound_valid=valid,
        )
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return milp.solve(backend="simplex")

    res = solve(ctx.col_lb, ctx.col_ub)
    print(f"\nshipped LP solve: status={res.status}  bound={res.bound}")

    # -- free aux columns (both bounds infinite) --------------------------- #
    free_aux = [j for j in range(ctx.n_orig, n_cols)
                if not math.isfinite(ctx.col_lb[j]) or not math.isfinite(ctx.col_ub[j])]
    print(f"\n=== free/semi-infinite aux columns: {len(free_aux)} ===")
    for j in free_aux:
        expr = ctx.aux_expr.get(j)
        print(f"  col {j:3d} box=[{ctx.col_lb[j]:.4g},{ctx.col_ub[j]:.4g}] "
              f"in_obj={j in obj_lin.coeffs} in_row={j in row_cols}  expr={expr!s:.100}")

    # -- big-M scaling test: clamp free aux, does bound scale with M? ------ #
    print("\n=== big-M clamp test (is the unboundedness a free ray?) ===")
    for M in (1e3, 1e4, 1e5, 1e6):
        lb2 = list(ctx.col_lb); ub2 = list(ctx.col_ub)
        for j in free_aux:
            if not math.isfinite(lb2[j]):
                lb2[j] = -M
            if not math.isfinite(ub2[j]):
                ub2[j] = M
        r = solve(lb2, ub2, valid=True)
        # bound is on the minimize-equivalent; report original-sense dual bound too
        orig = None if r.bound is None else sign * r.bound
        print(f"  M={M:.0e}: status={r.status:12s} minLP_bound={r.bound}  "
              f"orig_sense_dual={orig}")


if __name__ == "__main__":
    main()

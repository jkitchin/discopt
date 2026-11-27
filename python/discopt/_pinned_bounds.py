"""Columns that equality rows pin at a bound: the "no strict interior" signature (#2366).

An interior-point method keeps every iterate strictly inside the variable bounds.
When the equality rows of a model determine a column, and the value they determine
is *exactly* one of its bounds, the feasible set has no strictly interior point
near the solution in that coordinate. The barrier then cannot converge: the
iterates press against the bound while the equalities pull them onto it, and the
solve ends at an iteration or time limit with no hint why.

The case that prompted this is a DAE/MOL parameter fit (#2366): surface coverages
declared ``lb=0`` start at 0 and stay exactly 0 until an inlet pulse begins, so
every collocation value before the pulse is 0 *and* determined by the collocation
equations. Measured on the issue's reproducer (39 elements, 3-stage Radau,
``constr_viol_tol=1e-10``):

* ``lb=-1e-8``, ``bound_relax_factor=0``: ``local_optimal``, objective 223.587;
* ``lb=0``, ``bound_relax_factor=0``: ``iteration_limit`` after 450 iterations,
  with 281 coverage columns within 1e-6 of 0 at the final iterate, in elements
  0-12 (the pulse starts at element 4 and ramps as ``sin^2``);
* ``lb=0``, ``bound_relax_factor=1e-8``: ``local_limit``. POUNCE, like Ipopt, caps
  the relaxation at ``constr_viol_tol``, so the slab was 1e-10 wide, not 1e-8.
  With ``constr_viol_tol=1e-8`` the same run converges to 223.587.

Fixing such columns in a presolve would need a proof that every feasible point has
them at the bound. For a dynamic model that is an invariance argument (x(0) = 0
and the dynamics keep x there), which interval reasoning cannot supply, and fixing
a column on a guess changes the feasible set. So this module does not change the
model. It reports the signature after a solve that did not converge, so the user
learns why and what to change.

``solver="pounce"`` also acts on the report, without fixing anything: it re-solves
once with every pinned variable's bound on the pinned side moved outward by
``solver._PINNED_SLAB_RTOL * max(1, |b|)`` (1e-8, Ipopt's default
``bound_relax_factor``, but on those bounds only), projects the point back onto
the declared box, and adopts it only when the re-solve converged and the
projected point passes discopt's 1e-6 feasibility check. That restores a strict
interior without a proof, because the feasible set the user gets is the declared
one: the slab only shapes the path. On the reproducer above it turns
``iteration_limit`` (48 s) into ``local_optimal`` at objective 223.586908.

The test is deliberately simple, and it is a hint, not a proof: a column counts
when it is not fixed, sits within ``rtol * max(1, |bound|)`` of a finite bound at
the final iterate, and appears in at least one equality row's Jacobian structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

#: Distance to a bound, relative to ``max(1, |bound|)``, at which a column of an
#: unconverged interior-point iterate counts as pressed against it. The #2366
#: reproducer's stuck coverages were all within 1e-6 of 0 (and 41 more within
#: 1e-4), while its free states were nowhere near their bounds.
PINNED_RTOL = 1e-6


@dataclass
class PinnedGroup:
    """Pinned columns of one model variable, on one side."""

    variable: str
    side: str  # "lower" or "upper"
    count: int
    bounds: list[float]
    #: Leading-axis indices that hold a pinned column (for a ``DAEBuilder`` /
    #: ``MOLBuilder`` state the leading axis is the finite element). Empty for a
    #: scalar variable.
    leading_indices: list[int] = field(default_factory=list)

    def describe(self) -> str:
        bound = (
            f"{self.bounds[0]:g}"
            if len(self.bounds) == 1
            else f"{min(self.bounds):g}..{max(self.bounds):g}"
        )
        text = f"{self.variable}: {self.count} at {'lb' if self.side == 'lower' else 'ub'}={bound}"
        if self.leading_indices:
            text += f" (first index {_ranges(self.leading_indices)})"
        return text


@dataclass
class PinnedBoundReport:
    """Columns an unconverged interior-point solve left pressed against a bound
    that equality rows also determine (#2366). ``None`` on a result means the
    check did not run or found nothing; see :func:`pinned_bound_report`."""

    groups: list[PinnedGroup]
    n_columns: int
    rtol: float
    bound_relax_factor: Optional[float] = None
    constr_viol_tol: Optional[float] = None

    @property
    def effective_relaxation(self) -> Optional[float]:
        """The bound relaxation at a bound of magnitude <= 1, as POUNCE/Ipopt apply
        it: ``min(bound_relax_factor * max(1, |b|), constr_viol_tol)``."""
        if self.bound_relax_factor is None:
            return None
        relax = float(self.bound_relax_factor)
        if self.constr_viol_tol is not None:
            relax = min(relax, float(self.constr_viol_tol))
        return relax

    def message(
        self, status: str, solver: str = "solver='pounce'", options_kw: str = "pounce_options"
    ) -> str:
        """The warning text. ``solver`` names the solve that stopped and
        ``options_kw`` the ``Model.solve`` keyword its NLP options go through."""
        shown = "; ".join(g.describe() for g in self.groups[:8])
        if len(self.groups) > 8:
            shown += f"; and {len(self.groups) - 8} more groups"
        text = (
            f"{solver} stopped with status {status!r}, and {self.n_columns} "
            f"columns of its final iterate are within {self.rtol:g}*max(1,|bound|) of a "
            f"bound while equality rows also involve them: {shown}. If the equalities "
            "determine those columns exactly at the bound (for example a state that "
            "starts at lb=0 and stays there until an input switches on), the "
            "interior-point method has no strictly feasible region near the solution "
            "and cannot converge (#2366). For a DAEBuilder/MOLBuilder state the first "
            "index is the finite element."
        )
        relax = self.effective_relaxation
        if relax is not None:
            text += f" Bound relaxation in effect: {relax:g}"
            if (
                self.bound_relax_factor
                and self.constr_viol_tol is not None
                and self.constr_viol_tol < self.bound_relax_factor
            ):
                text += (
                    f" (bound_relax_factor={self.bound_relax_factor:g} is capped at "
                    f"constr_viol_tol={self.constr_viol_tol:g})"
                )
            text += "."
        text += (
            " Remedies: declare the bound slightly outside the values the model reaches "
            "(for a DAE state, e.g. lb=-1e-8 on the state's scale instead of 0), or pass "
            f"{options_kw}={{'bound_relax_factor': 1e-8, 'constr_viol_tol': 1e-8}}. A "
            "collocation mesh whose stage values undershoot the bound makes even the "
            "first remedy infeasible; refine the mesh where the input switches on. "
            "result.pinned_bounds holds this report."
        )
        return text


def _ranges(values: list[int]) -> str:
    """``[0, 1, 2, 5, 7, 8]`` -> ``"0-2, 5, 7-8"``."""
    parts: list[str] = []
    start = prev = values[0]
    for v in values[1:]:
        if v == prev + 1:
            prev = v
            continue
        parts.append(f"{start}-{prev}" if prev > start else f"{start}")
        start = prev = v
    parts.append(f"{start}-{prev}" if prev > start else f"{start}")
    return ", ".join(parts)


def equality_columns(evaluator: Any, cl: np.ndarray, cu: np.ndarray) -> np.ndarray:
    """Boolean mask over the columns that appear in an equality row (``cl == cu``)."""
    n = int(evaluator.n_variables)
    cl = np.asarray(cl, dtype=np.float64).reshape(-1)
    cu = np.asarray(cu, dtype=np.float64).reshape(-1)
    eq_rows = np.isfinite(cl) & (cl == cu)
    mask = np.zeros(n, dtype=bool)
    if not eq_rows.any():
        return mask
    rows, cols = evaluator.jacobian_structure()
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    keep = eq_rows[rows]
    mask[cols[keep]] = True
    return mask


def pinned_bound_report(
    model: Any,
    x_flat: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    in_equality: np.ndarray,
    *,
    rtol: float = PINNED_RTOL,
    bound_relax_factor: Optional[float] = None,
    constr_viol_tol: Optional[float] = None,
) -> Optional[PinnedBoundReport]:
    """Group the columns of ``x_flat`` that sit at a finite bound and are in an
    equality row, by model variable and side. ``None`` when there are none.

    ``lb``/``ub`` are the bounds the solve ran on, over the evaluator's columns;
    columns past the model's own variables (solve-time auxiliaries) are grouped
    under ``"<auxiliary>"``.
    """
    x = np.asarray(x_flat, dtype=np.float64).reshape(-1)
    lb = np.asarray(lb, dtype=np.float64).reshape(-1)
    ub = np.asarray(ub, dtype=np.float64).reshape(-1)
    in_eq = np.asarray(in_equality, dtype=bool).reshape(-1)
    if not (x.size == lb.size == ub.size == in_eq.size):
        raise ValueError(
            f"pinned_bound_report: sizes differ (x {x.size}, lb {lb.size}, ub {ub.size}, "
            f"equality mask {in_eq.size})"
        )
    # The 1e20 sentinel is "no bound" throughout discopt.
    free = ub > lb
    near_lo = free & (lb > -1e19) & (x - lb <= rtol * np.maximum(1.0, np.abs(lb))) & in_eq
    near_hi = free & (ub < 1e19) & (ub - x <= rtol * np.maximum(1.0, np.abs(ub))) & in_eq
    near_hi &= ~near_lo
    if not (near_lo.any() or near_hi.any()):
        return None

    groups: list[PinnedGroup] = []
    offset = 0
    spans: list[tuple[str, int, int, tuple[int, ...]]] = []
    for v in model._variables:
        size = int(v.size)
        spans.append((v.name, offset, offset + size, tuple(v.shape)))
        offset += size
    if offset < x.size:
        spans.append(("<auxiliary>", offset, x.size, (x.size - offset,)))

    for name, start, stop, shape in spans:
        for side, mask, bnd in (("lower", near_lo, lb), ("upper", near_hi, ub)):
            local = np.flatnonzero(mask[start:stop])
            if local.size == 0:
                continue
            values = np.unique(bnd[start:stop][local])
            leading: list[int] = []
            if shape and int(np.prod(shape)) == stop - start:
                leading = sorted({int(i) for i in np.unravel_index(local, shape)[0]})
            groups.append(
                PinnedGroup(
                    variable=name,
                    side=side,
                    count=int(local.size),
                    bounds=[float(b) for b in values[:8]],
                    leading_indices=leading,
                )
            )
    groups.sort(key=lambda g: -g.count)
    return PinnedBoundReport(
        groups=groups,
        n_columns=int(near_lo.sum() + near_hi.sum()),
        rtol=float(rtol),
        bound_relax_factor=bound_relax_factor,
        constr_viol_tol=constr_viol_tol,
    )

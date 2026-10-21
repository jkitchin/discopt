"""The master MILP of the native GDP algorithms (#1775 G3.2, spec §4.2).

The matrix MILP over the lowered (big-M) model's linear rows, the outer-
approximation cut pool and the no-good exclusions. The helper functions are LOA's
own master, moved here unchanged from ``solvers/gdpopt_loa.py`` (which imports
them back): :class:`MasterMILP` is the stateful wrapper the framework's
algorithms use.

Backend: ``get_milp_solver(backend=...)``. The default ``"auto"`` is the Rust
simplex (#356); ``"highs"`` stays an explicit opt-in, as the G3.2 entry
experiment on the ``gdp_parity`` panel's LOA masters confirmed (see
:data:`DEFAULT_MASTER_BACKEND`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

#: The master backend the framework's algorithms use unless told otherwise:
#: ``"auto"``, the Rust simplex (#356). Re-measured by the G3.2 entry experiment
#: (#1775): the 43 LOA masters of 21 ``gdp_parity`` instances, replayed on both
#: backends interleaved, 3 repeats each. On the 40 both solved to optimality the
#: objectives agree and the simplex is faster by SGM 2.8x (ratio simplex/HiGHS
#: 0.36), mostly small masters where HiGHS pays a ~10-20 ms floor. HiGHS wins the
#: large ones (modprodnet 12x, med_term_purchasing 2.1 s vs a 60 s simplex limit)
#: and so the total (81 s vs 114 s). The bar set before the run -- HiGHS at least
#: 1.5x faster by SGM, faster in total, no disagreement -- is not met, so #356
#: stands; ``backend="highs"`` stays a per-master opt-in. A size-gated default is
#: a new hypothesis, not measured here.
DEFAULT_MASTER_BACKEND = "auto"


def _int_config_key(x_master, int_indices) -> tuple[int, ...]:
    """Canonical hashable key for the integer part of a master solution."""
    return tuple(int(round(x_master[idx])) for idx in int_indices)


def _add_no_good_cut(x_master, int_indices, oa_A_rows, oa_b_rows, n_vars):
    """Add a binary integer-exclusion (no-good) cut.

    Valid ONLY when every discrete variable is 0/1: the single inequality below
    excludes exactly the incumbent 0/1 configuration and keeps all other 0/1
    configurations feasible. For general integer variables this form is UNSOUND
    (it also cuts feasible non-0/1 configurations — issue #756); those callers
    must use :func:`_build_no_good_augmentation` instead. Callers gate on
    :func:`_discrete_vars_all_binary`.
    """
    # sum_{i: y_i*=1} (1 - y_i) + sum_{i: y_i*=0} y_i >= 1
    # Equivalently: -sum_{y_i*=1} y_i + sum_{y_i*=0} y_i >= 1 - count(y_i*=1)
    # As <= form: sum_{y_i*=1} y_i - sum_{y_i*=0} y_i <= count(y_i*=1) - 1
    coeffs = np.zeros(n_vars)
    count_ones = 0
    for idx in int_indices:
        val = round(x_master[idx])
        if val >= 0.5:
            coeffs[idx] = 1.0
            count_ones += 1
        else:
            coeffs[idx] = -1.0
    oa_A_rows.append(coeffs)
    oa_b_rows.append(float(count_ones - 1))


def _discrete_vars_all_binary(int_indices, lb, ub, tol: float = 1e-9) -> bool:
    """True iff every discrete variable is a 0/1 variable (bounds ``[0, 1]``).

    The binary-only no-good cut (:func:`_add_no_good_cut`) is valid only in this
    case; otherwise the exact single-point exclusion is required (#756).
    """
    return all(abs(float(lb[i])) <= tol and abs(float(ub[i]) - 1.0) <= tol for i in int_indices)


def _build_no_good_augmentation(no_good_configs, int_indices, lb, ub, n_base):
    """Exact single-point integer no-good exclusions via auxiliary binaries (#756).

    Each recorded integer configuration ``v*`` is excluded with the disjunction
    "at least one discrete variable moves strictly away from its value at ``v*``".
    For each variable ``z_i`` with value ``v`` and bounds ``[l, u]`` we add, when a
    move is possible, a binary indicator forcing a real step:

    * down (``v > l``): ``z_i + (u - v + 1)·gamma <= u`` — ``gamma = 1`` forces
      ``z_i <= v - 1``;
    * up (``v < u``): ``-z_i + (v + 1 - l)·delta <= -l`` — ``delta = 1`` forces
      ``z_i >= v + 1``;

    plus an aggregate ``sum(indicators) >= 1`` (as ``-sum <= -1``). This removes
    EXACTLY the point ``v*`` — every other integer combination can activate the
    indicator for a coordinate in which it differs — so the exclusion is sound for
    general integers, not only 0/1. If a configuration is pinned at its bounds in
    every coordinate (no move possible), the empty aggregate row ``0 <= -1`` makes
    the master infeasible, which is correct: ``v*`` was then the only integer point.

    Returns ``(rows, rhs, n_aux)`` with each row already width ``n_base + n_aux``.
    """
    pending: list[tuple[dict[int, float], dict[int, float], float]] = []
    n_aux = 0
    for config in no_good_configs:
        indicator_cols: list[int] = []
        for idx, v in zip(int_indices, config):
            lo = float(lb[idx])
            hi = float(ub[idx])
            vv = float(v)
            if vv > lo + 0.5:  # a strictly-lower integer exists
                col = n_aux
                n_aux += 1
                pending.append(({idx: 1.0}, {col: (hi - vv + 1.0)}, hi))
                indicator_cols.append(col)
            if vv < hi - 0.5:  # a strictly-higher integer exists
                col = n_aux
                n_aux += 1
                pending.append(({idx: -1.0}, {col: (vv + 1.0 - lo)}, -lo))
                indicator_cols.append(col)
        pending.append(({}, {c: -1.0 for c in indicator_cols}, -1.0))

    n_total = n_base + n_aux
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for var_coeffs, aux_coeffs, r in pending:
        row = np.zeros(n_total)
        for idx, c in var_coeffs.items():
            row[idx] = c
        for col, c in aux_coeffs.items():
            row[n_base + col] = c
        rows.append(row)
        rhs.append(float(r))
    return rows, rhs, n_aux


def _solve_master_milp(
    linear_A_rows,
    linear_b_rows,
    linear_senses,
    oa_A_rows,
    oa_b_rows,
    n_vars,
    integrality,
    lb,
    ub,
    obj_coeffs,
    obj_is_linear,
    objective_bound_valid,
    time_limit,
    gap_tolerance,
    milp_solver="auto",
    no_good_configs=None,
    int_indices=None,
):
    """Build and solve the master MILP."""
    try:
        from discopt.solvers.lp_backend import get_milp_solver

        solve_milp = get_milp_solver(backend=milp_solver)
    except ImportError as e:
        raise ImportError(
            # #356 took HiGHS off the MILP path; #1060 put it back as the
            # explicit `milp_solver="highs"` master engine only, so highspy is a
            # live hint again -- but only for a caller who asks for it by name,
            # never for the default. The simplex backend is in-tree and needs no
            # install, which is why it is not listed here.
            "LOA solver requires a MILP backend for the master. Install one of: "
            "pip install pounce-solver  |  pip install gurobipy  |  "
            "pip install highspy (with milp_solver='highs')"
        ) from e

    # Determine master problem dimensions
    use_objective_epigraph = (not obj_is_linear) and objective_bound_valid
    n_master = n_vars
    if use_objective_epigraph:
        n_master += 1  # epigraph variable eta

    # Build A_ub, b_ub from linear <= constraints + OA cuts
    A_ub_rows = []
    b_ub_vals = []

    for i, sense in enumerate(linear_senses):
        row = linear_A_rows[i]
        rhs = linear_b_rows[i]
        if use_objective_epigraph:
            row = np.append(row, 0.0)
        if sense == "<=":
            A_ub_rows.append(row)
            b_ub_vals.append(rhs)
        elif sense == ">=":
            A_ub_rows.append(-row)
            b_ub_vals.append(-rhs)

    # OA cuts (all <= form)
    for i in range(len(oa_A_rows)):
        row = oa_A_rows[i]
        if use_objective_epigraph and len(row) == n_vars:
            row = np.append(row, 0.0)
        A_ub_rows.append(row)
        b_ub_vals.append(oa_b_rows[i])

    # Equality constraints from linear
    A_eq_rows = []
    b_eq_vals = []
    for i, sense in enumerate(linear_senses):
        if sense == "==":
            row = linear_A_rows[i]
            if use_objective_epigraph:
                row = np.append(row, 0.0)
            A_eq_rows.append(row)
            b_eq_vals.append(linear_b_rows[i])

    # #756: exact single-point integer no-good exclusions add auxiliary binary
    # columns to the master. Pad the existing rows with zeros for those columns and
    # append the exclusion rows (all <= form).
    n_extra = 0
    ng_rows: list[np.ndarray] = []
    ng_rhs: list[float] = []
    if no_good_configs:
        if int_indices is None:
            int_indices = [i for i in range(n_vars) if integrality[i]]
        ng_rows, ng_rhs, n_extra = _build_no_good_augmentation(
            no_good_configs, int_indices, lb, ub, n_master
        )
    if n_extra > 0:
        A_ub_rows = [np.concatenate([r, np.zeros(n_extra)]) for r in A_ub_rows]
        A_eq_rows = [np.concatenate([r, np.zeros(n_extra)]) for r in A_eq_rows]
        A_ub_rows.extend(ng_rows)
        b_ub_vals = list(b_ub_vals) + list(ng_rhs)

    # Build arrays
    A_ub = np.array(A_ub_rows) if A_ub_rows else None
    b_ub = np.array(b_ub_vals) if b_ub_vals else None
    A_eq = np.array(A_eq_rows) if A_eq_rows else None
    b_eq = np.array(b_eq_vals) if b_eq_vals else None

    # Objective
    if obj_is_linear:
        # The constant is not representable in the master; the caller adds it back
        # to the master's bound (``master_obj_offset``).
        c_vec, _off = obj_coeffs
        c = c_vec.copy()
    elif use_objective_epigraph:
        # min eta, with OA cuts: eta >= f(x*) + grad(x*) . (x - x*)
        c = np.zeros(n_master)
        c[-1] = 1.0  # minimize eta
    else:
        c = np.zeros(n_master)

    # Bounds
    bounds_list = list(zip(lb.tolist(), ub.tolist()))
    if use_objective_epigraph:
        bounds_list.append((-1e20, 1e20))  # eta unbounded

    # Integrality
    int_vec = np.zeros(n_master, dtype=np.int32)
    int_vec[:n_vars] = integrality

    # #756: extend objective / bounds / integrality for the no-good aux binaries.
    if n_extra > 0:
        c = np.concatenate([c, np.zeros(n_extra)])
        bounds_list = bounds_list + [(0.0, 1.0)] * n_extra
        int_vec = np.concatenate([int_vec, np.ones(n_extra, dtype=np.int32)])

    return solve_milp(
        c=c,
        A_ub=A_ub,
        b_ub=b_ub,
        A_eq=A_eq,
        b_eq=b_eq,
        bounds=bounds_list,
        integrality=int_vec,
        time_limit=time_limit,
        gap_tolerance=gap_tolerance,
    )


# ── the framework's master ───────────────────────────────────────────────────


@dataclass
class MasterResult:
    """One master solve. ``bound`` is a valid lower bound on the GDP (minimization
    convention, objective constant included) or ``None``: a master whose
    objective is neither linear nor certified convex bounds nothing.
    ``proven_infeasible`` is set only on the backend's rigorous INFEASIBLE verdict;
    a limit or an error leaves ``x`` ``None`` and proves nothing."""

    x: Optional[np.ndarray]
    config: Optional[tuple[int, ...]]
    bound: Optional[float]
    proven_infeasible: bool
    status: str


class MasterMILP:
    """The master MILP: linear rows, an OA cut pool and no-good exclusions.

    All rows live in the minimization convention of the lowered model's columns
    ``0..n_vars-1``. ``objective`` is ``(c, offset)`` when the objective is
    linear; otherwise the master minimizes an epigraph column ``n_vars`` when
    ``objective_convex`` (its objective cuts are then valid) and is a pure
    feasibility problem otherwise (``bound`` is ``None``).
    """

    def __init__(
        self,
        n_vars: int,
        lb: Any,
        ub: Any,
        integrality: Any,
        *,
        objective: Optional[tuple[np.ndarray, float]] = None,
        objective_convex: bool = False,
        backend: str = DEFAULT_MASTER_BACKEND,
    ) -> None:
        self.n_vars = int(n_vars)
        self.lb = np.asarray(lb, dtype=np.float64)
        self.ub = np.asarray(ub, dtype=np.float64)
        self.integrality = np.asarray(integrality, dtype=np.int32)
        if not (self.lb.shape == self.ub.shape == self.integrality.shape == (self.n_vars,)):
            raise ValueError("MasterMILP: lb, ub and integrality must each have n_vars entries")
        self.int_indices = [int(i) for i in np.flatnonzero(self.integrality)]
        self.objective = objective
        self.objective_convex = bool(objective_convex)
        self.backend = backend
        self.rows: list[np.ndarray] = []
        self.rhs: list[float] = []
        self.senses: list[str] = []
        self.cuts: list[np.ndarray] = []
        self.cut_rhs: list[float] = []
        self.no_good_configs: list[tuple[int, ...]] = []
        self.all_binary = _discrete_vars_all_binary(self.int_indices, self.lb, self.ub)
        self.n_solves = 0

    @property
    def bound_valid(self) -> bool:
        """Whether the master's optimum is a lower bound on the GDP."""
        return self.objective is not None or self.objective_convex

    @property
    def offset(self) -> float:
        return 0.0 if self.objective is None else float(self.objective[1])

    def add_row(self, coeffs: Any, sense: str, rhs: float) -> None:
        """A linear row of the problem itself (``coeffs @ x  sense  rhs``)."""
        if sense not in ("<=", ">=", "=="):
            raise ValueError(f"MasterMILP.add_row: unknown sense {sense!r}")
        row = np.asarray(coeffs, dtype=np.float64)
        if row.shape != (self.n_vars,):
            raise ValueError(f"MasterMILP.add_row: expected {self.n_vars} coefficients")
        self.rows.append(row)
        self.senses.append(sense)
        self.rhs.append(float(rhs))

    def add_cut(self, coeffs: Any, rhs: float) -> None:
        """A valid inequality ``coeffs @ x <= rhs``, over ``x`` or ``(x, eta)``."""
        row = np.asarray(coeffs, dtype=np.float64)
        if row.shape not in ((self.n_vars,), (self.n_vars + 1,)):
            raise ValueError(f"MasterMILP.add_cut: expected {self.n_vars} or {self.n_vars + 1}")
        if row.shape == (self.n_vars + 1,) and not self.objective_convex:
            raise ValueError("MasterMILP.add_cut: an epigraph cut needs objective_convex")
        if not (np.all(np.isfinite(row)) and math.isfinite(rhs)):
            raise ValueError("MasterMILP.add_cut: non-finite cut")
        self.cuts.append(row.copy())
        self.cut_rhs.append(float(rhs))

    def exclude(self, config: tuple[int, ...]) -> None:
        """Exclude exactly the integer configuration ``config``.

        Only for a configuration proved infeasible or dominated: the caller owns
        that proof (C-35). The binary form when every discrete column is 0/1,
        the exact auxiliary-binary form otherwise (#756).
        """
        config = tuple(int(v) for v in config)
        if len(config) != len(self.int_indices):
            raise ValueError("MasterMILP.exclude: config does not match the discrete columns")
        if self.all_binary:
            x = np.zeros(self.n_vars)
            for idx, v in zip(self.int_indices, config):
                x[idx] = v
            _add_no_good_cut(x, self.int_indices, self.cuts, self.cut_rhs, self.n_vars)
        else:
            self.no_good_configs.append(config)

    def config_of(self, x: np.ndarray) -> tuple[int, ...]:
        return _int_config_key(x, self.int_indices)

    def solve(
        self,
        time_limit: float,
        gap_tolerance: float = 1e-4,
        incumbent: Optional[float] = None,
    ) -> MasterResult:
        """Solve within ``time_limit`` (the caller's deadline remainder).

        ``incumbent`` (minimization convention, constant included) loosens the
        master's own gap to the outer one, as LOA does (``master_gap_tolerance``).
        """
        from discopt.solvers import SolveStatus
        from discopt.solvers._gap import master_gap_tolerance

        self.n_solves += 1
        lin = self.objective is not None
        obj = None if self.objective is None else (self.objective[0], self.objective[1])
        r = _solve_master_milp(
            self.rows,
            self.rhs,
            self.senses,
            self.cuts,
            self.cut_rhs,
            self.n_vars,
            self.integrality,
            self.lb,
            self.ub,
            obj,
            lin,
            self.objective_convex,
            time_limit=max(0.0, float(time_limit)),
            gap_tolerance=master_gap_tolerance(
                gap_tolerance, None if incumbent is None else incumbent - self.offset
            ),
            milp_solver=self.backend,
            no_good_configs=self.no_good_configs,
            int_indices=self.int_indices,
        )
        status = "error" if r is None else str(getattr(r.status, "value", r.status))
        if r is None or r.x is None:
            infeasible = r is not None and r.status == SolveStatus.INFEASIBLE
            return MasterResult(None, None, None, infeasible, status)
        x = np.asarray(r.x, dtype=np.float64)[: self.n_vars]
        bound = None
        if self.bound_valid and r.bound is not None and math.isfinite(float(r.bound)):
            bound = float(r.bound) + self.offset
        return MasterResult(x, self.config_of(x), bound, False, status)

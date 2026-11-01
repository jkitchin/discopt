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
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

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


#: ``id(row) -> (row, nonzero columns, their values, row length)``: a master
#: row's sparse form, read once (see :func:`_row_nonzeros`).
_NzCache = dict[int, tuple[Any, np.ndarray, np.ndarray, int]]


class _SparseMasterRow:
    """A master row held by its nonzeros: sorted ``idx``, their ``val``, width ``n``.

    The dense row it stands for is ``zeros(n)`` with ``val`` at ``idx``; reading
    it through ``np.asarray`` gives exactly that. :func:`build_master` stores the
    lowered model's rows this way (#2079): dense, a 7909 x 57828 master held
    3.7 GB of zeros and took most of its setup to write them.
    """

    __slots__ = ("idx", "val", "n")

    def __init__(self, idx: np.ndarray, val: np.ndarray, n: int) -> None:
        self.idx = np.asarray(idx, dtype=np.int64)
        self.val = np.asarray(val, dtype=np.float64)
        self.n = int(n)

    @property
    def shape(self) -> tuple[int]:
        return (self.n,)

    def __len__(self) -> int:
        return self.n

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        out = np.zeros(self.n, dtype=np.float64)
        out[self.idx] = self.val
        return out if dtype is None else out.astype(dtype)


def _row_nonzeros(row: Any, cache: Optional[_NzCache]) -> tuple[np.ndarray, np.ndarray, int]:
    """``(columns, values, length)`` of the nonzeros of a dense master row.

    With ``cache``, a row already read is not scanned again (#2079: the master
    is rebuilt on every iteration, and rescanning every full-width row of a
    7909 x 57828 master each time was most of its build). The entry keeps the row
    object itself and is used only for that same object, so a recycled ``id``
    cannot alias another row; the master never changes a row once it holds it.
    """
    if isinstance(row, _SparseMasterRow):
        return row.idx, row.val, row.n
    if cache is not None:
        hit = cache.get(id(row))
        if hit is not None and hit[0] is row:
            return hit[1], hit[2], hit[3]
    arr = np.asarray(row, dtype=np.float64).ravel()
    nz = np.flatnonzero(arr)
    vals = arr[nz]
    if cache is not None and isinstance(row, np.ndarray):
        cache[id(row)] = (row, nz, vals, int(arr.shape[0]))
    return nz, vals, int(arr.shape[0])


def _stack_sparse(
    rows: list[np.ndarray],
    width: int,
    signs: Optional[list[float]] = None,
    cache: Optional[_NzCache] = None,
) -> sp.csr_matrix:
    """``rows`` (dense, each at most ``width`` long) as a ``len(rows) x width`` CSR
    matrix, row ``k`` scaled by ``signs[k]``; a short row is zero-padded on the right.

    Exactly the nonzeros of the dense stack, in row order and sorted by column
    (#2079): building the dense stack first cost 39 s on a 7909 x 57828 master.
    ``cache`` (:func:`_row_nonzeros`) skips rescanning rows read on an earlier
    build; the matrix is the same either way.
    """
    indptr = np.zeros(len(rows) + 1, dtype=np.int64)
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    for k, row in enumerate(rows):
        nz, row_vals, length = _row_nonzeros(row, cache)
        if length > width:
            raise ValueError(f"master row {k} has {length} entries; the master has {width}")
        cols.append(nz)
        vals.append(row_vals if signs is None or signs[k] == 1.0 else signs[k] * row_vals)
        indptr[k + 1] = indptr[k] + nz.size
    return sp.csr_matrix(
        (
            np.concatenate(vals) if vals else np.zeros(0),
            np.concatenate(cols) if cols else np.zeros(0, dtype=np.int64),
            indptr,
        ),
        shape=(len(rows), width),
    )


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
    nz_cache: Optional[_NzCache] = None,
):
    """Build and solve the master MILP within ``time_limit`` seconds of this call.

    The build is charged to ``time_limit`` (#2079): the MILP gets what is left.
    ``nz_cache`` is the caller's :func:`_row_nonzeros` cache, kept across the
    solves of one master so its rows are scanned once.
    """
    t_enter = time.perf_counter()
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

    # #2079: the rows are stacked SPARSE. A dense ``np.array`` of them is
    # ``rows x columns`` floats -- 7909 x 57828 (3.7 GB) on GDPlib
    # ``stranded_gas``, 39 s to build per master solve, all of it after the
    # caller's time limit was read. Every backend takes a scipy matrix, and each
    # converted the dense one to sparse anyway (dropping the same zeros), so the
    # rows the engine sees are unchanged. Padding -- the epigraph column, the
    # no-good auxiliaries -- is implicit in the width.
    ub_rows: list[np.ndarray] = []
    ub_sign: list[float] = []
    b_ub_vals: list[float] = []
    for i, sense in enumerate(linear_senses):
        if sense == "<=":
            ub_rows.append(linear_A_rows[i])
            ub_sign.append(1.0)
            b_ub_vals.append(linear_b_rows[i])
        elif sense == ">=":
            ub_rows.append(linear_A_rows[i])
            ub_sign.append(-1.0)
            b_ub_vals.append(-linear_b_rows[i])

    # OA cuts (all <= form)
    for i in range(len(oa_A_rows)):
        ub_rows.append(oa_A_rows[i])
        ub_sign.append(1.0)
        b_ub_vals.append(oa_b_rows[i])

    # Equality constraints from linear
    eq_rows = [linear_A_rows[i] for i, sense in enumerate(linear_senses) if sense == "=="]
    b_eq_vals = [linear_b_rows[i] for i, sense in enumerate(linear_senses) if sense == "=="]

    # #756: exact single-point integer no-good exclusions add auxiliary binary
    # columns to the master, after every existing column, and their rows (all
    # <= form).
    n_extra = 0
    if no_good_configs:
        if int_indices is None:
            int_indices = [i for i in range(n_vars) if integrality[i]]
        ng_rows, ng_rhs, n_extra = _build_no_good_augmentation(
            no_good_configs, int_indices, lb, ub, n_master
        )
        ub_rows.extend(ng_rows)
        ub_sign.extend([1.0] * len(ng_rows))
        b_ub_vals = list(b_ub_vals) + list(ng_rhs)

    width = n_master + n_extra
    A_ub = _stack_sparse(ub_rows, width, ub_sign, nz_cache) if ub_rows else None
    b_ub = np.array(b_ub_vals, dtype=np.float64) if b_ub_vals else None
    A_eq = _stack_sparse(eq_rows, width, cache=nz_cache) if eq_rows else None
    b_eq = np.array(b_eq_vals, dtype=np.float64) if b_eq_vals else None

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
        time_limit=(
            None
            if time_limit is None
            else max(0.0, float(time_limit) - (time.perf_counter() - t_enter))
        ),
        gap_tolerance=gap_tolerance,
    )


# ── the framework's master ───────────────────────────────────────────────────


@dataclass
class MasterResult:
    """One master solve. ``bound`` is a lower bound (minimization convention,
    objective constant included) or ``None``: a master whose objective is neither
    linear nor certified convex bounds nothing. ``proven_infeasible`` is set only
    on the backend's rigorous INFEASIBLE verdict; a limit or an error leaves ``x``
    ``None`` and proves nothing.

    Both speak only for the configurations the master still admits. Once a
    caller has excluded a configuration that was *dominated* (solved, not proved
    infeasible), the valid lower bound on the GDP is ``min(incumbent, bound)``,
    and ``proven_infeasible`` means "no better configuration", not "the GDP is
    infeasible", unless every excluded configuration was itself infeasible."""

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
        if objective is not None and objective_convex:
            raise ValueError(
                "MasterMILP: objective_convex selects the epigraph form for a nonlinear "
                "objective; a linear objective (c, offset) needs none"
            )
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
        #: Each row's and cut's nonzeros, read once over every solve (#2079).
        self._nz_cache: _NzCache = {}

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
        # A copy, as ``add_cut`` takes: the master caches each row's nonzeros
        # (#2079), so a row must not change after it is added.
        self.rows.append(row.copy())
        self.senses.append(sense)
        self.rhs.append(float(rhs))

    def _add_sparse_row(self, idx: Any, val: Any, sense: str, rhs: float) -> None:
        """:meth:`add_row` of the row with ``val`` at the sorted columns ``idx``
        and zeros elsewhere, without writing it out at the master's width (#2079).

        ``idx``/``val`` must already be what ``flatnonzero`` of the dense row
        gives: sorted, unique, no zero values.
        """
        if sense not in ("<=", ">=", "=="):
            raise ValueError(f"MasterMILP.add_row: unknown sense {sense!r}")
        idx = np.asarray(idx, dtype=np.int64)
        val = np.asarray(val, dtype=np.float64)
        if idx.shape != val.shape or (
            idx.size and (idx[0] < 0 or idx[-1] >= self.n_vars or np.any(np.diff(idx) <= 0))
        ):
            raise ValueError("MasterMILP: a sparse row needs sorted, unique, in-range columns")
        if np.any(val == 0.0):
            raise ValueError("MasterMILP: a sparse row stores no zero coefficient")
        self.rows.append(_SparseMasterRow(idx, val, self.n_vars))
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
        if not self.all_binary:
            self._check_exact_no_good_bounds()
        if self.all_binary:
            x = np.zeros(self.n_vars)
            for idx, v in zip(self.int_indices, config):
                x[idx] = v
            _add_no_good_cut(x, self.int_indices, self.cuts, self.cut_rhs, self.n_vars)
        else:
            self.no_good_configs.append(config)

    def _check_exact_no_good_bounds(self) -> None:
        # The exact exclusion's big-M coefficients are u - v + 1 and v + 1 - l
        # (#756): with an infinite or default (1e20) bound they mean nothing.
        for idx in self.int_indices:
            lo, hi = float(self.lb[idx]), float(self.ub[idx])
            if not (math.isfinite(lo) and math.isfinite(hi)) or max(abs(lo), abs(hi)) >= 1e19:
                raise ValueError(
                    f"MasterMILP.exclude: discrete column {idx} has bounds [{lo:g}, {hi:g}]; "
                    "the exact no-good needs finite bounds on every general integer"
                )

    def config_of(self, x: np.ndarray) -> tuple[int, ...]:
        return _int_config_key(x, self.int_indices)

    def solve(
        self,
        time_limit: float,
        gap_tolerance: float = 1e-4,
        incumbent: Optional[float] = None,
        *,
        objective_override: Optional[Any] = None,
        exclude_also: Any = (),
        extra_cuts: Any = (),
    ) -> MasterResult:
        """Solve within ``time_limit`` (the caller's deadline remainder).

        ``incumbent`` (minimization convention, constant included) loosens the
        master's own gap to the outer one, as LOA does (``master_gap_tolerance``).

        ``objective_override`` (a coefficient vector over ``x``), with
        ``exclude_also`` (configurations) and ``extra_cuts`` (``(row, rhs)``
        pairs, ``row @ x <= rhs``), serves the initialization strategies (G3.3):
        the same rows, cuts and no-goods with another linear objective and extra
        exclusions for this call only. The epigraph cuts are left out (a
        relaxation of a feasibility question), no row, cut or no-good is
        recorded on the master (only ``n_solves`` counts the call), and the
        result carries no ``bound``.
        """
        from discopt.solvers import SolveStatus
        from discopt.solvers._gap import master_gap_tolerance

        self.n_solves += 1
        if incumbent is not None and not math.isfinite(incumbent):
            incumbent = None  # no incumbent yet: the master keeps its own gap
        override = objective_override is not None
        cuts, cut_rhs, no_goods = self.cuts, self.cut_rhs, self.no_good_configs
        obj: Optional[tuple[np.ndarray, float]]
        if override:
            c = np.asarray(objective_override, dtype=np.float64)
            if c.shape != (self.n_vars,) or not np.all(np.isfinite(c)):
                raise ValueError(f"objective_override must be {self.n_vars} finite numbers")
            obj, lin, convex = (c, 0.0), True, False
            keep = [i for i, row in enumerate(self.cuts) if row.shape == (self.n_vars,)]
            cuts = [self.cuts[i] for i in keep]
            cut_rhs = [self.cut_rhs[i] for i in keep]
            for row, rhs in extra_cuts:
                row = np.asarray(row, dtype=np.float64)
                if row.shape != (self.n_vars,) or not np.all(np.isfinite(row)):
                    raise ValueError(f"extra_cuts: each row must be {self.n_vars} finite numbers")
                cuts.append(row)
                cut_rhs.append(float(rhs))
            no_goods = list(self.no_good_configs)
            for config in exclude_also:
                config = tuple(int(v) for v in config)
                if len(config) != len(self.int_indices):
                    raise ValueError("exclude_also: config does not match the discrete columns")
                if self.all_binary:
                    x0 = np.zeros(self.n_vars)
                    for idx, v in zip(self.int_indices, config):
                        x0[idx] = v
                    _add_no_good_cut(x0, self.int_indices, cuts, cut_rhs, self.n_vars)
                else:
                    self._check_exact_no_good_bounds()
                    no_goods.append(config)
        else:
            if exclude_also or extra_cuts:
                raise ValueError(
                    "exclude_also and extra_cuts are only for an objective_override solve"
                )
            obj = None if self.objective is None else (self.objective[0], self.objective[1])
            lin, convex = self.objective is not None, self.objective_convex
        r = _solve_master_milp(
            self.rows,
            self.rhs,
            self.senses,
            cuts,
            cut_rhs,
            self.n_vars,
            self.integrality,
            self.lb,
            self.ub,
            obj,
            lin,
            convex,
            time_limit=max(0.0, float(time_limit)),
            gap_tolerance=master_gap_tolerance(
                gap_tolerance,
                None if incumbent is None or override else incumbent - self.offset,
            ),
            milp_solver=self.backend,
            no_good_configs=no_goods,
            int_indices=self.int_indices,
            nz_cache=self._nz_cache,
        )
        if override and extra_cuts:
            # This call's extra cuts are not the master's: do not keep them alive.
            held = {id(a) for a in self.rows} | {id(a) for a in self.cuts}
            for key in [k for k in self._nz_cache if k not in held]:
                del self._nz_cache[key]
        status = "error" if r is None else str(getattr(r.status, "value", r.status))
        if r is None or r.x is None:
            infeasible = r is not None and r.status == SolveStatus.INFEASIBLE
            return MasterResult(None, None, None, infeasible, status)
        x = np.asarray(r.x, dtype=np.float64)[: self.n_vars]
        if override:
            return MasterResult(x, self.config_of(x), None, False, status)
        bound = None
        if self.bound_valid and r.bound is not None and math.isfinite(float(r.bound)):
            bound = float(r.bound) + self.offset
        return MasterResult(x, self.config_of(x), bound, False, status)


class RelaxationMaster(MasterMILP):
    """A :class:`MasterMILP` over a MILP relaxation's columns (#1775 G4.3, GLOA).

    The relaxation (:func:`discopt._relax.milp_relaxation.build_milp_relaxation`
    of the lowered model) keeps the model's ``n_model`` columns first, with their
    integrality, and appends continuous auxiliaries for the relaxed nonlinear
    terms. Its rows are a valid outer approximation of *every* row -- a
    nonconvex one included -- so the master's dual bound is a valid LB with no
    convexity certificate; LOA's OA cuts (certified-convex rows only) still apply
    and are padded onto the auxiliary columns. A cut over the model's columns
    goes through :meth:`add_model_cut` (the caller states its space); rows,
    cuts, objective overrides and extra cuts otherwise span the relaxation's
    ``n_vars`` columns. A solve returns ``x`` over the model's columns only.
    """

    def __init__(
        self,
        relaxation: Any,
        n_model: int,
        *,
        objective_fallback: Optional[tuple[np.ndarray, float]] = None,
        backend: str = DEFAULT_MASTER_BACKEND,
    ) -> None:
        import scipy.sparse as sp

        bounds = np.asarray(relaxation._bounds, dtype=np.float64).reshape(-1, 2)
        n_cols = bounds.shape[0]
        n_model = int(n_model)
        if n_cols < n_model:
            raise ValueError(
                f"RelaxationMaster: the relaxation has {n_cols} columns, fewer than the "
                f"model's {n_model}"
            )
        integ = relaxation._integrality
        integ = (
            np.zeros(n_cols, dtype=np.int32)
            if integ is None
            else np.asarray(integ, dtype=np.int32).reshape(-1)
        )
        self.n_model = n_model
        self.cleaned_coefficients = 0
        objective: Optional[tuple[np.ndarray, float]] = None
        #: Whether the objective is the relaxation's (a linear form over its
        #: auxiliaries standing for the model's objective, minimization convention).
        self.objective_from_relaxation = bool(relaxation._objective_bound_valid)
        if relaxation._objective_bound_valid:
            objective = (
                np.asarray(relaxation._c, dtype=np.float64).reshape(-1),
                float(relaxation._obj_offset),
            )
        elif objective_fallback is not None:
            c, off = objective_fallback
            c = np.asarray(c, dtype=np.float64)
            objective = (np.concatenate([c, np.zeros(n_cols - n_model)]), float(off))
        super().__init__(
            n_cols, bounds[:, 0], bounds[:, 1], integ, objective=objective, backend=backend
        )
        aux_int = [i for i in self.int_indices if i >= n_model]
        if aux_int:
            # A configuration is the model's integer columns; an integer
            # auxiliary would enter every no-good and the selection key.
            raise ValueError(
                f"RelaxationMaster: the relaxation adds {len(aux_int)} integer column(s); "
                "a configuration must be the model's own integer columns"
            )
        A = relaxation._A_ub
        if A is not None:
            b = np.asarray(relaxation._b_ub, dtype=np.float64).reshape(-1)
            if A.shape != (b.shape[0], n_cols):
                raise ValueError("RelaxationMaster: the relaxation's rows do not match its columns")
            if sp.issparse(A):
                # #2079: row by row from CSR, never at full width. Canonicalised
                # first (duplicates summed, columns sorted, explicit zeros dropped),
                # so each row is exactly ``flatnonzero`` of its dense form.
                csr = sp.csr_matrix(A, dtype=np.float64, copy=True)
                csr.sum_duplicates()
                csr.sort_indices()
                csr.eliminate_zeros()
                for i, rhs in enumerate(b):
                    lo, hi = csr.indptr[i], csr.indptr[i + 1]
                    self._add_sparse_row(csr.indices[lo:hi], csr.data[lo:hi], "<=", float(rhs))
            else:
                for row, rhs in zip(np.asarray(A, dtype=np.float64), b):
                    self.add_row(row, "<=", float(rhs))
        self.truncated = bool(getattr(relaxation, "_build_truncated", False))

    def model_row(self, coeffs: Any, *, epigraph: bool = False) -> np.ndarray:
        """A row over the model's columns (``x``, or ``(x, eta)`` when
        ``epigraph``) written over the relaxation's: zeros on the auxiliaries,
        ``eta`` last. The caller states the row's space; a length alone cannot,
        since with one auxiliary a relaxation row and an ``(x, eta)`` row have
        the same length (review of #1880)."""
        row = np.asarray(coeffs, dtype=np.float64).reshape(-1)
        want = self.n_model + (1 if epigraph else 0)
        if row.shape != (want,):
            raise ValueError(
                f"RelaxationMaster.model_row: expected {want} coefficients, got {row.shape[0]}"
            )
        aux = np.zeros(self.n_vars - self.n_model)
        return np.concatenate([row[: self.n_model], aux, row[self.n_model :]])

    def add_model_cut(self, coeffs: Any, rhs: float, *, epigraph: bool = False) -> None:
        """:meth:`add_cut` for a cut over the model's columns (see :meth:`model_row`)."""
        self.add_cut(self.model_row(coeffs, epigraph=epigraph), rhs)

    #: A cut coefficient below this fraction of the cut's largest is moved to the
    #: right-hand side at its worst case over the column's box: the cut stays
    #: valid (no feasible point is cut) and weakens by at most that fraction of
    #: the column's range. Measured (#1775 G4.3): one 6.2e-9 entry in an OA
    #: tangent made the default simplex B&B abandon a 19-column master after 11
    #: nodes with no bound; cleaned at 1e-7 it solves to optimality.
    CLEAN_REL = 1e-7

    def add_cut(self, coeffs: Any, rhs: float) -> None:
        """A cut over the relaxation's columns (``n_vars``, or ``n_vars + 1``
        with the epigraph column); :meth:`add_model_cut` for the model's."""
        row = np.array(coeffs, dtype=np.float64).reshape(-1)
        rhs = float(rhs)
        n = min(row.shape[0], self.n_vars)  # an epigraph column has no box
        big = float(np.max(np.abs(row))) if row.size else 0.0
        part = row[:n]
        lo, hi = self.lb[:n], self.ub[:n]
        boxed = np.isfinite(lo) & np.isfinite(hi) & (np.abs(lo) < 1e19) & (np.abs(hi) < 1e19)
        tiny = (part != 0.0) & (np.abs(part) < self.CLEAN_REL * big) & boxed
        for j in np.flatnonzero(tiny):
            a = float(part[j])
            rhs -= min(a * float(lo[j]), a * float(hi[j]))
            part[j] = 0.0
        self.cleaned_coefficients += int(np.count_nonzero(tiny))
        super().add_cut(row, rhs)

    def solve(
        self,
        time_limit: float,
        gap_tolerance: float = 1e-4,
        incumbent: Optional[float] = None,
        *,
        objective_override: Optional[Any] = None,
        exclude_also: Any = (),
        extra_cuts: Any = (),
    ) -> MasterResult:
        # ``objective_override`` and ``extra_cuts`` are over the relaxation's
        # columns, as every caller sizes them by ``n_vars`` (the init strategies).
        r = super().solve(
            time_limit,
            gap_tolerance,
            incumbent,
            objective_override=objective_override,
            exclude_also=exclude_also,
            extra_cuts=extra_cuts,
        )
        if r.x is not None:
            r.x = r.x[: self.n_model]
        return r


def build_master(lowered: Any, *, backend: str = DEFAULT_MASTER_BACKEND) -> MasterMILP:
    """The master of a lowered (algebraic) model: every row that is one linear row.

    The same extraction LOA performs (``_extract_body_coeffs`` is the single gate,
    #1039): a row it reads becomes a master row, any other row is left to the
    subproblems. A linear objective goes in the minimization convention
    (MAXIMIZE negated, constant kept, #1611/#1773); a nonlinear one leaves the
    master without an objective (``objective_convex`` is the caller's to set
    once it has a convexity certificate). Bounds and integrality are the
    lowered model's own.
    """
    from discopt._relax.deadline import setup_checkpoint
    from discopt._relax.gdp_reformulate import _extract_body_coeffs, _extract_body_terms
    from discopt.modeling.core import Constraint, ObjectiveSense, VarType

    variables = lowered._variables
    n = sum(int(v.size) for v in variables)

    def flat(attr: str) -> np.ndarray:
        return np.concatenate(
            [
                np.ravel(np.broadcast_to(np.asarray(getattr(v, attr), dtype=np.float64), v.shape))
                for v in variables
            ]
        )

    integrality = np.concatenate(
        [
            np.full(int(v.size), int(v.var_type in (VarType.BINARY, VarType.INTEGER)))
            for v in variables
        ]
    ).astype(np.int32)
    objective = None
    if lowered._objective is not None:
        coeffs = _extract_body_coeffs(lowered._objective_expression(), lowered, n)
        if coeffs is not None:
            c, off = coeffs
            if lowered._objective.sense == ObjectiveSense.MAXIMIZE:
                c, off = -c, -off
            objective = (np.asarray(c, dtype=np.float64), float(off))
    master = MasterMILP(
        n, flat("lb"), flat("ub"), integrality, objective=objective, backend=backend
    )
    for i_con, con in enumerate(lowered._constraints):
        if not i_con & 255:
            setup_checkpoint("the master build")  # #2079; a no-op outside a GDP setup
        if not isinstance(con, Constraint):
            continue
        # #2079: the row's terms, stored by its nonzeros -- exactly ``flatnonzero``
        # of the dense ``_extract_body_coeffs`` vector (slots past ``n`` dropped,
        # zero coefficients dropped, columns sorted), without writing it out.
        terms = _extract_body_terms(con.body, lowered)
        if terms is None:
            continue
        coeff_map, off = terms
        cols = sorted(k for k, v in coeff_map.items() if k < n and v != 0.0)
        sense = con.sense if isinstance(con.sense, str) else getattr(con.sense, "value", con.sense)
        master._add_sparse_row(
            np.asarray(cols, dtype=np.int64),
            np.asarray([coeff_map[k] for k in cols], dtype=np.float64),
            str(sense),
            float(con.rhs) - float(off),
        )
    return master

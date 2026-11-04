"""Irreducible Infeasible Subsystem (IIS) computation.

When a model is infeasible, the full constraint set does not explain *why* — most
constraints are irrelevant to the conflict. An **IIS** is a minimal subset of
constraints (and variable bounds) that is itself infeasible but becomes feasible
if *any* single member is removed. It is the smallest self-contained explanation
of the infeasibility, the analogue of Gurobi's ``computeIIS`` / CPLEX's conflict
refiner.

The algorithm is **deletion filtering** (Chinneck & Dravnieks 1991): start from
the full (infeasible) model and tentatively drop one candidate at a time; if the
remainder is still *provably* infeasible the candidate is redundant and is
dropped permanently, otherwise it is essential and kept. What remains is an IIS.

Soundness with an incomplete solver: a candidate is dropped only when the reduced
model is **proven** infeasible (``status == "infeasible"``). A solve that is
merely inconclusive (e.g. a nonconvex MINLP that hits the time limit without a
proof) is treated as "keep the candidate". The returned set is therefore always a
genuine infeasible subsystem; it is *irreducible* exactly when every filtering
solve was conclusive (always the case for LP/MILP/convex models). See
:attr:`IISResult.proven_irreducible`.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from discopt.modeling.core import (
    Constant,
    Constraint,
    Objective,
    ObjectiveSense,
    VarType,
)

if TYPE_CHECKING:
    from discopt.modeling.core import Model, Variable


# discopt treats |bound| >= this as unbounded; relaxing a bound sets it here.
_INF = 1e20
_FINITE = 1e19


@dataclass
class IISResult:
    """An Irreducible Infeasible Subsystem.

    Attributes
    ----------
    constraints : list[Constraint]
        Constraints belonging to the IIS.
    variable_bounds : list[tuple[Variable, str]]
        Variables with at least one bound in the IIS, each as
        ``(variable, "lower"|"upper")`` (one entry per variable and side).
    bound_elements : list[tuple[Variable, str, tuple]]
        The same bounds at **element** granularity, ``(variable, side, index)``
        with ``index`` the element's multi-index (``()`` for a scalar). For a
        vector variable only the elements whose bound is essential are listed,
        so ``z[0] <= 1`` and ``z[1] <= 1`` rather than the whole ``z <= 1``
        (#1618 C-17).
    integrality_elements : list[tuple[Variable, tuple]]
        Only with ``include_integrality=True``: the integer/binary elements whose
        **integrality** is in the IIS, as ``(variable, index)``. Every other
        integrality is relaxed. The bounds of an element listed here always
        apply and are not listed in ``bound_elements``; a relaxed element's
        bounds are ordinary members. Empty when integrality was not filtered
        (then every integrality is implicitly part of the subsystem).
    certificates : dict[tuple, numpy.ndarray]
        Only with ``include_integrality=True``: for a kept member, a point
        proving it is necessary -- it satisfies every *other* member (within
        ``1e-6``) and violates this one. Keys are ``("constraint", i)`` with
        ``i`` an index into ``model._constraints``, ``("bound", col, side)`` and
        ``("integrality", col)``, where ``col`` is the flat column (the
        concatenated C-order layout of ``model._variables``); values are flat
        points in that layout. A member with a certificate is proven
        necessary by arithmetic alone; one without (its probe point failed the
        re-check) rests on the solver's feasible verdict. See
        :attr:`fully_certified`.
    confirm_fallback : bool
        Only with ``include_integrality=True``: the finished subsystem failed its
        two-configuration infeasibility check, so the result comes from a full
        rerun that confirms every verdict (see :mod:`discopt._iis_milp`).
    n_solves : int
        Number of feasibility solves performed (deletion-filter cost).
    proven_irreducible : bool
        ``True`` when every filtering solve was conclusive, so the result is a
        true (minimal) IIS. ``False`` means some solves were inconclusive (or the
        budget ran out) and the set is a valid — but possibly non-minimal —
        infeasible subsystem.
    complete : bool
        ``False`` when ``total_time_limit`` stopped filtering early: an anytime
        result whose undecided members are still in the subsystem.
    portfolio_members : list[dict] or None
        Only for a ``portfolio`` run: one entry per order, with ``order``,
        ``outcome`` (``"ok"``, ``"timeout"``, ``"abandoned"``, ``"not started"``),
        ``size``, ``complete`` and ``wall`` (seconds).
    """

    constraints: list[Constraint] = field(default_factory=list)
    variable_bounds: list[tuple["Variable", str]] = field(default_factory=list)
    bound_elements: list[tuple["Variable", str, tuple]] = field(default_factory=list)
    integrality_elements: list[tuple["Variable", tuple]] = field(default_factory=list)
    certificates: dict[tuple, np.ndarray] = field(default_factory=dict)
    confirm_fallback: bool = False
    n_solves: int = 0
    proven_irreducible: bool = True
    complete: bool = True
    portfolio_members: list[dict] | None = None

    def __len__(self) -> int:
        # Members are counted at element granularity when it is known.
        n_bounds = len(self.bound_elements) if self.bound_elements else len(self.variable_bounds)
        return len(self.constraints) + n_bounds + len(self.integrality_elements)

    def __bool__(self) -> bool:
        return len(self) > 0

    @property
    def fully_certified(self) -> bool:
        """Every member carries a certificate, so irreducibility is proven by
        arithmetic alone, independent of any solver verdict. Always ``False``
        when certificates were not computed (``include_integrality=False``)."""
        return len(self) > 0 and len(self.certificates) == len(self)

    def summary(self) -> str:
        """A human-readable multi-line explanation of the infeasibility."""
        lines = [
            f"Irreducible Infeasible Subsystem ({len(self)} members"
            + ("" if self.proven_irreducible else ", not proven minimal")
            + (f", certified {len(self.certificates)}/{len(self)}" if self.certificates else "")
            + "):"
        ]
        if self.constraints:
            lines.append("  Constraints:")
            for c in self.constraints:
                label = c.name if c.name else repr(c)
                lines.append(f"    - {label}:  {c.body} {c.sense} {c.rhs:g}")
        elems = self.bound_elements or [(v, s, ()) for v, s in self.variable_bounds]
        if elems:
            lines.append("  Variable bounds:")
            for var, which, idx in elems:
                bound = np.asarray(var.lb if which == "lower" else var.ub, dtype=np.float64)
                op = ">=" if which == "lower" else "<="
                val = float(np.broadcast_to(bound, var.shape)[idx]) if idx else float(bound.flat[0])
                label = var.name + (f"[{', '.join(str(i) for i in idx)}]" if idx else "")
                lines.append(f"    - {label} {op} {val:g}")
        if self.integrality_elements:
            lines.append("  Integrality:")
            for var, idx in self.integrality_elements:
                label = var.name + (f"[{', '.join(str(i) for i in idx)}]" if idx else "")
                lines.append(f"    - {label} integer")
        if not self:
            lines.append("  (empty)")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _bound_candidates(model: "Model") -> list[tuple[int, str]]:
    """Finite lower/upper bounds of non-binary variables, as (var_index, side).

    Binary variables are skipped: their ``{0, 1}`` domain is intrinsic, not a
    removable bound. Infinite bounds cannot cause infeasibility and are excluded.

    A vector variable is a candidate when ANY element's bound is finite; the
    element refinement in :func:`compute_iis` then filters only the finite
    elements. Requiring EVERY element to be finite (as this once did) left the
    finite bounds of a partly-free vector applied in every probe yet never
    reported, so the returned "IIS" could be feasible on its own (MIPcc27
    ``dell``).
    """
    items: list[tuple[int, str]] = []
    for vi, v in enumerate(model._variables):
        if v.var_type == VarType.BINARY:
            continue
        if float(np.max(np.asarray(v.lb))) > -_FINITE:
            items.append((vi, "lower"))
        if float(np.min(np.asarray(v.ub))) < _FINITE:
            items.append((vi, "upper"))
    return items


def _solve_is_infeasible(
    model: "Model",
    active_constraints: list[Constraint],
    relaxed_bounds: set[tuple],
    saved_bounds: list[tuple[np.ndarray, np.ndarray]],
    time_limit: float,
) -> tuple[bool, bool]:
    """Solve the model restricted to ``active_constraints`` with ``relaxed_bounds``
    dropped, under a feasibility (constant) objective.

    Returns ``(is_infeasible, conclusive)``: ``is_infeasible`` is True only on a
    proven ``"infeasible"`` status; ``conclusive`` is False when the solver could
    neither prove infeasibility nor find a feasible point.
    """
    # Deferred: ``import discopt`` loads this module eagerly and must not pull in
    # the solver with it.
    from discopt.solver import probe_relaxed_bounds

    saved_cons = model._constraints
    saved_obj = model._objective
    model._constraints = active_constraints
    model._objective = Objective(Constant(0.0), ObjectiveSense.MINIMIZE)
    # Every (var, element, side) this probe sets to the ±_INF sentinel, so the
    # solve's large-bound warning does not blame the user for them (#1723).
    marked: set[tuple[int, int, str]] = set()
    for item in relaxed_bounds:
        # ``(vi, side)`` relaxes the whole variable; ``(vi, side, flat_j)`` one element.
        vi, side = item[0], item[1]
        v = model._variables[vi]
        cur = np.array(v.lb if side == "lower" else v.ub, dtype=np.float64)
        if len(item) == 2:
            cur = np.full_like(cur, -_INF if side == "lower" else _INF)
            marked.update((vi, j, side) for j in range(v.size))
        else:
            cur = np.array(np.broadcast_to(cur, v.shape), dtype=np.float64)
            cur.reshape(-1)[item[2]] = -_INF if side == "lower" else _INF
            marked.add((vi, item[2], side))
        if side == "lower":
            v.lb = cur
        else:
            v.ub = cur
    try:
        # ``solve`` is typed ``SolveResult | Iterator`` (streaming overload); the
        # non-streaming call returns a SolveResult with ``.status``.
        with probe_relaxed_bounds(model, marked):
            status = getattr(model.solve(time_limit=time_limit), "status", "error")
    # No ``except``: a probe that RAISES is a defect, not an inconclusive solve.
    # Swallowing it turned a tape-lowering bug (MissingJaxError) into "could not
    # prove the model infeasible" and would silently keep every element a broken
    # probe touched. A *returned* non-proof status stays inconclusive (below).
    finally:
        model._constraints = saved_cons
        model._objective = saved_obj
        for vi, (lb, ub) in zip(range(len(model._variables)), saved_bounds):
            model._variables[vi].lb = lb
            model._variables[vi].ub = ub

    if status == "infeasible":
        return True, True
    if status in ("optimal", "feasible"):
        return False, True
    return False, False  # inconclusive (time_limit / error / unbounded)


def compute_iis(
    model: "Model",
    *,
    include_bounds: bool = True,
    include_integrality: bool = False,
    time_limit: float = 30.0,
    total_time_limit: float | None = None,
    portfolio: Sequence[str] | None = None,
    workers: int | None = None,
) -> IISResult:
    """Compute an Irreducible Infeasible Subsystem of an infeasible ``model``.

    Parameters
    ----------
    model : Model
        A model whose constraints (and optionally variable bounds) are
        collectively infeasible.
    include_bounds : bool, default True
        Also consider finite variable bounds as removable members of the IIS.
        Required to explain infeasibilities driven by bounds (e.g. ``x >= 5`` and
        ``x <= 2`` declared as bounds rather than constraints).
    include_integrality : bool, default False
        Also treat the **integrality of each integer/binary element** as a
        removable member (Guieu & Chinneck 1999), so the IIS says which
        integrality restrictions the conflict actually needs -- often far fewer
        than all of them. Linear (LP/MILP) models only; runs the matrix-form
        engine in :mod:`discopt._iis_milp`, whose returned subsystem is proven
        infeasible with presolve on AND off (the #1634 rule; intermediate
        verdicts are confirmed once, on the finished set, with a full per-probe
        rerun if that check fails -- see ``confirm`` there) and returns a feasibility
        certificate for each kept member whose probe point passes an arithmetic
        re-check (``IISResult.certificates``; ``IISResult.fully_certified`` says
        whether every member has one). The default
        keeps today's semantics (and Gurobi/CPLEX's): every integrality is part
        of the subsystem and only constraints and bounds are filtered.
    time_limit : float, default 30.0
        Per-solve wall-clock limit for each feasibility probe.
    total_time_limit : float, optional
        Wall-clock budget for the whole computation. Each probe's limit is cut
        to the time left; when the budget runs out filtering stops and the
        current subsystem is returned with ``complete=False`` and
        ``proven_irreducible=False`` -- still proven infeasible, possibly with
        undecided members. With ``include_integrality=True`` every HiGHS run is
        also stopped by a wall-clock interrupt (HiGHS's own limit was measured
        overrunning 7-10x), and :class:`TimeoutError` is raised if no subsystem
        can be confirmed infeasible in time. On the default path the bound is
        as good as ``Model.solve``'s ``time_limit``.
    portfolio : sequence of str, optional
        Only with ``include_integrality=True``. Run one deletion filter per
        listed order, each in its own process, against one shared
        ``total_time_limit``, and return the smallest complete result (else the
        smallest anytime one). An order is a permutation of ``"ICB"`` --
        integrality, constraints, bounds phases -- with optional flags after a
        ``"-"``: ``r`` visits elements in reverse, ``n`` skips the candidate
        stages -- the SAT core on pure-binary models and the mod-2 (parity)
        certificate on integer equality systems (``"CIB-r"``, ``"ICB-n"``);
        :data:`discopt._iis_milp.DEFAULT_PORTFOLIO` is
        a measured default. Which IIS a filter finds depends on its order, so a
        portfolio buys size (and, when orders differ in speed, time) with cores.
        Each member's result is sound on its own; ``IISResult.portfolio_members``
        reports all of them. ``None`` (default) runs the single ``"ICB"`` order
        in-process.
    workers : int, optional
        Processes for ``portfolio`` (default: one per order, at most the CPU
        count). Orders beyond ``workers`` start as earlier members finish.

    Returns
    -------
    IISResult
        The constraints and variable bounds forming the IIS.

    Raises
    ------
    ValueError
        If ``model`` is not (provably) infeasible to begin with — there is no
        infeasibility to explain.
    NotImplementedError
        ``include_integrality=True`` on a model with a nonlinear or
        non-algebraic constraint.
    TimeoutError
        ``total_time_limit`` ran out before any subsystem was proven infeasible.
    """
    if include_integrality:
        return _compute_iis_with_integrality(
            model,
            include_bounds=include_bounds,
            time_limit=time_limit,
            total_time_limit=total_time_limit,
            portfolio=portfolio,
            workers=workers,
        )
    if portfolio is not None or workers is not None:
        raise ValueError("portfolio/workers require include_integrality=True")
    deadline = None if total_time_limit is None else time.perf_counter() + total_time_limit

    all_constraints = list(model._constraints)
    bound_items: list[tuple] = _bound_candidates(model) if include_bounds else []

    # Snapshot every variable's bounds once; each probe restores from this.
    saved_bounds = [(np.asarray(v.lb).copy(), np.asarray(v.ub).copy()) for v in model._variables]

    # Active set: indices into all_constraints, plus active bound items.
    active_con_idx = set(range(len(all_constraints)))
    active_bounds: set[tuple] = set(bound_items)
    n_solves = 0
    proven = True

    complete = True

    def out_of_time() -> bool:
        """Budget exhausted: stop filtering. Every drop so far was proven, so the
        current subsystem is still infeasible; it is just not reduced further."""
        nonlocal complete, proven
        if deadline is not None and time.perf_counter() >= deadline:
            complete = False
            proven = False
        return not complete

    def probe() -> tuple[bool, bool]:
        nonlocal n_solves
        n_solves += 1
        cons = [all_constraints[i] for i in sorted(active_con_idx)]
        relaxed = set(bound_items) - active_bounds
        tl = time_limit
        if deadline is not None:
            tl = max(0.0, min(tl, deadline - time.perf_counter()))
        return _solve_is_infeasible(model, cons, relaxed, saved_bounds, tl)

    # Baseline: the full system must be provably infeasible.
    base_infeasible, base_conclusive = probe()
    if not base_infeasible and deadline is not None and time.perf_counter() >= deadline:
        raise TimeoutError(
            "compute_iis: total_time_limit ran out before the model was proven infeasible"
        )
    if not base_infeasible:
        if base_conclusive:
            raise ValueError(
                "compute_iis: the model is feasible — there is no infeasibility to explain."
            )
        raise ValueError(
            "compute_iis: could not prove the model infeasible within the time limit; "
            "an IIS is only defined for an infeasible model."
        )

    # Deletion filter over constraints, then bounds. Drop a member only when the
    # remainder is *proven* infeasible; an inconclusive probe keeps the member
    # (and flags the result as not-proven-minimal).
    for i in list(active_con_idx):
        if out_of_time():
            break
        active_con_idx.discard(i)
        infeasible, conclusive = probe()
        if infeasible:
            continue  # constraint i is redundant to the conflict
        active_con_idx.add(i)  # essential — keep it
        if not conclusive:
            proven = False

    for item in list(active_bounds):
        if out_of_time():
            break
        active_bounds.discard(item)
        infeasible, conclusive = probe()
        if infeasible:
            continue
        active_bounds.add(item)
        if not conclusive:
            proven = False

    # Element refinement (#1618 C-17): a surviving *vector* bound is split into
    # its elements and filtered again, so the IIS names ``z[0] <= 1`` rather than
    # the whole ``z <= 1``. Only surviving variables are refined, so an IIS that
    # involves no vector bound costs no extra probes.
    for vi, side in sorted(i for i in active_bounds if len(i) == 2):
        if out_of_time():
            break
        v = model._variables[vi]
        if v.size <= 1:
            continue
        bnd = np.broadcast_to(
            np.asarray(saved_bounds[vi][0 if side == "lower" else 1], dtype=np.float64), v.shape
        ).reshape(-1)
        finite = [
            j for j in range(v.size) if (bnd[j] > -_FINITE if side == "lower" else bnd[j] < _FINITE)
        ]
        # Swap the whole-variable member for its finite elements, then filter them.
        active_bounds.discard((vi, side))
        bound_items.remove((vi, side))
        elems = [(vi, side, j) for j in finite]
        bound_items.extend(elems)
        active_bounds.update(elems)
        for e in elems:
            if out_of_time():
                break
            active_bounds.discard(e)
            infeasible, conclusive = probe()
            if infeasible:
                continue
            active_bounds.add(e)
            if not conclusive:
                proven = False

    elements: list[tuple["Variable", str, tuple]] = []
    for item in sorted(active_bounds, key=lambda t: (t[0], t[1], t[2] if len(t) > 2 else -1)):
        v = model._variables[item[0]]
        idx: tuple = ()
        if len(item) == 3:
            idx = tuple(int(i) for i in np.unravel_index(item[2], v.shape))
        elements.append((v, item[1], idx))
    per_var: list[tuple["Variable", str]] = []
    for v, side, _ in elements:
        if not any(pv is v and ps == side for pv, ps in per_var):
            per_var.append((v, side))

    return IISResult(
        constraints=[all_constraints[i] for i in sorted(active_con_idx)],
        variable_bounds=per_var,
        bound_elements=elements,
        n_solves=n_solves,
        proven_irreducible=proven,
        complete=complete,
    )


def _compute_iis_with_integrality(
    model: "Model",
    *,
    include_bounds: bool,
    time_limit: float,
    total_time_limit: float | None,
    portfolio: Sequence[str] | None = None,
    workers: int | None = None,
) -> IISResult:
    """:func:`compute_iis` with integrality elements, via the matrix engine."""
    from discopt._iis_milp import milp_iis

    res = milp_iis(
        model,
        include_bounds=include_bounds,
        time_limit=time_limit,
        total_time_limit=total_time_limit,
        portfolio=portfolio,
        workers=workers,
    )

    # Flat column -> (variable, multi-index).
    owners: list[tuple["Variable", int]] = []
    for v in model._variables:
        owners.extend((v, k) for k in range(v.size))

    def elem(col: int) -> tuple["Variable", tuple]:
        v, k = owners[col]
        idx = tuple(int(i) for i in np.unravel_index(k, v.shape)) if v.shape else ()
        return v, idx

    bound_elements: list[tuple["Variable", str, tuple]] = []
    for j, side in res.bounds:
        bv, bidx = elem(j)
        bound_elements.append((bv, side, bidx))
    per_var: list[tuple["Variable", str]] = []
    for pv_, side, _ in bound_elements:
        if not any(pv is pv_ and ps == side for pv, ps in per_var):
            per_var.append((pv_, side))
    all_constraints = list(model._constraints)
    return IISResult(
        constraints=[all_constraints[i] for i in res.constraints],
        variable_bounds=per_var,
        bound_elements=bound_elements,
        integrality_elements=[elem(j) for j in res.integrality],
        certificates=dict(res.certificates),
        confirm_fallback=res.fallback,
        n_solves=res.n_solves,
        proven_irreducible=res.proven_irreducible,
        complete=res.complete,
        portfolio_members=res.members,
    )

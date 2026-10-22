"""Logic-based discrete-steepest descent (#1775 G4.5, spec §5.3; Bernal et al. 2022).

:class:`LDSDA` is a **local** search over *external variables*: each is an
``exactly(1)`` group of Booleans -- a disjunction's disjuncts, or the operands of
an ``exactly(1, ...)`` logical constraint -- whose integer value ``1..k`` says
which member is True. A point fixes every group; its subproblem is the work model
with those Booleans pinned (the rest of the GDP stays as it is), solved by discopt
under a per-subproblem budget, and a verified point is an incumbent.

* **Search** (as Pyomo's ``gdpopt.ldsda``): from ``starting_point``, a *neighbor
  search* over the L2 (``±e_i``) or L∞ (every ``{-1, 0, 1}`` step) directions
  moves to the best improving neighbor -- ties within ``integer_tolerance`` go to
  the farther one -- then a *line search* repeats that direction while it keeps
  improving. A point with no improving neighbor is a local optimum and ends the
  run. Points outside the groups' ranges, already explored, or refuted by the
  logic (:mod:`discopt.gdp.propagate`) are not solved.
* **No bound** (§5.4): the method proves nothing about the configurations it
  never visits, so ``bound`` is ``None`` and the status is never ``optimal``;
  with a verified incumbent it is ``feasible``.

External groups are named: ``logical_constraint_list`` takes the names of
``exactly(1)`` logical constraints, ``disjunction_list`` the names of root
disjunctions. With neither, every root exactly-one disjunction is a group.
"""

from __future__ import annotations

import itertools
import math
import warnings
from typing import Any, Optional, Sequence

import numpy as np

from discopt.gdp.algorithms.base import GDPAlgorithm
from discopt.gdp.algorithms.init import GDPSpace
from discopt.modeling.core import Model, SelectorCardinality

#: Neighborhoods (Pyomo's ``direction_norm``).
DIRECTION_NORMS = ("L2", "Linf")
#: ``"local"``: a point whose pins, propagated through the logic, decide every
#: active disjunction is solved as its reduced subproblem by the local NLP (as
#: Pyomo's test drives LD-SDA, with Ipopt); otherwise -- or when the local solve
#: proves nothing -- by discopt on the pinned model. ``"global"``: always the latter.
LDSDA_SUBPROBLEMS = ("local", "global")
#: The local NLP solvers a ``"local"`` subproblem tries, in order.
LOCAL_NLP_SOLVERS = ("pounce", "ipopt")
#: The per-subproblem budget ceiling (seconds), as LBB's node budget (#1153).
SUBPROBLEM_TIME_LIMIT = 30.0


def directions(dimension: int, norm: str) -> list[tuple[int, ...]]:
    """The neighborhood's steps: ``±e_i`` (L2) or every nonzero ``{-1, 0, 1}^d`` (L∞)."""
    if norm == "L2":
        out = []
        for i in range(dimension):
            for s in (1, -1):
                out.append(tuple(s if j == i else 0 for j in range(dimension)))
        return out
    if norm == "Linf":
        return [d for d in itertools.product((-1, 0, 1), repeat=dimension) if any(d)]
    raise ValueError(f"direction_norm must be one of {DIRECTION_NORMS}, got {norm!r}")


class LDSDA(GDPAlgorithm):
    """Logic-based discrete-steepest descent; see the module docstring."""

    route = "gdpopt-ldsda: logic-based discrete-steepest descent (local)"
    bound_needs_resolution = False

    def __init__(
        self,
        model: Model,
        *,
        starting_point: Optional[Sequence[int]] = None,
        logical_constraint_list: Optional[Sequence[str]] = None,
        disjunction_list: Optional[Sequence[str]] = None,
        direction_norm: str = "L2",
        time_limit: Optional[float] = None,
        gap_tolerance: float = 1e-4,
        integer_tolerance: float = 1e-5,
        subproblem_time_limit: Optional[float] = None,
        subproblem: str = "local",
        propagate: bool = True,
        iteration_limit: Optional[int] = None,
        **kw: Any,
    ) -> None:
        self.space = GDPSpace.from_model(model)
        super().__init__(
            self.space.work,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            iteration_limit=iteration_limit,
            **kw,
        )
        # A local method: no configuration it skips is bounded (§5.4).
        self.lb_valid = False
        self.lb_invalid_reason = "LD-SDA is a local search; it proves no bound"
        self.groups = self._groups(logical_constraint_list, disjunction_list)
        if not self.groups:
            raise ValueError("ldsda: the model has no external variables to search over")
        self.directions = directions(len(self.groups), direction_norm)
        self.direction_norm = direction_norm
        from discopt.gdp.propagate import LogicChecker

        self.checker: Optional[LogicChecker] = LogicChecker(self.space) if propagate else None
        if starting_point is None:
            starting_point = self._default_start()
        point = tuple(int(p) for p in starting_point)
        if len(point) != len(self.groups) or any(
            not 1 <= p <= len(g) for p, g in zip(point, self.groups)
        ):
            raise ValueError(
                f"ldsda: starting_point {list(point)} does not fit the {len(self.groups)} "
                f"external variables with ranges {[len(g) for g in self.groups]}"
            )
        if subproblem not in LDSDA_SUBPROBLEMS:
            raise ValueError(f"subproblem must be one of {LDSDA_SUBPROBLEMS}, got {subproblem!r}")
        self.subproblem = subproblem
        self.n_local = 0
        self.n_global = 0
        self.integer_tolerance = float(integer_tolerance)
        if subproblem_time_limit is None:
            # As LBB's node budget: a tenth of the run, within [1 s, 30 s] (#1153).
            subproblem_time_limit = SUBPROBLEM_TIME_LIMIT
            if time_limit is not None:
                subproblem_time_limit = min(
                    SUBPROBLEM_TIME_LIMIT, max(1.0, 0.1 * float(time_limit))
                )
        if not subproblem_time_limit > 0:
            raise ValueError(f"subproblem_time_limit must be > 0, got {subproblem_time_limit!r}")
        self.subproblem_time_limit = float(subproblem_time_limit)
        self.current = point
        self.explored: dict[tuple[int, ...], Optional[float]] = {}
        self.best_direction: Optional[tuple[int, ...]] = None
        self.n_logic_pruned = 0
        self.local_optimum = False
        #: Why the search stopped with nowhere to go: ``"local_optimum"`` (the point
        #: has a value and no neighbor improves it) or ``"no_feasible_neighbor"``.
        self._dead_end: Optional[str] = None
        self._phase = "initial"

    # ── external variables ───────────────────────────────────────────────

    def _groups(
        self, logical: Optional[Sequence[str]], disjunctions: Optional[Sequence[str]]
    ) -> list[list[Any]]:
        """Each external variable as its ordered BooleanVar members."""
        from discopt.modeling.core import BooleanVar, LogicalExactly

        tree = self.space.tree
        groups: list[list[Any]] = []
        if logical is not None:
            by_name = {lc.name: lc for lc in tree.logic if lc.name is not None}
            for name in logical:
                lc = by_name.get(name)
                if lc is None:
                    raise ValueError(f"ldsda: no logical constraint named {name!r}")
                e = lc.expression
                if not (
                    isinstance(e, LogicalExactly)
                    and int(e.k) == 1
                    and all(isinstance(o, BooleanVar) for o in e.operands)
                ):
                    raise ValueError(
                        f"ldsda: logical constraint {name!r} is not exactly(1, ...) over "
                        "Boolean variables, so it defines no external variable"
                    )
                groups.append(list(e.operands))

        def choice(node: Any) -> bool:
            return node.kind != "implication" and (
                node.semantics.cardinality is SelectorCardinality.EXACTLY_ONE
            )

        if disjunctions is not None:
            roots = {r.name: r for r in tree.disjunctions if r.name is not None}
            for name in disjunctions:
                node = roots.get(name)
                if node is None:
                    raise ValueError(
                        f"ldsda: no root disjunction named {name!r} (roots: {sorted(roots)})"
                    )
                if not choice(node):
                    raise ValueError(
                        f"ldsda: disjunction {name!r} is not an exactly-one disjunction, "
                        "so it defines no external variable"
                    )
                groups.append([d.indicator for d in node.disjuncts])
        elif logical is None:
            # Every root exactly-one disjunction, named or not; any other root
            # (OR, an implication) stays a disjunction in the subproblem.
            groups.extend(
                [d.indicator for d in node.disjuncts] for node in tree.disjunctions if choice(node)
            )
        return groups

    def _default_start(self) -> list[int]:
        """Group by group, the first value the logic does not refute given the
        earlier groups' (Pyomo requires a ``starting_point``; ``(1, ..., 1)`` can be
        refuted outright -- every point around it was on syngas -- leaving nothing
        to search). A group none of whose values survives takes 1."""
        point: list[int] = []
        for g in self.groups:
            for value in range(1, len(g) + 1):
                pins = self._pins(tuple(point + [value]), partial=True)
                if pins and (self.checker is None or not self.checker.refutes(pins)):
                    point.append(value)
                    break
            else:
                point.append(1)
        return point

    def _pins(self, point: tuple[int, ...], partial: bool = False) -> dict[int, int]:
        """Flat column -> 0/1 for every external Boolean at ``point``."""
        from discopt.validation.feasibility import _column_indices

        pins: dict[int, int] = {}
        if not partial and len(point) != len(self.groups):
            raise RuntimeError("ldsda: a point fixes every external variable")
        for value, group in zip(point, self.groups):
            for j, b in enumerate(group, start=1):
                cols = _column_indices(b.variable, self.space.work)
                if cols.size != 1:
                    raise RuntimeError(f"ldsda: Boolean {b!r} is not one column")
                on = int(j == value)
                col = int(cols[0])
                if pins.get(col, on) != on:
                    return {}  # one Boolean asked to be both True and False
                pins[col] = on
        return pins

    # ── one point ────────────────────────────────────────────────────────

    def _solve_point(self, point: tuple[int, ...]) -> Optional[float]:
        """The verified objective (minimization convention) at ``point``, or ``None``."""
        if point in self.explored:
            return self.explored[point]
        pins = self._pins(point)
        self.explored[point] = None
        if not pins or (self.checker is not None and self.checker.refutes(pins)):
            self.n_logic_pruned += 1
            return None
        limit = self.deadline.sub_limit(self.subproblem_time_limit)
        if not limit > 0.0:
            del self.explored[point]  # never solved: it may be visited again
            return None
        x = None
        if self.subproblem == "local":
            x = self._local(pins, limit)
            limit = self.deadline.sub_limit(self.subproblem_time_limit)
        if x is None and limit > 0.0:
            x = self._global(pins, limit)
        if x is None or not self.offer_incumbent(x):
            return None
        from discopt.validation.feasibility import verify_point

        vr = verify_point(self.space.work, x, with_objective=True)
        assert vr.objective is not None  # offer_incumbent verified it
        obj = self.obj_sign * float(vr.objective)
        self.explored[point] = obj
        return obj

    def _pinned_rows(self, model: Model, pins: dict[int, int]) -> list:
        """``column == value`` rows over ``model``'s variables (laid out as work's)."""
        offsets = np.cumsum([0] + [int(v.size) for v in model._variables])
        rows = []
        for col, on in pins.items():
            k = int(np.searchsorted(offsets, col, side="right")) - 1
            var = model._variables[k]
            expr = (
                var
                if int(var.size) == 1
                else var[np.unravel_index(col - int(offsets[k]), var.shape)]
            )
            rows.append(expr == float(on))
        return rows

    def _global(self, pins: dict[int, int], limit: float) -> Optional[np.ndarray]:
        """discopt on the work model with ``pins`` added: the rest of the GDP stays."""
        from discopt.gdp.algorithms.subproblem import _flat

        sub = self.space.work.clone()
        for row in self._pinned_rows(sub, pins):
            sub.subject_to(row)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            r = sub.solve(time_limit=limit, gap_tolerance=self.gap_tolerance)
        self.n_subproblems += 1
        self.n_global += 1
        return None if r.x is None else _flat(self.space.work, r.x)

    def _local(self, pins: dict[int, int], limit: float) -> Optional[np.ndarray]:
        """The local NLP on the reduced subproblem, when the logic decides the GDP."""
        from discopt._relax.gdp_reformulate import _extract_body_coeffs
        from discopt.gdp.algorithms.subproblem import _local
        from discopt.gdp.propagate import _kleene
        from discopt.gdp.state import reduced_subproblem
        from discopt.modeling.core import Constraint, _LogicalConstraint

        assign = self._propagate(pins)
        if assign is None or self.checker is None:
            return None
        selection = self._configuration(assign)
        if selection is None:
            return None
        # On a private copy: the assigned Booleans are fixed by their bounds, as
        # Pyomo fixes them, and the rows that only restate a fixed value go. As
        # equality rows they leave a near-square model (Pyomo's
        # four_stage_dynamic_model: 275 equalities, 272 columns) with fewer degrees
        # of freedom than equalities, and the NLP refuses it (Ipopt status -10).
        sub = reduced_subproblem(self.space.work.clone(), selection)
        offsets = np.cumsum([0] + [int(v.size) for v in sub._variables])
        for col, value in assign.items():
            k = int(np.searchsorted(offsets, col, side="right")) - 1
            var = sub._variables[k]
            lb = np.array(var.lb, dtype=np.float64, copy=True).reshape(-1)
            ub = np.array(var.ub, dtype=np.float64, copy=True).reshape(-1)
            if not lb[col - offsets[k]] <= value <= ub[col - offsets[k]]:
                return None  # the assignment leaves the column's box
            lb[col - offsets[k]] = ub[col - offsets[k]] = float(value)
            var.lb, var.ub = lb.reshape(var.shape), ub.reshape(var.shape)
        n = int(offsets[-1])
        # The copy's logic reads the copy's columns, not the checker's.
        col_of = {id(v): int(offsets[k]) for k, v in enumerate(sub._variables) if int(v.size) == 1}

        def value_of(bv: Any) -> Optional[bool]:
            col = col_of.get(id(bv.variable))
            return None if col is None or col not in assign else bool(assign[col])

        kept = []
        for c in sub._constraints:
            if isinstance(c, _LogicalConstraint):
                # Satisfied by the assignment: it constrains nothing that is free.
                if _kleene(c.expression, value_of) is True:
                    continue
            elif type(c) is Constraint and c.sense == "==":
                coeffs = _extract_body_coeffs(c.body, sub, n)
                if coeffs is not None:
                    a, const = np.asarray(coeffs[0], dtype=np.float64), float(coeffs[1])
                    nz = np.flatnonzero(a)
                    # ``a x + const == rhs`` on one fixed column, at its fixed value.
                    if (
                        nz.size == 1
                        and int(nz[0]) in assign
                        and abs(a[nz[0]] * assign[int(nz[0])] + const - float(c.rhs)) <= 1e-12
                    ):
                        continue
            kept.append(c)
        sub._constraints = kept
        self.n_subproblems += 1
        self.n_local += 1
        # POUNCE first, then Ipopt when it is installed: on Pyomo's
        # four_stage_dynamic_model at (1, 4) POUNCE stops in its step computation
        # (status -3) while Ipopt converges in 0.1 s to the -23.3053 optimum.
        for nlp_solver in LOCAL_NLP_SOLVERS:
            try:
                res = _local(sub, self.deadline.sub_limit(limit), nlp_solver)
            except ImportError:
                continue  # cyipopt is an optional extra
            if res.status == "feasible":
                return res.x
        return None

    def _propagate(self, pins: dict[int, int]) -> Optional[dict[int, int]]:
        """``pins`` closed under the logic: every Boolean column one of whose values
        the logic refutes takes the other (unit propagation; exact). ``None`` when
        both values of a column are refuted."""
        if self.checker is None:
            return dict(pins)
        cols = sorted(set(self.space.columns.values()) | set(self.checker._col_of_var.values()))
        assign = dict(pins)
        changed = True
        while changed:
            changed = False
            for c in cols:
                if c in assign:
                    continue
                off = self.checker.refutes({**assign, c: 0})
                on = self.checker.refutes({**assign, c: 1})
                if off and on:
                    return None
                if off or on:
                    assign[c] = 1 if off else 0
                    changed = True
        return assign

    def _configuration(self, assign: dict[int, int]) -> Optional[list]:
        """The disjuncts ``assign`` selects, when it decides every active disjunction."""
        out: list = []

        def visit(node: Any) -> bool:
            if node.kind == "implication":
                return True  # left conditional in the reduced subproblem, which is exact
            chosen = []
            for d in node.disjuncts:
                v = assign.get(self.space.columns[d])
                if v is None:
                    return False
                if v == d.active_value:
                    chosen.append(d)
            if not chosen or (
                node.semantics.cardinality is SelectorCardinality.EXACTLY_ONE and len(chosen) != 1
            ):
                return False
            out.extend(chosen)
            return all(visit(c) for d in chosen for c in d.children)

        return out if all(visit(r) for r in self.space.tree.disjunctions) else None

    def _valid(self, point: tuple[int, ...]) -> bool:
        return point not in self.explored and all(
            1 <= p <= len(g) for p, g in zip(point, self.groups)
        )

    # ── the search ───────────────────────────────────────────────────────

    def iterate(self) -> bool:
        if self._phase == "initial":
            self._phase = "neighbor"
            self._solve_point(self.current)
            return True
        if self._phase == "line":
            nxt = tuple(c + s for c, s in zip(self.current, self.best_direction or ()))
            here = self.explored.get(self.current)
            if self._valid(nxt):
                obj = self._solve_point(nxt)
                if obj is not None and (here is None or obj < here - self.integer_tolerance):
                    self.current = nxt
                    return True
            self._phase = "neighbor"
            return True
        return self._neighbor_search()

    def _neighbor_search(self) -> bool:
        here = self.explored.get(self.current)
        best: Optional[tuple[int, ...]] = None
        best_obj, best_dist = math.inf, 0
        for step in self.directions:
            nb = tuple(c + s for c, s in zip(self.current, step))
            if not self._valid(nb):
                continue
            if self.deadline.expired():
                return True  # the loop stops on the deadline, as time_limit
            obj = self._solve_point(nb)
            if obj is None or (here is not None and obj >= here - self.integer_tolerance):
                continue  # does not improve on the current point
            dist = sum(s * s for s in step)
            if abs(obj - best_obj) < self.integer_tolerance:
                if dist > best_dist:
                    best, best_dist, self.best_direction = nb, dist, step
            elif obj < best_obj:
                best, best_obj, best_dist, self.best_direction = nb, obj, dist, step
        if best is None:
            # A point with a value and no better neighbor is a local optimum; one
            # with none is a dead end, which is not the same claim.
            self.local_optimum = here is not None
            self._dead_end = "local_optimum" if here is not None else "no_feasible_neighbor"
            return False
        self.current = best
        self._phase = "line"
        return True

    def run(self):
        result = super().run()
        if self.stop_reason == "exhausted" and self._dead_end is not None:
            result.solver_stats["stop_reason"] = self._dead_end
        return result

    def extra_stats(self) -> dict[str, Any]:
        return {
            "n_external": len(self.groups),
            "point": list(self.current),
            "n_points": len(self.explored),
            "n_logic_pruned": self.n_logic_pruned,
            "direction_norm": self.direction_norm,
            "n_local": self.n_local,
            "n_global": self.n_global,
        }


def solve_ldsda(model: Model, **options: Any):
    """Run :class:`LDSDA` on ``model`` and return its ``SolveResult``."""
    return LDSDA(model, **options).run()


__all__ = [
    "DIRECTION_NORMS",
    "LDSDA",
    "LDSDA_SUBPROBLEMS",
    "SUBPROBLEM_TIME_LIMIT",
    "directions",
    "solve_ldsda",
]

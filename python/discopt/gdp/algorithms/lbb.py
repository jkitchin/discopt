"""Logic-based branch and bound (#1775 G4.4 stage 1, spec §5.3; Lee & Grossmann 2000).

:class:`LBB` branches on disjunctions. A node decides some disjunctions -- a
partial selection -- and leaves the rest disjunctions:

* **The node relaxation** (as Pyomo's): the work model with the node's selection
  applied by :func:`~discopt.gdp.state.transform_current_disjunctive_state`
  (decided disjunctions pinned, their unselected rows gone), and the *nonlinear*
  rows of every still-undecided disjunct dropped -- a disjunct left with no row
  is a "do nothing" option, which every lowering keeps selectable. Dropping rows
  only relaxes, so discopt's certified bound for it, under a per-node budget, is
  a valid bound on the subtree; the relaxation keeps nodes cheap (measured on
  eight_process: keeping the rows made every node the whole MINLP again, and 16
  six-second node solves never closed what one 18 s solve certifies). A
  certified infeasibility closes the subtree; so does a node bound the incumbent
  closes, e.g. once the relaxation's own optimum verifies on the model.
* **A leaf** (every disjunction decided) is solved exactly: nothing is dropped,
  with the whole remaining budget, and a certified optimum closes it.
* **Best-first** on a heap keyed by node LB. A node's LB is its parent's,
  raised by its own certified bound; it is never taken from an uncertified solve.
  A node is pruned only when its LB closes the gap to the incumbent.
* **Branching** on the first undecided disjunction in tree order (Pyomo's
  "first unbranched"): one child per disjunct, or per nonempty subset under OR
  semantics (Pyomo raises ``NotImplementedError`` for OR). Choosing a disjunct
  makes its nested disjunctions branchable. Implications are never branched;
  their rows stay conditional, which is exact.
* **Logic propagation** (:mod:`discopt.gdp.propagate`, G5.3) replaces Pyomo's
  ``check_sat``: a child whose decided indicators the logic refutes is never
  solved. The refutation is exact.
* **Leaves.** A node with nothing left to branch that its solve did not close is
  *unresolved*: its bound (or ``-inf``) stays in the LB, and ``optimal`` is
  refused while it is open, unless the incumbent later closes the gap to it.

LB (§5.4): the minimum over open nodes and unresolved leaves. ``optimal`` when it
closes the gap with nothing unresolved; ``infeasible`` when every node closed
with no incumbent. Pyomo's time-limit branch raises ``AttributeError``
(``_get_final_results_object``); here the time limit is the base loop's, and the
LB it publishes is the one above.
"""

from __future__ import annotations

import bisect
import heapq
import itertools
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Optional

from discopt.gdp.algorithms.base import GDPAlgorithm
from discopt.gdp.algorithms.enumerate import _choices
from discopt.gdp.algorithms.init import GDPSpace
from discopt.gdp.tree import DisjunctionNode, DisjunctNode
from discopt.modeling.core import Model

#: The default per-node time budget (seconds): the whole of it without a time
#: limit, the ceiling of a tenth of the time limit otherwise.
NODE_TIME_LIMIT = 30.0


@dataclass
class _Node:
    lb: float
    selection: list[DisjunctNode]
    pending: list[DisjunctionNode]
    depth: int = 0
    assignment: dict[int, int] = field(default_factory=dict)


def _branchable(nodes: list[DisjunctionNode]) -> list[DisjunctionNode]:
    return [n for n in nodes if n.kind != "implication"]


class LBB(GDPAlgorithm):
    """Logic-based branch and bound; see the module docstring."""

    route = "gdpopt-lbb: logic-based branch and bound (global node solves)"
    # The LB is built from open nodes and unresolved leaves; it is valid throughout.
    bound_needs_resolution = False

    def __init__(
        self,
        model: Model,
        *,
        time_limit: Optional[float] = None,
        gap_tolerance: float = 1e-4,
        propagate: bool = True,
        node_time_limit: Optional[float] = None,
        iteration_limit: Optional[int] = None,
        **kw: Any,
    ) -> None:
        if node_time_limit is not None and not node_time_limit > 0:
            raise ValueError(f"node_time_limit must be > 0, got {node_time_limit!r}")
        self.space = GDPSpace.from_model(model)
        super().__init__(
            self.space.work,
            time_limit=time_limit,
            gap_tolerance=gap_tolerance,
            iteration_limit=iteration_limit,
            **kw,
        )
        from discopt.gdp.propagate import LogicChecker

        self.checker: Optional[LogicChecker] = LogicChecker(self.space) if propagate else None
        if node_time_limit is None:
            # A tenth of the run, at least 1 s, and never more than NODE_TIME_LIMIT:
            # a node budget that grew with the caller's would spend a bigger budget
            # on longer node solves instead of more nodes (#1153's saturation rule).
            node_time_limit = NODE_TIME_LIMIT
            if time_limit is not None:
                node_time_limit = min(NODE_TIME_LIMIT, max(1.0, 0.1 * float(time_limit)))
        self.node_time_limit = float(node_time_limit)
        self._tick = itertools.count()
        self._heap: list[tuple[float, int, _Node]] = []
        #: Unresolved leaves: id -> their bound (minimization convention; -inf if none).
        self.open_leaves: dict[int, float] = {}
        self.n_nodes = 0
        self.n_logic_pruned = 0
        self.n_bound_pruned = 0
        self.n_closed = 0
        self.n_rows_dropped = 0
        self.max_open = 0
        root = _Node(-math.inf, [], _branchable(list(self.space.tree.disjunctions)))
        if self.checker is not None and self.checker.refutes({}):
            self.n_logic_pruned += 1
        else:
            self._push(root)

    # ── the heap ──────────────────────────────────────────────────────────

    def _push(self, node: _Node) -> None:
        heapq.heappush(self._heap, (node.lb, next(self._tick), node))
        self.max_open = max(self.max_open, len(self._heap))

    def _closes(self, lb: float) -> bool:
        """Whether a region bounded below by ``lb`` cannot improve the incumbent."""
        from discopt.solvers._gap import optimality_gap

        if self.incumbent is None:
            return False
        if lb >= self.ub:
            return True
        g = optimality_gap(lb, self.ub, abs_tol=self.abs_gap_tolerance, denom_floor=1e-10)
        return g <= self.gap_tolerance

    def _assignment(self, parent: _Node, node: DisjunctionNode, chosen: list) -> dict[int, int]:
        """The parent's indicator assignment extended by ``node`` choosing ``chosen``."""
        a = dict(parent.assignment)
        cols = self.space.columns
        picked = {id(d) for d in chosen}

        def off(d: DisjunctNode) -> None:
            if d in cols:
                a[cols[d]] = 1 - d.active_value
            for child in d.children:
                if child.kind != "implication":
                    for dd in child.disjuncts:
                        off(dd)

        for d in node.disjuncts:
            if id(d) in picked:
                if d in cols:
                    a[cols[d]] = d.active_value
            else:
                off(d)
        return a

    # ── one node ──────────────────────────────────────────────────────────

    def _solve_node(self, node: _Node, leaf: bool) -> Any:
        from discopt.gdp.state import transform_current_disjunctive_state

        cap = None if leaf else self.node_time_limit
        limit = self.deadline.sub_limit(cap)
        if not limit > 0.0:
            return None
        sub = self.space.work.clone()
        transform_current_disjunctive_state(sub, node.selection, exact_or=True)
        if not leaf:
            self.n_rows_dropped += drop_undecided_nonlinear_rows(sub)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            return sub.solve(time_limit=limit, gap_tolerance=self.gap_tolerance)

    def iterate(self) -> bool:
        from discopt.gdp.algorithms.subproblem import _flat

        self._resolve_leaves()
        if not self._heap:
            self._finish()
            return False
        _, _, node = heapq.heappop(self._heap)
        if self._closes(node.lb):
            # Best-first: every open node is at least this one, so all close.
            self.n_bound_pruned += 1 + len(self._heap)
            self._heap.clear()
            self._finish()
            return False
        leaf = not node.pending
        r = self._solve_node(node, leaf)
        if r is None:  # no time left: the node stays open, the loop stops on the deadline
            self._push(node)
            self._publish()
            return True
        self.n_nodes += 1
        self.n_subproblems += 1
        if r.x is not None:
            x = _flat(self.space.work, r.x)
            if x is not None:
                self.offer_incumbent(x)
        certified = bool(r.gap_certified)
        if r.status == "infeasible" and certified:
            self.n_closed += 1
            self._publish()
            return True
        valid = r.bound_valid and r.bound is not None and math.isfinite(r.bound)
        lb = node.lb
        if valid:
            lb = max(lb, self.obj_sign * float(r.bound))
        if leaf and r.status == "optimal" and certified and valid:
            # A leaf's model is its configuration exactly: it is solved.
            self.n_closed += 1
            self._publish()
            return True
        if not leaf and self._closes(lb):
            # The relaxation's bound cannot beat the incumbent (which its own
            # optimum, verified above, may just have become).
            self.n_closed += 1
            self._publish()
            return True
        if leaf:
            key = next(self._tick)
            self.open_leaves[key] = lb
            self.mark_unresolved(key)
            self._resolve_leaves()
            self._publish()
            return True
        node.lb = lb
        self._branch(node)
        self._publish()
        return True

    def _branch(self, node: _Node) -> None:
        target, rest = node.pending[0], node.pending[1:]
        for chosen in _choices(target):
            nested = _branchable([c for d in chosen for c in d.children])
            assignment = self._assignment(node, target, chosen)
            if self.checker is not None and self.checker.refutes(assignment):
                self.n_logic_pruned += 1
                continue
            child = _Node(
                node.lb,
                node.selection + list(chosen),
                rest + nested,
                node.depth + 1,
                assignment,
            )
            if self._closes(child.lb):
                self.n_bound_pruned += 1
                continue
            self._push(child)

    # ── the bound ─────────────────────────────────────────────────────────

    def _resolve_leaves(self) -> None:
        """An unresolved leaf whose bound the incumbent now closes is resolved."""
        for key, lb in list(self.open_leaves.items()):
            if self._closes(lb):
                del self.open_leaves[key]
                self.mark_solved(key)

    def _publish(self) -> None:
        self._resolve_leaves()
        floors = [lb for lb, _, _ in self._heap] + list(self.open_leaves.values())
        if floors:
            lb = min(floors)
            if math.isfinite(lb):
                self.offer_bound(min(lb, self.ub))
        elif self.incumbent is not None:
            self.offer_bound(self.ub)

    def _finish(self) -> None:
        self._publish()
        if self._heap or self.open_leaves:
            return
        if self.incumbent is None and not self.unresolved:
            self.prove_infeasible()

    def extra_stats(self) -> dict[str, Any]:
        return {
            "n_nodes": self.n_nodes,
            "n_open": len(self._heap),
            "n_closed": self.n_closed,
            "n_logic_pruned": self.n_logic_pruned,
            "n_bound_pruned": self.n_bound_pruned,
            "n_unresolved_leaves": len(self.open_leaves),
            "max_open": self.max_open,
            "n_rows_dropped": self.n_rows_dropped,
        }


def drop_undecided_nonlinear_rows(model: Model) -> int:
    """Drop, in place, the nonlinear rows of every disjunct still undecided in
    ``model`` -- a node model, after its selection was applied, so every remaining
    ``either_or``/``add_disjunction``/``if_then`` row is undecided. Returns how many
    rows were dropped. Only a row whose body is not affine is dropped; anything
    else (affine rows, nested disjunctions, unknown items) is kept, which can only
    keep the relaxation tighter. ``model`` must be a private copy (``clone``).
    """
    from discopt._relax.gdp_reformulate import _extract_body_coeffs
    from discopt.modeling.core import Constraint, _DisjunctiveConstraint, _IndicatorConstraint

    n = sum(int(v.size) for v in model._variables)
    dropped = 0

    def nonlinear(c: Any) -> bool:
        return type(c) is Constraint and _extract_body_coeffs(c.body, model, n) is None

    def relax(dc: _DisjunctiveConstraint) -> None:
        nonlocal dropped
        groups, ms = [], []
        for j, group in enumerate(dc.disjuncts):
            keep, keep_m = [], []
            for i, item in enumerate(group):
                if isinstance(item, _DisjunctiveConstraint):
                    relax(item)
                elif nonlinear(item):
                    dropped += 1
                    continue
                keep.append(item)
                keep_m.append(None if dc.big_m is None else dc.big_m[j][i])
            groups.append(keep)
            ms.append(tuple(keep_m))
        dc.disjuncts = groups
        if dc.big_m is not None:
            dc.big_m = tuple(ms)

    rows, gone = [], []
    for i, c in enumerate(model._constraints):
        if isinstance(c, _DisjunctiveConstraint):
            relax(c)
        elif isinstance(c, _IndicatorConstraint) and nonlinear(c.constraint):
            dropped += 1
            gone.append(i)
            continue
        rows.append(c)
    model._constraints = rows
    members = model.__dict__.get("_gdp_block_members")
    if members and gone:
        # A block's ``start`` is a row position: move it down past every dropped
        # row before it, as ``transform_current_disjunctive_state`` does, or a
        # block left row-less is misplaced in the tree and the reader refuses it.
        model.__dict__["_gdp_block_members"] = {
            k: v._replace(start=v.start - bisect.bisect_left(gone, v.start))
            for k, v in members.items()
        }
    return dropped


def solve_lbb(model: Model, **options: Any):
    """Run :class:`LBB` on ``model`` and return its ``SolveResult``."""
    return LBB(model, **options).run()


__all__ = ["LBB", "NODE_TIME_LIMIT", "drop_undecided_nonlinear_rows", "solve_lbb"]

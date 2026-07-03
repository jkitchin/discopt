"""Per-node Lagrangian dual bound for the MILP branch-and-bound loop.

``solve_lagrangian`` (see :mod:`discopt.decomposition.lagrangian.solver`) runs a
full dual loop as a standalone solver. For use *inside* the B&B tree we instead
want a cheap, valid lower bound at each node: fix good multipliers ``λ*`` once at
the root, then at a node with tightened bounds ``[node_lb, node_ub]`` solve the
relaxed (coupling-dualized) subproblem as an integer program and return its
objective. Because the dualized term is ``<= 0`` at any feasible point and the
subproblem keeps its integrality, ``L(λ*)`` is a valid lower bound on the node's
optimum — and it can dominate the node LP relaxation exactly when the subproblem
has a nonzero integrality gap (the reason Lagrangian relaxation helps at all).

The bound is computed in the same ``min(c·x + const)`` convention the B&B uses,
so the driver can combine it with ``result_lbs[i] = max(result_lbs[i], L)``.
v1 is restricted to **minimization, linear** models with coupling structure;
``try_build`` returns ``None`` (hook disabled) otherwise.
"""

from __future__ import annotations

import logging
import time

import numpy as np

from discopt.decomposition._linear import extract_linear
from discopt.decomposition.structure import detect_decomposition
from discopt.modeling.core import Model, ObjectiveSense, VarType
from discopt.solvers import SolveStatus

logger = logging.getLogger(__name__)


class LagrangianNodeBounder:
    """Computes a valid Lagrangian lower bound for a B&B node.

    Build via :meth:`try_build` (returns ``None`` when the hook does not apply),
    call :meth:`solve_root_dual` once, then :meth:`node_bound` per node.
    """

    def __init__(
        self,
        c: np.ndarray,
        c_offset: float,
        A_c: np.ndarray,
        r_c: np.ndarray,
        A_b: np.ndarray,
        r_b: np.ndarray,
        integrality: np.ndarray,
        milp_solver,
    ) -> None:
        self.c = c
        self.c_offset = c_offset
        self.A_c = A_c
        self.r_c = r_c
        self.A_b = A_b
        self.r_b = r_b
        self.integrality = integrality
        self._milp = milp_solver
        self.n = c.shape[0]
        self.m_coup = A_c.shape[0]
        self.lam = np.zeros(self.m_coup)
        self.root_bound: float | None = None

    # ── construction ──────────────────────────────────────────

    @classmethod
    def try_build(
        cls,
        model: Model,
        *,
        prefer_pounce: bool = True,
    ) -> "LagrangianNodeBounder | None":
        """Build a bounder, or return ``None`` if the hook does not apply.

        Applies only to minimization, linear models that have coupling
        constraints (annotated via ``model.mark_coupling`` or auto-detected).
        """
        obj = model._objective
        if obj is not None and obj.sense != ObjectiveSense.MINIMIZE:
            return None  # v1: minimization only (sign-alignment safety)
        try:
            lin = extract_linear(model)
        except Exception:
            # Nonlinear constraint/objective, or an unsupported construct (e.g.
            # multi-dimensional index extraction): cleanly disable the hook.
            return None
        if not lin.minimize:
            return None
        try:
            structure = detect_decomposition(model)
        except Exception:
            return None
        coupling_src = set(structure.coupling_constraints)
        if not coupling_src:
            return None

        n = lin.n
        A_leq, b_leq, src_leq = lin.rows_leq()
        Ac_rows, rc_rows, Ab_rows, rb_rows = [], [], [], []
        for vec, rhs, src in zip(A_leq, b_leq, src_leq):
            if src in coupling_src:
                Ac_rows.append(vec)
                rc_rows.append(rhs)
            else:
                Ab_rows.append(vec)
                rb_rows.append(rhs)
        if not Ac_rows:
            return None
        A_c = np.array(Ac_rows)
        r_c = np.array(rc_rows)
        A_b = np.array(Ab_rows) if Ab_rows else np.zeros((0, n))
        r_b = np.array(rb_rows) if rb_rows else np.zeros(0)

        integrality = np.zeros(n, dtype=np.int32)
        off = 0
        for v in model._variables:
            if v.var_type in (VarType.BINARY, VarType.INTEGER):
                integrality[off : off + v.size] = 1
            off += v.size

        from discopt.solvers.lp_backend import get_milp_solver

        try:
            milp = get_milp_solver(prefer_pounce=prefer_pounce)
        except ImportError:
            return None
        return cls(lin.c, lin.c_offset, A_c, r_c, A_b, r_b, integrality, milp)

    # ── subproblem ────────────────────────────────────────────

    def _subproblem(
        self,
        lam: np.ndarray,
        node_lb: np.ndarray,
        node_ub: np.ndarray,
        time_limit: float,
    ) -> tuple[float | None, np.ndarray | None]:
        """Return ``(L, residual)`` for the relaxed subproblem, or ``(None, None)``."""
        c_lag = self.c + (self.A_c.T @ lam if self.m_coup else 0.0)
        bounds = [(float(node_lb[i]), float(node_ub[i])) for i in range(self.n)]
        res = self._milp(
            c_lag,
            A_ub=self.A_b if self.A_b.shape[0] else None,
            b_ub=self.r_b if self.A_b.shape[0] else None,
            bounds=bounds,
            integrality=self.integrality,
            time_limit=time_limit,
            gap_tolerance=1e-6,
        )
        if res.x is None:
            return None, None
        # Use the rigorous dual bound on the subproblem so L stays a valid
        # lower bound even if the subproblem solve stops early.
        sub_lb = (
            res.bound
            if res.bound is not None
            else (res.objective if res.status == SolveStatus.OPTIMAL else None)
        )
        if sub_lb is None:
            return None, None
        z = np.asarray(res.x, dtype=np.float64)
        L = float(sub_lb) - float(lam @ self.r_c) + self.c_offset
        residual = self.A_c @ z - self.r_c
        return L, residual

    # ── root dual ─────────────────────────────────────────────

    def solve_root_dual(
        self,
        root_lb: np.ndarray,
        root_ub: np.ndarray,
        *,
        max_iters: int = 40,
        time_budget: float = 5.0,
        alpha0: float = 1.5,
        patience: int = 5,
    ) -> float | None:
        """Subgradient ascent over the root box to fix good multipliers ``λ*``."""
        t0 = time.perf_counter()
        lam = np.zeros(self.m_coup)
        best = -np.inf
        alpha = alpha0
        stall = 0
        for _ in range(max_iters):
            if time.perf_counter() - t0 > time_budget:
                break
            L, residual = self._subproblem(lam, root_lb, root_ub, time_limit=2.0)
            if L is None or residual is None:
                break
            if L > best:
                best = L
                self.lam = lam.copy()
                stall = 0
            else:
                stall += 1
            gnorm2 = float(residual @ residual)
            if gnorm2 < 1e-16:
                break
            target = best + max(1.0, abs(best))  # optimistic (no UB at root)
            step = alpha * max(target - L, 1e-9) / gnorm2
            lam = np.maximum(0.0, lam + step * residual)
            if stall >= patience:
                alpha *= 0.5
                stall = 0
        self.root_bound = float(best) if best > -np.inf else None
        return self.root_bound

    # ── per-node bound ────────────────────────────────────────

    def node_bound(
        self,
        node_lb: np.ndarray,
        node_ub: np.ndarray,
        *,
        time_limit: float = 2.0,
    ) -> float | None:
        """Valid lower bound on the node's optimum at the fixed root ``λ*``."""
        L, _ = self._subproblem(self.lam, node_lb, node_ub, time_limit=time_limit)
        return L


__all__ = ["LagrangianNodeBounder"]

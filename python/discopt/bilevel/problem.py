"""Bilevel problem front-end.

``BilevelProblem`` mirrors the :class:`~discopt.ro.counterpart.RobustCounterpart`
builder: the user builds a normal :class:`~discopt.modeling.core.Model` holding
the **leader's** objective and constraints, describes the follower separately, and
calls :meth:`~BilevelProblem.formulate`, which rewrites the model in place into a
single-level MPEC that ``model.solve()`` handles.

Scope (``docs/dev/bilevel-module-plan.md`` §1): **optimistic** bilevel with a lower
level that is **convex in the follower variables** (an LP or convex-QP in ``y``), via
either the KKT reformulation (:mod:`~discopt.bilevel.kkt`, ``method="kkt"``) or the
strong-duality reformulation (:mod:`~discopt.bilevel.strong_duality`,
``method="strong_duality"``). Finite follower-variable bounds are folded into the
KKT system automatically. Everything the reduction is unsound for is refused loudly
rather than silently approximated:

* integer follower variables — KKT does not characterize integer optima;
* a lower level that is **nonconvex in** ``y`` — refused outright; a non-quadratic
  (variable-dependent curvature) lower level is refused pending the full convexity
  certifier (the gate names the offending expression);
* pessimistic semantics.

Example
-------
>>> from discopt.modeling.core import Model
>>> from discopt.bilevel import BilevelProblem
>>> m = Model("toll")
>>> x = m.continuous("x", lb=0, ub=10)          # leader (toll)
>>> y = m.continuous("y", lb=0, ub=10)          # follower (flow)
>>> m.minimize(x - 4 * y)                        # leader objective
>>> bl = BilevelProblem(
...     m, upper_vars=[x], lower_vars=[y],
...     lower_objective=y,                        # follower minimizes y
...     lower_constraints=[x + y >= 3, y <= 2 * x],
... )
>>> bl.formulate(method="kkt", mpec_method="gdp")   # doctest: +SKIP
>>> result = m.solve()                              # doctest: +SKIP
"""

from __future__ import annotations

import numpy as np

from discopt.bilevel import kkt as _kkt
from discopt.bilevel import strong_duality as _sd
from discopt.bilevel.symbolic_diff import diff
from discopt.modeling.core import Constant, Constraint, Expression, Model, Variable, VarType
from discopt.mpec import reformulate_gdp, reformulate_sos1

__all__ = ["BilevelProblem"]


def _is_const(expr: Expression) -> bool:
    return isinstance(expr, Constant) and expr.value.ndim == 0


def _hessian_in_y(expr: Expression, ys: list[Variable]):
    """Constant Hessian ``∂²expr/∂y_j∂y_k`` if ``expr`` is (at most) quadratic in ``ys``.

    Uses the symbolic differentiator twice. Returns ``(H, None)`` with ``H`` an
    ``n×n`` numpy matrix when every second partial is a *constant* (so ``expr`` is
    affine or quadratic in ``ys`` with data-constant curvature); returns
    ``(None, witness)`` when some second partial still depends on a variable — i.e.
    ``expr`` is non-quadratic in ``y`` (or has a variable-dependent curvature such as
    ``x·y²``) and needs the full convexity certifier rather than a constant-Hessian
    test.
    """
    n = len(ys)
    H = np.zeros((n, n))
    grads = [diff(expr, yk) for yk in ys]
    for k, gk in enumerate(grads):
        for j in range(k, n):
            hjk = diff(gk, ys[j])
            if not _is_const(hjk):
                return None, ys[j]
            val = float(hjk.value)
            H[k, j] = val
            H[j, k] = val
    return H, None


def _convexity_status(expr: Expression, ys: list[Variable]):
    """Classify curvature of ``expr`` in ``ys``: affine / convex / nonconvex / nonquadratic."""
    H, witness = _hessian_in_y(expr, ys)
    if H is None:
        return "nonquadratic", witness
    if np.allclose(H, 0.0, atol=1e-12):
        return "affine", None
    if np.linalg.eigvalsh(H).min() >= -1e-9:  # PSD -> convex
        return "convex", None
    return "nonconvex", None


class BilevelProblem:
    """Optimistic bilevel program with a convex lower level (LP or convex-QP)."""

    def __init__(
        self,
        model: Model,
        *,
        upper_vars: list[Variable],
        lower_vars: list[Variable],
        lower_objective: Expression,
        lower_constraints: list[Constraint],
        lower_sense: str = "min",
        prefix: str = "bl",
        include_follower_bounds: bool = True,
    ):
        self.model = model
        self.upper_vars = list(upper_vars)
        self.lower_vars = list(lower_vars)
        self.lower_objective = lower_objective
        self.lower_constraints = list(lower_constraints)
        self.lower_sense = lower_sense
        self.prefix = prefix
        self.include_follower_bounds = include_follower_bounds
        self._formulated = False
        self.kkt: _kkt.KKTSystem | None = None
        self.strong_duality: _sd.StrongDualitySystem | None = None
        # The full lower constraint set actually reformulated (user constraints +
        # synthesized finite follower-variable bounds); populated in formulate().
        self.lower_constraints_full: list[Constraint] = list(lower_constraints)

        self._validate_inputs()

    # ── validation / gates ────────────────────────────────────────────

    def _validate_inputs(self) -> None:
        if self.lower_sense not in ("min", "max"):
            raise ValueError(f"lower_sense must be 'min' or 'max', got {self.lower_sense!r}")
        if not self.lower_vars:
            raise ValueError("bilevel problem needs at least one lower (follower) variable")

        for v in self.lower_vars:
            if not isinstance(v, Variable) or v.size != 1:
                raise NotImplementedError(
                    "Phase 1 requires scalar lower variables; pass array components "
                    "(e.g. y[i]) individually."
                )
            if v.var_type is not VarType.CONTINUOUS:
                # Integer follower: KKT / strong-duality do not characterize an
                # integer optimum (the follower value function is discontinuous).
                raise NotImplementedError(
                    f"lower variable '{v.name}' is {v.var_type.name}; the KKT "
                    f"reformulation is only valid for a continuous (convex) follower. "
                    f"Integer lower levels are out of scope (a value-function / "
                    f"branch-in-the-lower-level method is future work)."
                )

        upper_set = {id(v) for v in self.upper_vars}
        for v in self.lower_vars:
            if id(v) in upper_set:
                raise ValueError(f"variable '{v.name}' is in both upper_vars and lower_vars")

    def _gate_convexity(self) -> None:
        """Gate: the lower level must be **convex in y** (LP or convex-QP).

        KKT / strong-duality characterize follower optimality only when the lower
        problem is convex in the follower variables. We require the (sign-adjusted)
        objective and every inequality body to be convex in ``y`` (PSD constant
        Hessian), and every equality body to be affine in ``y`` (a nonlinear equality
        makes the lower feasible set nonconvex). A nonconvex lower level is refused
        outright; a non-quadratic curvature is refused pending the convexity
        certifier (rather than assuming convexity).
        """
        # Follower minimizes f (or maximizes f == minimizes -f): the *minimized*
        # objective must be convex in y.
        signed = self.lower_objective if self.lower_sense == "min" else -self.lower_objective
        self._require_convex(signed, "lower objective")
        for i, con in enumerate(self.lower_constraints):
            if con.sense == "==":
                self._require_affine(con.body, f"lower equality constraint {i}")
            else:
                self._require_convex(con.body, f"lower inequality constraint {i}")

    def _require_convex(self, expr: Expression, what: str) -> None:
        status, info = _convexity_status(expr, self.lower_vars)
        if status in ("affine", "convex"):
            return
        if status == "nonquadratic":
            raise NotImplementedError(
                f"{what} is nonlinear-but-not-quadratic in the follower variables "
                f"(witness '{info.name}'); the KKT reduction needs a certified-convex "
                f"lower level. LP and convex-QP lower levels are supported; general "
                f"convex-NLP lower levels await the convexity-certifier hook. Refusing "
                f"rather than assuming convexity."
            )
        raise NotImplementedError(  # nonconvex
            f"{what} is nonconvex in the follower variables (indefinite Hessian in y). "
            f"The KKT reduction is unsound for a nonconvex lower level; refusing."
        )

    def _require_affine(self, expr: Expression, what: str) -> None:
        status, _ = _convexity_status(expr, self.lower_vars)
        if status != "affine":
            raise NotImplementedError(
                f"{what} must be affine in the follower variables (a nonlinear equality "
                f"makes the lower feasible set nonconvex); got curvature '{status}'."
            )

    # ── follower variable bounds are follower constraints ──────────────

    _BIG = 1e19  # discopt's "unbounded" sentinel is 9.999e19; treat |b| >= 1e19 as ∞

    def _follower_bound_constraints(self) -> list[Constraint]:
        """Synthesize KKT constraints for each *finite* follower-variable bound.

        A follower's variable bounds are part of *its* feasible region, so they must
        enter the KKT system — otherwise the reformulation is wrong whenever the
        follower optimum sits on a bound. Emits ``lb - y <= 0`` and/or ``y - ub <= 0``
        for finite bounds (skipping the ±sentinel that means "unbounded").
        """
        cons: list[Constraint] = []
        for v in self.lower_vars:
            lb = float(v.lb)
            ub = float(v.ub)
            if lb > -self._BIG:
                cons.append(
                    Constraint(
                        Constant(lb) - v, sense="<=", rhs=0.0, name=f"{self.prefix}_{v.name}_lb"
                    )
                )
            if ub < self._BIG:
                cons.append(
                    Constraint(
                        v - Constant(ub), sense="<=", rhs=0.0, name=f"{self.prefix}_{v.name}_ub"
                    )
                )
        return cons

    # ── build ─────────────────────────────────────────────────────────

    def formulate(self, *, method: str = "kkt", mpec_method: str = "gdp") -> None:
        """Rewrite the model in place into a single-level MPEC.

        Parameters
        ----------
        method : {"kkt", "strong_duality"}
            The single-level reduction. ``"kkt"`` replaces the follower by its KKT
            conditions with per-row complementarity handed to :mod:`discopt.mpec`.
            ``"strong_duality"`` keeps stationarity/primal/dual feasibility but uses
            the single aggregate equality ``Σ μ_i g_i == 0`` (one bilinear equality,
            no disjunctive branching). Both are exact for a convex lower level and
            agree on the optimum. ``"pessimistic"`` is out of scope.
        mpec_method : {"gdp", "sos1"}
            For ``method="kkt"``: how the complementarity conditions are encoded for
            the **global** solve. ``"gdp"`` (disjunction) typically branches least;
            ``"sos1"`` uses a Special Ordered Set. (The local ``"scholtes"`` homotopy
            is driven at solve time via :func:`discopt.mpec.solve_mpec`.) Ignored for
            ``method="strong_duality"``.
        """
        if self._formulated:
            raise RuntimeError("formulate() has already been called on this BilevelProblem")
        if method == "pessimistic":
            raise NotImplementedError(
                "pessimistic bilevel (leader hedges against adversarial follower ties) "
                "is out of scope; the reduction is optimistic."
            )
        if method not in ("kkt", "strong_duality"):
            raise ValueError(f"unknown method {method!r}; use 'kkt' or 'strong_duality'")

        self._gate_convexity()

        # Follower variable bounds are follower constraints: fold finite ones in so
        # the KKT system is complete (user constraints first, so multipliers[0:k]
        # remain aligned with the user's lower_constraints).
        extra = self._follower_bound_constraints() if self.include_follower_bounds else []
        self.lower_constraints_full = list(self.lower_constraints) + extra

        if method == "kkt":
            if mpec_method not in ("gdp", "sos1"):
                raise ValueError(
                    f"mpec_method must be 'gdp' or 'sos1' for a global certificate, got "
                    f"{mpec_method!r}. (For the local Scholtes homotopy, call "
                    f"discopt.mpec.solve_mpec on the formulated pairs.)"
                )
            self.kkt = _kkt.build_kkt(
                self.model,
                lower_vars=self.lower_vars,
                lower_objective=self.lower_objective,
                lower_constraints=self.lower_constraints_full,
                lower_sense=self.lower_sense,
                prefix=self.prefix,
            )
            if self.kkt.comp_pairs:
                if mpec_method == "gdp":
                    reformulate_gdp(self.model, self.kkt.comp_pairs)
                else:
                    reformulate_sos1(self.model, self.kkt.comp_pairs)
        else:  # strong_duality
            self.strong_duality = _sd.build_strong_duality(
                self.model,
                lower_vars=self.lower_vars,
                lower_objective=self.lower_objective,
                lower_constraints=self.lower_constraints_full,
                lower_sense=self.lower_sense,
                prefix=self.prefix,
            )

        self._formulated = True

    def solve(self, **kwargs):
        """Formulate (if needed) and solve the resulting single-level MPEC.

        Thin wrapper over :meth:`Model.solve`; the global optimality certificate is
        the MPEC's (see :mod:`discopt.mpec`).
        """
        if not self._formulated:
            self.formulate()
        return self.model.solve(**kwargs)

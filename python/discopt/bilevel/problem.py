"""Bilevel problem front-end.

``BilevelProblem`` mirrors the :class:`~discopt.ro.counterpart.RobustCounterpart`
builder: the user builds a normal :class:`~discopt.modeling.core.Model` holding
the **leader's** objective and constraints, describes the follower separately, and
calls :meth:`~BilevelProblem.formulate`, which rewrites the model in place into a
single-level MPEC that ``model.solve()`` handles.

Phase 1 scope (``docs/dev/bilevel-module-plan.md`` §1): **optimistic** bilevel with
a lower level that is **affine in the follower variables** (an LP in ``y``), via the
KKT reformulation (:mod:`~discopt.bilevel.kkt`). Everything the reduction is unsound
for is refused loudly rather than silently approximated:

* integer follower variables — KKT does not characterize integer optima;
* a lower level that is *nonlinear in* ``y`` — Phase 1 handles the LP case; convex
  QP/NLP lower levels are Phase 2 (the gate names the offending expression);
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

from discopt.bilevel import kkt as _kkt
from discopt.bilevel.symbolic_diff import diff
from discopt.modeling.core import Constant, Constraint, Expression, Model, Variable, VarType
from discopt.mpec import reformulate_gdp, reformulate_sos1

__all__ = ["BilevelProblem"]


def _is_structural_zero(expr: Expression) -> bool:
    return isinstance(expr, Constant) and expr.value.ndim == 0 and float(expr.value) == 0.0


def _is_affine_in(expr: Expression, variables: list[Variable]) -> Variable | None:
    """Return a variable witnessing nonlinearity in ``variables``, or None if affine.

    ``expr`` is affine in ``variables`` iff ``∂expr/∂y_k`` is independent of every
    ``y_j`` — i.e. the second partials ``∂²expr/∂y_j∂y_k`` all vanish. Uses the
    symbolic differentiator, so bilinear coupling ``x·y`` (a *coefficient* ``x`` on
    ``y``) is correctly recognized as affine in ``y``.
    """
    for yk in variables:
        d = diff(expr, yk)
        for yj in variables:
            if not _is_structural_zero(diff(d, yj)):
                return yk
    return None


class BilevelProblem:
    """Optimistic bilevel program with a convex (Phase 1: LP) lower level."""

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
    ):
        self.model = model
        self.upper_vars = list(upper_vars)
        self.lower_vars = list(lower_vars)
        self.lower_objective = lower_objective
        self.lower_constraints = list(lower_constraints)
        self.lower_sense = lower_sense
        self.prefix = prefix
        self._formulated = False
        self.kkt: _kkt.KKTSystem | None = None

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
        """Phase 1 LP gate: lower objective + constraints must be affine in ``y``."""
        witness = _is_affine_in(self.lower_objective, self.lower_vars)
        if witness is not None:
            raise NotImplementedError(
                f"lower objective is nonlinear in follower variable '{witness.name}'. "
                f"Phase 1 handles LP (affine-in-y) lower levels; convex QP/NLP lower "
                f"levels are Phase 2. (A nonconvex lower level would make the KKT "
                f"reduction unsound and is refused outright.)"
            )
        for i, con in enumerate(self.lower_constraints):
            witness = _is_affine_in(con.body, self.lower_vars)
            if witness is not None:
                raise NotImplementedError(
                    f"lower constraint {i} is nonlinear in follower variable "
                    f"'{witness.name}'. Phase 1 handles affine-in-y (LP) lower levels."
                )

    # ── build ─────────────────────────────────────────────────────────

    def formulate(self, *, method: str = "kkt", mpec_method: str = "gdp") -> None:
        """Rewrite the model in place into a single-level MPEC.

        Parameters
        ----------
        method : {"kkt"}
            The single-level reduction. ``"kkt"`` replaces the follower by its KKT
            conditions (Phase 1). ``"strong_duality"`` (LP strong-duality, no
            complementarity) is Phase 2. ``"pessimistic"`` is out of scope.
        mpec_method : {"gdp", "sos1"}
            How the complementarity conditions are encoded for the **global**
            solve. ``"gdp"`` (disjunction) typically branches least; ``"sos1"`` uses
            a Special Ordered Set. (The local ``"scholtes"`` homotopy is driven at
            solve time via :func:`discopt.mpec.solve_mpec`, not here.)
        """
        if self._formulated:
            raise RuntimeError("formulate() has already been called on this BilevelProblem")
        if method == "pessimistic":
            raise NotImplementedError(
                "pessimistic bilevel (leader hedges against adversarial follower ties) "
                "is out of scope; Phase 1 is optimistic."
            )
        if method == "strong_duality":
            raise NotImplementedError(
                "the LP strong-duality reformulation is Phase 2; use method='kkt'."
            )
        if method != "kkt":
            raise ValueError(f"unknown method {method!r}; use 'kkt'")
        if mpec_method not in ("gdp", "sos1"):
            raise ValueError(
                f"mpec_method must be 'gdp' or 'sos1' for a global certificate, got "
                f"{mpec_method!r}. (For the local Scholtes homotopy, call "
                f"discopt.mpec.solve_mpec on the formulated pairs.)"
            )

        self._gate_convexity()

        self.kkt = _kkt.build_kkt(
            self.model,
            lower_vars=self.lower_vars,
            lower_objective=self.lower_objective,
            lower_constraints=self.lower_constraints,
            lower_sense=self.lower_sense,
            prefix=self.prefix,
        )

        if self.kkt.comp_pairs:
            if mpec_method == "gdp":
                reformulate_gdp(self.model, self.kkt.comp_pairs)
            else:
                reformulate_sos1(self.model, self.kkt.comp_pairs)

        self._formulated = True

    def solve(self, **kwargs):
        """Formulate (if needed) and solve the resulting single-level MPEC.

        Thin wrapper over :meth:`Model.solve`; the global optimality certificate is
        the MPEC's (see :mod:`discopt.mpec`).
        """
        if not self._formulated:
            self.formulate()
        return self.model.solve(**kwargs)

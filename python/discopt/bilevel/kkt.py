"""KKT reformulation of a convex lower level into a single-level MPEC.

Phase 1 of the bilevel module (``docs/dev/bilevel-module-plan.md`` §4, KKT path).
Given a lower level that is **convex in the follower variables** ``y`` (Phase 1:
affine, i.e. an LP in ``y``),

    min_y  f(x, y)   s.t.   g_i(x, y) <= 0   (i in I),   h_j(x, y) == 0   (j in E)

its KKT conditions are *necessary and sufficient* for follower optimality, so the
follower can be replaced by:

* **stationarity**   ``∇_y L == 0``      with ``L = f + Σ_i μ_i g_i + Σ_j ν_j h_j``
* **primal feas.**   ``g_i <= 0``, ``h_j == 0``
* **dual feas.**     ``μ_i >= 0``        (``ν_j`` free)
* **complementarity** ``0 <= μ_i ⊥ -g_i >= 0``

The complementarity conditions are handed to :mod:`discopt.mpec` (GDP / SOS1 /
Scholtes). Stationarity is emitted as ordinary model constraints using the
:mod:`~discopt.bilevel.symbolic_diff` engine — that is what keeps the whole
reformulation inside discopt's certified global path (a numeric gradient wrapped
in a ``CustomCall`` would forfeit the certificate and reject integers).

For a follower that *maximizes* ``f``, ``L`` uses ``-f`` (``max f`` ≡ ``min -f``).
"""

from __future__ import annotations

from dataclasses import dataclass

from discopt.bilevel.symbolic_diff import diff
from discopt.modeling.core import Constraint, Expression, Model, Variable
from discopt.mpec import Complementarity, complementarity


@dataclass
class KKTSystem:
    """The pieces emitted by :func:`build_kkt`, kept for inspection/testing.

    ``multipliers`` are aligned 1:1 with ``lower_constraints``; ``comp_pairs`` are
    the complementarity conditions for the *inequality* rows only (equalities carry
    a free multiplier and no complementarity).
    """

    multipliers: list[Variable]
    comp_pairs: list[Complementarity]
    stationarity: list[Constraint]
    primal: list[Constraint]
    lagrangian: Expression


def build_kkt(
    model: Model,
    *,
    lower_vars: list[Variable],
    lower_objective: Expression,
    lower_constraints: list[Constraint],
    lower_sense: str = "min",
    prefix: str = "bl",
) -> KKTSystem:
    """Emit the KKT conditions of the lower level onto ``model`` (in place).

    Adds one multiplier variable per lower constraint, the stationarity equalities
    (one per lower variable), the primal-feasibility constraints, and returns the
    complementarity pairs for the caller to hand to :mod:`discopt.mpec`. The
    model's objective (the *leader's*) is not touched.
    """
    # Follower minimizes f; a maximizing follower contributes -f to L.
    if lower_sense == "min":
        f_signed: Expression = lower_objective
    elif lower_sense == "max":
        f_signed = -lower_objective
    else:
        raise ValueError(f"lower_sense must be 'min' or 'max', got {lower_sense!r}")

    multipliers: list[Variable] = []
    comp_pairs: list[Complementarity] = []
    lagrangian: Expression = f_signed

    for i, con in enumerate(lower_constraints):
        body = con.body  # normalized: body <= 0 for '<=', body == 0 for '=='
        if con.sense == "<=":
            mu = model.continuous(f"{prefix}_mu{i}", lb=0.0)  # μ_i >= 0
            comp_pairs.append(
                # 0 <= μ_i ⊥ (-g_i) >= 0 :  μ_i >= 0, -g_i >= 0, μ_i·g_i = 0
                complementarity(mu, -body, name=f"{prefix}_compl{i}")
            )
        elif con.sense == "==":
            mu = model.continuous(f"{prefix}_nu{i}")  # free multiplier (default bounds)
        else:
            raise ValueError(
                f"lower constraint {i} has unsupported sense {con.sense!r}; expected '<=' or '=='."
            )
        multipliers.append(mu)
        lagrangian = lagrangian + mu * body

    # Stationarity: ∂L/∂y_k == 0 for each (scalar) lower variable.
    stationarity: list[Constraint] = []
    for yk in lower_vars:
        d = diff(lagrangian, yk)
        c = Constraint(body=d, sense="==", rhs=0.0, name=f"{prefix}_stat_{yk.name}")
        model.subject_to(c, name=c.name)
        stationarity.append(c)

    # Primal feasibility: re-assert every lower constraint on the model. (For the
    # inequality rows the complementarity reformulation also emits -g_i >= 0; the
    # duplicate is redundant, not unsound. Correctness before economy — Phase 2 can
    # dedupe.)
    primal: list[Constraint] = []
    for i, con in enumerate(lower_constraints):
        model.subject_to(con, name=f"{prefix}_lower_primal_{i}")
        primal.append(con)

    return KKTSystem(
        multipliers=multipliers,
        comp_pairs=comp_pairs,
        stationarity=stationarity,
        primal=primal,
        lagrangian=lagrangian,
    )

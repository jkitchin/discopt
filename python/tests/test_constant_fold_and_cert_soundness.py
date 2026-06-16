"""Soundness regressions for two coupled relaxation/certification fixes.

1. **Constant-factor folding in the product linearizer.** A product factor that
   is a *variable-free* subexpression — e.g. ``neg(2.5)`` (the unary negation of
   a literal, how ``(-2.5)`` is parsed) or ``(-3) * (-3)`` from an expanded
   square — was not recognized as a constant. The term ``(-2.5) * x`` (an exact
   linear term!) was therefore sent to bilinear decomposition, which failed with
   ``Cannot decompose product: (neg(3) * neg(3))``, and the *entire constraint*
   was dropped from the MILP relaxation. With an objective-defining constraint
   omitted, the dual bound could not rise to the incumbent and the gap never
   closed. This is exactly how MINLPLib's ``nvs08`` (whose objective equation
   contains ``sqr((-3) + i1)``, expanding to ``(-3) * (-3)``) failed to certify:
   bound stuck at ~16.1 against the true optimum 23.4497. Folding such factors
   is exact, so it only tightens the relaxation while staying sound
   (``bound <= optimum``).

2. **Certification invariant.** A merely-*feasible* exit (search not closed —
   budget exhausted with an open gap) must never be reported as
   ``gap_certified=True``. The no-incumbent resource-limit branch cleared the
   flag, but a feasible exit *with* an incumbent had its own path and could
   retain a stale ``gap_certified=True`` — a false certification of a
   non-optimal point. The invariant now holds: ``status != "optimal"`` implies
   ``gap_certified is False``.

Both guards are deterministic and self-contained: the constant-fold case builds
the exact ``neg(3) * neg(3)`` AST the GAMS tokenizer emits (Python's own
``-3 * -3`` would fold to a literal and never exercise the bug), and the
certification case stops B&B after the root node with ``max_nodes=1``.
"""

import os

os.environ["JAX_PLATFORMS"] = "cpu"
os.environ["JAX_ENABLE_X64"] = "1"

import discopt.modeling as dm
import pytest
from discopt.modeling.core import BinaryOp, Constant, UnaryOp, sqrt


def test_eval_constant_expr_folds_negated_literals():
    """A variable-free factor must fold to its value, not reach bilinear decomp."""
    from discopt._jax.milp_relaxation import _eval_constant_expr

    # neg(2.5) folds to -2.5 (the exact case that broke the linearizer: a
    # ``(-2.5)`` literal parses as the unary negation of a Constant).
    assert _eval_constant_expr(UnaryOp("neg", Constant(2.5))) == -2.5
    # (-3) * (-3), produced when an expanded square's constant term is built
    # from two negated literals, folds to 9.
    prod = BinaryOp("*", UnaryOp("neg", Constant(3.0)), UnaryOp("neg", Constant(3.0)))
    assert _eval_constant_expr(prod) == 9.0


def test_eval_constant_expr_returns_none_for_variable_terms():
    """The folder must never mis-fold an expression that depends on a variable."""
    from discopt._jax.milp_relaxation import _eval_constant_expr

    m = dm.Model("t")
    x = m.continuous("x", lb=0, ub=10)
    assert _eval_constant_expr(-x) is None
    assert _eval_constant_expr(2.0 * x) is None


@pytest.mark.correctness
def test_negated_constant_product_keeps_constraint_and_certifies():
    """A nonlinear constraint carrying a ``(-3)*(-3)`` constant term must be kept.

    This mirrors MINLPLib ``nvs08``'s objective equation, where expanding
    ``sqr((-3) + i1)`` produces the literal product ``(-3) * (-3)``. The
    surrounding ``x*y`` term forces the constraint through product
    decomposition, where the constant factor used to raise
    ``Cannot decompose product: (neg(3) * neg(3))`` and the whole constraint
    was dropped — leaving the relaxation unable to certify the optimum.
    """
    m = dm.Model("negconst_product")
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    # Build the exact AST a tokenizer emits for ``(-3) * (-3)`` (== 9.0).
    nine = BinaryOp("*", UnaryOp("neg", Constant(3.0)), UnaryOp("neg", Constant(3.0)))
    # x*y + 9 <= 13  =>  x*y <= 4. Active at the optimum, so dropping it would
    # loosen the bound from x+y<=10 down to the real best 5.8.
    m.subject_to(x * y + nine <= 13)
    m.maximize(x + y)
    r = m.solve(time_limit=30, gap_tolerance=1e-4)

    assert r.status == "optimal", f"status={r.status} (constraint dropped → uncertified before fix)"
    assert r.objective is not None and abs(r.objective - 5.8) <= 1e-3, f"obj={r.objective}"
    # Soundness invariant: a valid dual bound never exceeds the optimum.
    assert r.bound is not None, "certified solve must report a dual bound"
    assert r.bound >= r.objective - 1e-3, f"dual bound {r.bound} below maximized obj {r.objective}"


@pytest.mark.correctness
def test_feasible_exit_is_never_certified():
    """A budget-capped feasible exit with an incumbent must not be gap_certified.

    ``max_nodes=1`` deterministically stops B&B after the root node: the root
    subsolve yields a feasible incumbent but the gap is wide open, so the search
    is not closed. This is precisely the path that used to retain a stale
    ``gap_certified=True``, falsely certifying a non-optimal point.
    """
    m = dm.Model("mini_minlp")
    i1 = m.integer("i1", lb=0, ub=200)
    i2 = m.integer("i2", lb=0, ub=200)
    x3 = m.continuous("x3", lb=0.001, ub=200)
    m.subject_to(sqrt(x3) + i1 + 2 * i2 >= 10)
    m.subject_to(i2**2 - 4 * i1 >= -12)
    m.minimize((i1 - 3) ** 2 + (i2 - 2) ** 2 + (x3 + 4) ** 2)
    r = m.solve(time_limit=30, max_nodes=1)

    # The core invariant: only an 'optimal' status may be certified.
    if r.status != "optimal":
        assert r.gap_certified is False, (
            f"status={r.status} but gap_certified={r.gap_certified} (false certification)"
        )
    # And whenever a solve IS certified, the bound must be sound.
    if r.gap_certified:
        assert r.bound is not None and r.objective is not None
        assert r.bound <= r.objective + 1e-3

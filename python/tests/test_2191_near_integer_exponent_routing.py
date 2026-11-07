"""#2191 class, on the routing side: a near-integer exponent is not a monomial.

``ExprArena::max_degree`` (Rust) read an exponent within 1e-12 of a nonnegative
integer as that integer, so ``x**(1 + 2**-43)`` made a model ``LP`` and
``x**(2 + 2**-42)`` made it ``QP``. The LP tape rung then took the objective's
gradient at the origin as ``c`` -- with no check that the objective is affine --
and ``minimize x - 2*x**(1 + 2**-43)`` on ``[0, 8]`` returned ``optimal`` 0.0
with ``gap_certified=True``; the optimum is -8 at ``x = 8``. (The power term is
0 at the returned point, so objective reconciliation saw no discrepancy.)

Fixed at the root (exact exponent tests in ``max_degree``, ``term_classifier``,
presolve ``polynomial``, FBBT ``interval_pow`` and the Python QP extractor) and
backstopped: both tape rungs now refuse an objective their LP/QP model does not
reproduce (:class:`ClassificationMismatchError`).
"""

from __future__ import annotations

import discopt.modeling as dm
import pytest
from discopt._relax import problem_classifier as pc
from discopt._relax.problem_classifier import ProblemClass, classify_problem


@pytest.mark.smoke
@pytest.mark.parametrize(
    "p", [1.0 + 2.0**-43, 1.0 - 2.0**-43, 2.0 + 2.0**-42, 2.0 - 2.0**-42, 2.0**-50]
)
def test_near_integer_exponent_is_not_lp_or_qp(p):
    m = dm.Model("near_int")
    x = m.continuous("x", lb=1.0, ub=8.0)
    m.minimize(x + x**p)
    cls = classify_problem(m)
    assert cls not in (ProblemClass.LP, ProblemClass.QP, ProblemClass.MILP), cls


@pytest.mark.smoke
@pytest.mark.parametrize("p,expect", [(1.0, ProblemClass.LP), (2.0, ProblemClass.QP)])
def test_exact_integer_exponent_keeps_class(p, expect):
    m = dm.Model("exact_int")
    x = m.continuous("x", lb=1.0, ub=8.0)
    m.minimize(x + x**p)
    assert classify_problem(m) == expect


@pytest.mark.smoke
def test_dropped_power_term_no_false_certificate():
    """The reproducer: was ``optimal`` 0.0 / ``gap_certified=True``; optimum is -8."""
    p = 1.0 + 2.0**-43
    m = dm.Model("dropped_power")
    x = m.continuous("x", lb=0.0, ub=8.0)
    m.minimize(x - 2.0 * x**p)
    r = m.solve(time_limit=30)
    true_opt = 8.0 - 2.0 * 8.0**p
    if r.bound is not None:
        assert r.bound <= true_opt + 1e-5, (r.status, r.bound)
    assert r.objective == pytest.approx(true_opt, abs=1e-4)


@pytest.mark.smoke
def test_lp_tape_rung_refuses_nonaffine_objective():
    """Backstop: called on a misclassified model, the LP tape rung refuses."""
    m = dm.Model("lp_backstop")
    x = m.continuous("x", lb=0.0, ub=8.0)
    m.minimize(x - 2.0 * x ** (1.0 + 2.0**-43))
    with pytest.raises(pc.ClassificationMismatchError):
        pc._extract_lp_data_tape(m)


@pytest.mark.smoke
def test_qp_tape_rung_refuses_nonquadratic_objective():
    m = dm.Model("qp_backstop")
    x = m.continuous("x", lb=1.0, ub=8.0)
    m.minimize(2.0**46 * x ** (2.0 + 2.0**-42) - 2.0**46 * x**2 + x)
    with pytest.raises(pc.ClassificationMismatchError):
        pc._qp_terms_tape(m, 1)


@pytest.mark.smoke
def test_tape_gates_accept_genuine_lp_and_qp_with_large_cancelling_terms():
    """The gate's tolerance scales with the term magnitudes, not the cancelled value."""
    m = dm.Model("lp_ok")
    x = m.continuous("x", lb=0.0, ub=8.0)
    y = m.continuous("y", lb=0.0, ub=8.0)
    m.subject_to(x + y <= 10.0)
    m.minimize(1e9 * x - 1e9 * y + 0.5 * x)
    lp = pc._extract_lp_data_tape(m)
    assert lp is not None
    assert lp.c[0] == pytest.approx(1e9 + 0.5) and lp.c[1] == pytest.approx(-1e9)

    m2 = dm.Model("qp_ok")
    u = m2.continuous("u", lb=-3.0, ub=3.0)
    v = m2.continuous("v", lb=-3.0, ub=3.0)
    m2.minimize(1e8 * u**2 - 1e8 * u * v + 3.0 * v**2 + u)
    Q, c, _ = pc._qp_terms_tape(m2, 2)
    assert Q[0, 0] == pytest.approx(2e8) and Q[0, 1] == pytest.approx(-1e8)


@pytest.mark.smoke
def test_qp_algebraic_extractor_refuses_near_integer_exponent():
    m = dm.Model("qp_alg")
    x = m.continuous("x", lb=1.0, ub=8.0)
    m.minimize(x ** (2.0 + 2.0**-42))
    with pytest.raises(pc._EXTRACTION_DECLINES):
        pc.extract_qp_data_algebraic(m)

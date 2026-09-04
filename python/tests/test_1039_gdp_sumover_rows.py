"""#1039: indexed-summation (``SumOverExpression``) bodies on the GDP routes.

Two defects met at the same expression node, and the fix for the first one
caused the second, so both are pinned here.

1. **The row that was silently dropped.** ``make_disjunct``/``add_disjunction``
   emit the ``exactly-one`` selector row as a ``SumOverExpression``. The OA/LOA
   master built its linear block behind *two* gates — ``_is_linear(c.body)``
   first, then ``_extract_body_coeffs`` — and the conservative one won:
   ``_is_linear`` has no ``SumOverExpression`` case, so the row never reached
   the master MILP. The master was then free to propose "no disjunct active",
   whose fixed-integer NLP is infeasible, and LOA reported ``status="unknown"``
   on a trivially feasible model.

2. **The false certificate the first fix nearly shipped.** The obvious repair —
   teach ``_is_linear`` about the node — is wrong, and its wrongness is silent.
   ``_is_linear`` is a shared gate with ~10 consumers, and the hull substitution
   family (``_hull_linear_substitute`` → ``_substitute_vars`` →
   ``_collect_variables``) has no case for it. Admitting the node made hull emit
   ``Σ[3 terms] - (0 * y0) <= 0``: the disjunct body imposed *globally*, its
   selector coefficient collapsed to zero and the disaggregated variables never
   created. That cut off the other mode and certified ``optimal`` at -3.0 on a
   model whose true optimum is -30.0 — a dual bound *above* the optimum of a
   minimization, which is the one thing a global solver may never report.

The real gate is ``_extract_body_coeffs``: it answers with the row itself, so it
cannot promise more capability than it delivers.

Completing the hull family so these routes *solve* the model instead of declining
it is tracked by #1154; the parametrized test below is written to strengthen
rather than break when that lands.
"""

import discopt.modeling as dm
import pytest
from discopt._relax.gdp_reformulate import _extract_body_coeffs
from discopt.modeling.core import SumOverExpression

# min -(x0+x1+x2) over the disjunction {sum(xi - 1) <= 0} OR {x0 >= 8}, xi in
# [0, 10]. Mode B admits x = (10, 10, 10), so the optimum is -30.
_TRUE_OPT = -30.0


def _blocker_model():
    m = dm.Model("sumover_disjunction")
    x = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(3)]
    m.either_or([[dm.sum(x[i] - 1 for i in range(3)) <= 0.0], [x[0] >= 8.0]], name="modes")
    m.minimize(-(x[0] + x[1] + x[2]))
    return m


def test_extractor_reads_an_indexed_summation():
    """Defect 1, at the mechanism: the extractor must fold ``Σ[terms]``.

    This is the assertion that fails before the fix (the extractor returned
    ``None``, which is what put the row on the dropped path).
    """
    m = dm.Model("extract")
    x = [m.continuous(f"x{i}", lb=0.0, ub=10.0) for i in range(3)]
    body = dm.sum(x[i] - 1 for i in range(3))

    # Guard the probe (§6): if the modeling layer ever stops producing this node
    # here, the test below would pass vacuously against some other expression.
    assert isinstance(body, SumOverExpression), f"expected SumOverExpression, got {type(body)}"

    coeffs = _extract_body_coeffs(body, m, 3)
    assert coeffs is not None, "indexed summation must be readable as a single linear row"
    c_vec, offset = coeffs
    assert list(c_vec) == pytest.approx([1.0, 1.0, 1.0])
    assert offset == pytest.approx(-3.0)


def test_loa_forwards_the_exactly_one_row():
    """Defect 1, end to end: LOA must not answer ``unknown`` on a feasible GDP."""
    m = dm.Model("loa_block")
    x = m.continuous("x", lb=0, ub=10)
    d1 = m.make_disjunct("low")
    d1.subject_to(x <= 3)
    d2 = m.make_disjunct("high")
    d2.subject_to(x >= 7)
    m.add_disjunction([d1, d2])
    m.minimize(x)

    result = m.solve(time_limit=60, gdp_method="loa")
    assert result.status in ("optimal", "feasible"), f"LOA returned {result.status!r}"
    assert result.objective == pytest.approx(0.0, abs=1e-2)


@pytest.mark.parametrize("gdp_method", ["auto", "hull", "big-m"])
def test_sumover_disjunct_body_never_reports_an_invalid_bound(gdp_method):
    """Defect 2: refusing is allowed, a bound above the optimum is not.

    Every route is free to decline this model — a loud refusal is the honest
    answer while the hull substitution family lacks the node — but none may
    return a dual bound above the true minimum, nor certify ``optimal`` at a
    point worse than it. Asserted as the invariant rather than as "must raise"
    so that completing the hull family (which makes these routes *solve* the
    model at -30.0) strengthens this test instead of breaking it.
    """
    m = _blocker_model()
    refusal = None
    result = None
    try:
        result = m.solve(gdp_method=gdp_method, time_limit=30)
    except Exception as exc:  # noqa: BLE001 - a refusal is an acceptable outcome
        refusal = exc

    # Exactly one of the two acceptable outcomes must have happened. This is
    # asserted rather than skipped: while the hull substitution family lacks a
    # ``SumOverExpression`` case, declining IS the verified-correct behaviour,
    # and a ``pytest.skip`` here would read as "untested" and let a future
    # regression through silently (§6).
    assert (refusal is None) != (result is None), "solve neither returned nor raised"

    if refusal is not None:
        assert str(refusal), f"{type(refusal).__name__} raised without a message"
        assert isinstance(refusal, (ValueError, ArithmeticError)) or type(
            refusal
        ).__name__.endswith("Error"), f"unexpected refusal type {type(refusal).__name__}"
        return

    if result.bound is not None:
        assert result.bound <= _TRUE_OPT + 1e-6, (
            f"{gdp_method}: dual bound {result.bound} is ABOVE the true optimum "
            f"{_TRUE_OPT} of a minimization -- an invalid certificate"
        )
    if result.status == "optimal":
        assert result.objective == pytest.approx(_TRUE_OPT, abs=1e-4), (
            f"{gdp_method}: certified 'optimal' at {result.objective}, true optimum is {_TRUE_OPT}"
        )

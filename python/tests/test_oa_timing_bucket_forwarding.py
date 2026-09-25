"""Guard: OA's feasibility evaluators must forward ``timing_bucket`` (issue #74).

``nlp_ipopt``'s ``_charge_evaluator`` reads ``evaluator.timing_bucket`` to decide
which layer to charge derivative-callback time to. Its "undeclared" arm exists
for *duck-typed evaluators from outside the package* — the docstring at
``nlp_ipopt.py:130`` says so — and it does two things: it logs a warning, and it
leaves the callback time with the enclosing solver region, so the layer profile
over-reports that layer by exactly the restoration cost.

``solver.py``'s cut-augmented proxy forwards the attribute explicitly for this
reason (``solver.py:2301``, issue #74). ``oa.py``'s two feasibility evaluators
were written later and did not, so every OA feasibility subproblem fell to that
arm: an in-tree wrapper tripping a warning documented as impossible for in-tree
code, and a measurably wrong timing profile.

Found by running an MINLP for the browser demo page and reading its output:

    Evaluator _ElasticFeasibilityEvaluator declares no `timing_bucket`; its
    derivative-callback time will be left with the enclosing solver region and
    the layer profile will over-report that layer [timing-bucket-unknown].

Both classes are checked directly (the attribute contract) and through a real
solve (no warning escapes), because a wrapper added later would pass the first
and fail the second.
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.solvers.oa import _ElasticFeasibilityEvaluator, _FeasibilityEvaluator

pytestmark = pytest.mark.smoke


def _real_evaluator():
    """A genuine evaluator, not a stub.

    ``_ElasticFeasibilityEvaluator`` calls ``_infer_constraint_bounds``, which
    reads private evaluator internals (``_source_constraints``,
    ``_constraint_flat_sizes``). A hand-rolled stub would have to mirror those
    and would drift; wrapping the real thing tests the contract that actually
    ships.
    """
    from discopt._tape_nlp_evaluator import make_evaluator

    m = dm.Model("inner")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=1.0)
    m.minimize(x[0] + x[1])
    m.subject_to(x[0] * x[0] + x[1] * x[1] <= 1.0, name="disk")
    return make_evaluator(m)


@pytest.mark.parametrize("cls", [_FeasibilityEvaluator, _ElasticFeasibilityEvaluator])
@pytest.mark.parametrize("norm", ["L1", "L2", "L_infinity"])
def test_feasibility_evaluators_forward_timing_bucket(cls, norm) -> None:
    inner = _real_evaluator()
    # The wrapped evaluator must declare a bucket, or "the wrapper forwards it"
    # is a statement about nothing.
    assert getattr(inner, "timing_bucket", None) is not None, (
        "the in-tree evaluator stopped declaring timing_bucket -- this test's "
        "premise is gone, not just its assertion"
    )
    wrapper = cls(inner, np.array([0.0, 0.0]), np.array([1.0, 1.0]), norm)
    assert wrapper.timing_bucket == inner.timing_bucket, (
        f"{cls.__name__} does not forward timing_bucket; "
        "nlp_ipopt will charge its callbacks to the enclosing region"
    )


def test_minlp_solve_emits_no_undeclared_bucket_warning(caplog) -> None:
    """The end-to-end symptom: a plain MINLP solve must not log the warning."""
    m = dm.Model("reactor network")
    build = m.binary("build", shape=(3,))
    load = m.continuous("load", shape=(3,), lb=0.0, ub=1.0)
    fixed = [55.0, 38.0, 72.0]
    rate = [90.0, 65.0, 120.0]
    opcost = [40.0, 33.0, 46.0]
    m.minimize(
        dm.sum([fixed[i] * build[i] for i in range(3)])
        + dm.sum([opcost[i] * load[i] * load[i] for i in range(3)])
    )
    m.subject_to(
        dm.sum([rate[i] * load[i] * (1 - 0.25 * load[i]) for i in range(3)]) >= 150.0,
        name="demand",
    )
    for i in range(3):
        m.subject_to(load[i] <= build[i], name=f"link{i}")

    with caplog.at_level(logging.WARNING, logger="discopt.solvers.nlp_ipopt"):
        result = m.solve()

    # The model has to actually reach the OA feasibility path, or this passes
    # vacuously (CLAUDE.md, "Measurement & instrumentation discipline" §6).
    assert result.status == "optimal", result.status
    offenders = [
        r.getMessage() for r in caplog.records if "timing-bucket-unknown" in r.getMessage()
    ]
    assert not offenders, "in-tree evaluator hit the undeclared-bucket arm:\n  " + "\n  ".join(
        offenders
    )

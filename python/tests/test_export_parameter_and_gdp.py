"""Export fidelity for ``Parameter`` bodies and GDP refusal.

Both behaviours were found by an export matrix over the construct families in
``discopt_benchmarks/scripts/issue1215_expressiveness_audit.py``: the modeling
language could express more than the writers could write, so a model that solved
fine was silently un-portable to SCIP / BARON / Couenne / GAMS.

Before the fix:

* ``Parameter`` raised ``ValueError: Cannot write expression type to .nl:
  Parameter``. Three separate writers (``nl.py``, ``gams.py``, ``_extract.py``)
  each independently omitted it.
* A model still holding GDP rows died with ``AttributeError:
  '_DisjunctiveConstraint' object has no attribute 'rhs'`` from inside the
  writer's row loop -- an internal error rather than an answer.

The ``Parameter`` tests assert **round-trip fidelity**, not merely that bytes
were written: a writer that emits a wrong model is worse than one that refuses.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling import from_nl

pytestmark = pytest.mark.smoke


def _model(pval):
    """``min x0+x1+x2  s.t.  p . x >= 6``, so the optimum is a function of ``p``.

    Deliberately does NOT put ``p`` on both the objective and the constraint: a
    first draft used ``min p*x0 ... s.t. p*x0 ... >= 6``, whose optimum is 6 for
    every ``p``, and so could not have detected a writer that froze the value.
    """
    m = dm.Model("param_export")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=5.0)
    p = m.parameter("p", value=pval)
    if np.asarray(pval).ndim == 0:
        m.subject_to(p * x[0] + x[1] + x[2] >= 6.0, name="c1")
    else:
        m.subject_to(dm.sum([p[i] * x[i] for i in range(3)]) >= 6.0, name="c1")
    m.minimize(x[0] + x[1] + x[2])
    return m


@pytest.mark.parametrize(
    "pval",
    [2.0, 5.0, np.array([1.0, 2.0, 3.0])],
    ids=["scalar", "scalar-other-value", "shaped-indexed"],
)
def test_parameter_nl_export_round_trips(pval, tmp_path):
    """Re-importing the exported ``.nl`` must reproduce the same optimum."""
    m = _model(pval)
    want = m.solve(time_limit=60)
    assert want.status == "optimal", f"reference model not optimal: {want.status}"

    path = tmp_path / "m.nl"
    m.to_nl(str(path))
    assert path.stat().st_size > 0, "wrote an empty .nl file"

    got = from_nl(str(path)).solve(time_limit=60)
    assert got.status == want.status
    assert got.objective == pytest.approx(want.objective, abs=1e-6)


def test_parameter_export_tracks_the_current_value(tmp_path):
    """Export is a snapshot at the current value, not frozen at declaration."""
    obj = {}
    for pval in (2.0, 5.0):
        m = _model(pval)
        path = tmp_path / f"m_{pval}.nl"
        m.to_nl(str(path))
        obj[pval] = from_nl(str(path)).solve(time_limit=60).objective
    # min sum(x) s.t. p*x0 + x1 + x2 >= 6 -> 6/p for p >= 1.
    assert obj[2.0] == pytest.approx(3.0, abs=1e-5)
    assert obj[5.0] == pytest.approx(1.2, abs=1e-5)
    assert obj[2.0] != pytest.approx(obj[5.0], abs=1e-3), (
        "the exported file did not track Parameter.value"
    )


def test_whole_array_parameter_product_exports(tmp_path):
    """``p * x`` with a shaped ``p`` scalarizes element-wise rather than refusing."""
    m = dm.Model("whole_array")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=5.0)
    p = m.parameter("p", value=np.array([1.0, 2.0, 3.0]))
    m.subject_to(dm.sum(p * x) >= 6.0, name="c1")
    m.minimize(x[0] + x[1] + x[2])
    path = tmp_path / "m.nl"
    m.to_nl(str(path))
    got = from_nl(str(path)).solve(time_limit=60)
    assert got.status == "optimal"
    # Cheapest unit of the >= 6 requirement is via the largest coefficient (3).
    assert got.objective == pytest.approx(2.0, abs=1e-5)


def _gdp_model():
    m = dm.Model("gdp_export")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.either_or([[x <= 3.0, y >= 5.0], [x >= 7.0, y <= 2.0]], name="dj")
    m.minimize(x + y)
    return m


@pytest.mark.parametrize("fmt", ["to_nl", "to_lp", "to_mps"])
def test_gdp_export_refuses_loudly_not_attributeerror(fmt, tmp_path):
    """An un-reformulated GDP model must raise a ValueError naming the fix.

    Regression guard: this used to be
    ``AttributeError: '_DisjunctiveConstraint' object has no attribute 'rhs'``
    raised from inside the writer's row loop.
    """
    m = _gdp_model()
    with pytest.raises(ValueError) as exc:
        getattr(m, fmt)(str(tmp_path / "m.out"))
    msg = str(exc.value)
    assert "disjunctive" in msg.lower()
    assert "reformulate_gdp" in msg, "the refusal must name the call that fixes it"


def test_reformulated_gdp_exports(tmp_path):
    """The remedy the refusal recommends actually works."""
    from discopt._relax.gdp_reformulate import reformulate_gdp

    flat = reformulate_gdp(_gdp_model(), method="big-m")
    path = tmp_path / "flat.nl"
    flat.to_nl(str(path))
    assert path.stat().st_size > 0

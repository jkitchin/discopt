"""Issue #1362: ``partitions=k`` is a no-op off the alphaBB route, and said nothing.

``solve(partitions=k)`` is documented as "k partitions per variable for tighter
bounds", but it is read in exactly one place --
``compile_objective_relaxation`` / ``compile_constraint_relaxation``, reached
only when ``_mc_mode == "nlp"``. Any model with a relaxable nonlinearity takes
the LP-form McCormick route (``_mc_mode == "lp"``) instead, where the option is
never consulted and the bound is the standard McCormick one.

That silence is how ``docs/notebooks/advanced_features.ipynb`` came to compare

    result_std = m.solve(partitions=0, max_nodes=500)
    result_pw  = m.solve(partitions=4, max_nodes=500)

under prose promising "smaller gaps and fewer Branch & Bound nodes", and print
**5 nodes for both** -- because the second solve was the first solve.

Nothing about which route is taken changes here, and no bound moves: the only
change is that a request the solver cannot honour now says so.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest


def _lp_route_minlp():
    """The advanced_features model: nonconvex, so it takes the LP route."""
    m = dm.Model("mccormick_demo")
    x = m.continuous("x", lb=0.1, ub=5)
    y = m.continuous("y", lb=0.1, ub=5)
    z = m.binary("z")
    m.minimize(dm.exp(x) * y + z)
    m.subject_to(x + y >= 2)
    m.subject_to(x * y <= 5 * z + 1)
    return m


def _partition_warnings(record):
    return [str(w.message) for w in record if "partitions" in str(w.message)]


def test_an_ignored_partitions_request_warns():
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        result = _lp_route_minlp().solve(partitions=4, max_nodes=500)

    messages = _partition_warnings(record)
    assert messages, "an ignored option must not be silent"
    assert "partitions=4" in messages[0]
    assert "'lp'" in messages[0], messages[0]
    # The warning is the whole change: the answer is unaffected.
    assert result.status == "optimal"


def test_partitions_zero_is_silent():
    """The default must not warn -- nothing was requested."""
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        _lp_route_minlp().solve(partitions=0, max_nodes=500)
    assert _partition_warnings(record) == []


def test_the_two_solves_agree_which_is_why_the_no_op_was_invisible():
    """Pins the measurement the notebook's prose contradicted."""
    std = _lp_route_minlp().solve(partitions=0, max_nodes=500)
    pw = _lp_route_minlp().solve(partitions=4, max_nodes=500)
    assert std.node_count == pw.node_count
    assert std.objective == pytest.approx(pw.objective, abs=1e-9)

"""The primal heuristics skip binaries a bound or a single-variable row fixes.

Found while building the GDP framework (#1775 G3.3): a fixed-configuration
subproblem pins its indicators, and the one-hot searches
(``one_hot_config_subnlp`` / ``one_hot_config_dive``) and ``integer_local_search``
kept flipping them. Every such flip is infeasible (rows) or outside the box
(bounds; the incumbent guard rejected those points), and each one costs a
sub-NLP. Measured on the model below: 101 sub-NLPs with the pins as rows and 80
with them as bounds, 33.7 s / 2.0 s of wall; 12 sub-NLPs either way after, same
status, objective and node count.

Before: ``_prune_decided_groups`` / ``_pinned_bounds`` did not exist and the
solves below issued 101 and 80 sub-NLPs.
"""

from __future__ import annotations

import warnings

import discopt._relax.primal_heuristics as ph
import discopt.modeling as dm
import numpy as np
import pytest

pytestmark = pytest.mark.smoke


def _pinned(pin):
    """Two one-hot pairs over four binaries, every binary pinned; optimum 4.6."""
    m = dm.Model(f"pinned_{pin}")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    b = [m.binary(f"b{i}") for i in range(4)]
    m.subject_to(x >= 5)
    m.subject_to(x * y >= 3)
    m.subject_to(y <= 1)
    m.subject_to(b[0] + b[1] == 1)
    m.subject_to(b[2] + b[3] == 1)
    for v, val in zip(b, (0, 1, 1, 0)):
        if pin == "row":
            m.subject_to(v == val)
        else:
            v.lb = v.ub = val
    m.minimize((x - 3) ** 2 + y)
    return m


def test_prune_decided_groups():
    lb = np.array([0, 0, 0, 0, 0, 0, 1, 0.0])
    ub = np.array([1, 1, 0, 1, 1, 0, 1, 1.0])
    groups = [[0, 1, 2], [3, 5], [6, 7]]
    open_groups, decided = ph._prune_decided_groups(groups, lb, ub)
    # [0, 1, 2]: 2 is fixed at 0 and leaves; [3, 5]: one live member, decided;
    # [6, 7]: 6 is fixed at 1, decided.
    assert open_groups == [[0, 1]]
    assert decided == {2, 3, 5, 6, 7}


def test_pinned_bounds_reads_single_variable_rows_exactly():
    m = _pinned("row")
    lb, ub = ph._get_variable_bounds(m)
    plb, pub = ph._pinned_bounds(m, lb, ub)
    assert plb[2:].tolist() == pub[2:].tolist() == [0.0, 1.0, 1.0, 0.0]
    assert (plb[:2].tolist(), pub[:2].tolist()) == (lb[:2].tolist(), ub[:2].tolist())
    assert (lb[2:].tolist(), ub[2:].tolist()) == ([0.0] * 4, [1.0] * 4)  # inputs untouched


@pytest.mark.parametrize("pin", ["row", "bound"])
def test_one_hot_search_has_nothing_to_flip(pin):
    m = _pinned(pin)
    stats = {}
    x_relax = np.array([5.0, 0.6, 0.0, 1.0, 1.0, 0.0])
    out = ph.one_hot_config_subnlp(m, x_relax, stats=stats)
    assert out == [] and stats["attempted"] == 0


@pytest.mark.parametrize("pin", ["row", "bound"])
def test_a_pinned_model_costs_few_sub_nlps(pin, monkeypatch):
    calls = []
    real = ph.subnlp
    monkeypatch.setattr(ph, "subnlp", lambda *a, **k: calls.append(1) or real(*a, **k))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = _pinned(pin).solve(time_limit=60)
    assert r.status == "optimal" and r.objective == pytest.approx(4.6, abs=1e-6)
    assert calls, "the probe never saw a sub-NLP"
    assert len(calls) <= 20, f"{len(calls)} sub-NLPs (101 / 80 before the fix)"

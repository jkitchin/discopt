"""MIP-NLP decomposition method/option matrix on a convex MINLP (#87).

One closed-form convex MINLP (optimum 1.94 at i=1, x=1.5) solved through
every mip-nlp method (OA, ECP, feasibility pump, global OA) and the main
option toggles. Decomposition options may change the path, never the
certificate — except FP, which is a primal heuristic and reports
'feasible' with the same incumbent.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest
from discopt.modeling.core import Model

pytestmark = pytest.mark.smoke

_OPT = 1.94  # i=1, x=1.5: (1.5-0.7)^2 + 1.3


def _cvx_minlp():
    m = Model("mipnlp")
    i = m.integer("i", lb=0, ub=4)
    x = m.continuous("x", lb=0.0, ub=4.0)
    m.subject_to(x + i >= 2.5)
    m.minimize((x - 0.7) ** 2 + 1.3 * i)
    return m


@pytest.mark.parametrize("method", ["oa", "ecp", "goa"])
def test_mip_nlp_methods_certify(method):
    res = _cvx_minlp().solve(solver="mip-nlp", mip_nlp_method=method, time_limit=60.0)
    assert res.status == "optimal"
    assert res.objective == pytest.approx(_OPT, abs=1e-4)


def test_feasibility_pump_finds_incumbent():
    res = _cvx_minlp().solve(solver="mip-nlp", mip_nlp_method="fp", time_limit=60.0)
    # FP is a primal heuristic: feasible incumbent, no optimality claim
    # beyond what it earned; on this instance it lands on the optimum.
    assert res.status in ("optimal", "feasible")
    assert res.objective == pytest.approx(_OPT, abs=1e-4)


@pytest.mark.parametrize("strategy", ["rNLP", "initial_binary", "max_binary"])
def test_oa_init_strategies(strategy):
    res = _cvx_minlp().solve(
        solver="mip-nlp", mip_nlp_method="oa", init_strategy=strategy, time_limit=60.0
    )
    assert res.status == "optimal"
    assert res.objective == pytest.approx(_OPT, abs=1e-4)


def test_oa_option_toggles():
    for kw in (
        {"add_no_good_cuts": True},
        {"equality_relaxation": True},
        {"add_slack": True},
    ):
        res = _cvx_minlp().solve(solver="mip-nlp", mip_nlp_method="oa", time_limit=60.0, **kw)
        assert res.status == "optimal", kw
        assert res.objective == pytest.approx(_OPT, abs=1e-4), kw


def test_unknown_mip_nlp_method_rejected_loudly():
    with pytest.raises(ValueError, match="mip_nlp_method"):
        _cvx_minlp().solve(solver="mip-nlp", mip_nlp_method="bogus", time_limit=10.0)


def test_unknown_init_strategy_rejected_loudly():
    with pytest.raises(ValueError, match="init_strategy"):
        _cvx_minlp().solve(
            solver="mip-nlp", mip_nlp_method="oa", init_strategy="bogus", time_limit=10.0
        )

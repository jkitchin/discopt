"""#2143: the root LOA configuration proposer (``DISCOPT_GDP_LOA_PROPOSER``, default OFF).

GDPlib ``syngas`` under ``gdp_method="hull"`` has no incumbent at 120 s. No root
heuristic proposes a feasible disjunct configuration, while LOA on the source GDP
reaches the optimal one in two iterations. With the flag ON, when the root
configuration wave and its dive return nothing on a lowered GDP, LOA runs on the
source model until its first verified incumbent. Its point is completed onto the
lowered model and posed through ``subnlp`` there (verified on the lowered
evaluator), then injected like a wave point. Primal only.

The toy below makes the wave return nothing (monkeypatched), so the only way an
incumbent can reach the search at the root is the proposer. Under hull the
proposer's sub-NLP needs ``DISCOPT_SUBNLP_REDUCED``: on the legacy pose it fails
for the reason #2143 is about, even on the right configuration (measured on this
toy: status ERROR legacy, OPTIMAL 2.0 reduced). The tests set both flags.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

pytest.importorskip("pounce")

import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402
from discopt._relax import primal_heuristics as PH  # noqa: E402
from discopt._tape_nlp_evaluator import make_evaluator  # noqa: E402
from discopt.gdp.algorithms.loa import LOA  # noqa: E402
from discopt.transformations import get as get_transformation  # noqa: E402
from discopt.validation.feasibility import verify_point  # noqa: E402

FLAG = "DISCOPT_GDP_LOA_PROPOSER"
#: The toy's optimum: x = 2 in the left disjunct (cost (2-3)^2 + 1).
TOY_OPT = 2.0


def _toy():
    """One disjunction; the left disjunct holds the optimum."""
    m = dm.Model("loa_proposer_toy")
    x = m.continuous("x", lb=0, ub=10)
    z = m.continuous("z", lb=0, ub=10)
    m.either_or(
        [
            [x <= 2, z >= 1],
            [x >= 6, z >= 4 + dm.exp(0.1 * x)],
        ],
        name="d",
    )
    m.minimize((x - 3) ** 2 + z)
    return m


def test_flag_defaults_off(monkeypatch):
    monkeypatch.delenv(FLAG, raising=False)
    assert S._gdp_loa_proposer_enabled() is False
    monkeypatch.setenv(FLAG, "1")
    assert S._gdp_loa_proposer_enabled() is True


def test_loa_stops_at_its_first_incumbent():
    alg = LOA(_toy(), time_limit=60, stop_at_first_incumbent=True)
    r = alg.run()
    assert r.x is not None
    assert alg.incumbent is not None
    assert alg.stop_reason in ("first_incumbent", "converged")


def test_proposal_is_a_point_of_the_lowered_model(monkeypatch):
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "1")
    src = _toy()
    low = get_transformation("gdp").apply(src, method="hull")
    assert low is not src
    ev = make_evaluator(low)
    names = low._gdp_disjunct_selectors
    assert names and all(v is not None for v in names.values()), names
    found = S._gdp_loa_root_proposal(
        src,
        low,
        selector_names=names,
        evaluator=ev,
        backend=None,
        nlp_options=None,
        deadline=1e18,
    )
    assert found is not None
    x, obj = found
    assert verify_point(low, np.asarray(x, dtype=float), with_objective=True).ok
    assert obj == pytest.approx(TOY_OPT, abs=1e-5)


def test_lowering_records_selectors_in_disjunct_order():
    low = get_transformation("gdp").apply(_toy(), method="hull")
    (names,) = low._gdp_disjunct_selectors.values()
    assert len(names) == 2
    known = {v.name for v in low._variables}
    assert all(n in known for n in names)


def test_without_the_selector_record_the_completion_misses(monkeypatch):
    """Why the record exists: the generic completion cannot place hull selectors."""
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "1")
    src = _toy()
    low = get_transformation("gdp").apply(src, method="hull")
    found = S._gdp_loa_root_proposal(
        src,
        low,
        selector_names=None,
        evaluator=make_evaluator(low),
        backend=None,
        nlp_options=None,
        deadline=1e18,
    )
    assert found is None


def test_proposal_refuses_a_misaligned_model():
    src = _toy()
    other = dm.Model("other")
    other.continuous("w", lb=0, ub=1)
    other.minimize(other._variables[0])
    assert (
        S._gdp_loa_root_proposal(
            src,
            other,
            evaluator=make_evaluator(other),
            backend=None,
            nlp_options=None,
            deadline=1e18,
        )
        is None
    )


def test_proposal_refuses_too_small_a_budget():
    src = _toy()
    low = get_transformation("gdp").apply(src, method="hull")
    import time

    assert (
        S._gdp_loa_root_proposal(
            src,
            low,
            evaluator=make_evaluator(low),
            backend=None,
            nlp_options=None,
            deadline=time.perf_counter() + 1.0,
        )
        is None
    )


def _solve_with_failed_wave(monkeypatch, flag: str):
    """Solve the toy at a 1-node budget with the root wave forced to fail.

    Returns ``(result, proposer_calls, log_lines)``.
    """
    monkeypatch.setenv(FLAG, flag)
    monkeypatch.setenv("DISCOPT_SUBNLP_REDUCED", "1")
    monkeypatch.setattr(PH, "one_hot_config_subnlp", lambda *a, **k: [])
    calls = []
    real = S._gdp_loa_root_proposal

    def spy(*a, **k):
        out = real(*a, **k)
        calls.append(out)
        return out

    monkeypatch.setattr(S, "_gdp_loa_root_proposal", spy)
    records = []

    class H(logging.Handler):
        def emit(self, rec):
            records.append(rec.getMessage())

    h = H()
    lg = logging.getLogger("discopt")
    old = lg.level
    lg.setLevel(logging.INFO)
    lg.addHandler(h)
    try:
        r = _toy().solve(gdp_method="hull", time_limit=60, max_nodes=1)
    finally:
        lg.removeHandler(h)
        lg.setLevel(old)
    return r, calls, records


def test_failed_wave_gets_the_loa_point_injected(monkeypatch):
    """Fails without the proposer: the wave returns nothing and no LOA point exists."""
    r, calls, records = _solve_with_failed_wave(monkeypatch, "1")
    assert len(calls) == 1 and calls[0] is not None
    assert any("GDP config subNLP incumbent" in m for m in records), records
    assert r.objective == pytest.approx(TOY_OPT, abs=1e-5)


def test_flag_off_never_calls_the_proposer(monkeypatch):
    r, calls, records = _solve_with_failed_wave(monkeypatch, "0")
    assert calls == []
    assert not any("GDP LOA proposer" in m for m in records)


def test_proposer_does_not_fire_inside_its_own_loa(monkeypatch):
    """LOA solves configuration subproblems with a nested solve; no proposer there."""
    token = S._IN_GDP_LOA_PROPOSER.set(True)
    try:
        r, calls, records = _solve_with_failed_wave(monkeypatch, "1")
    finally:
        S._IN_GDP_LOA_PROPOSER.reset(token)
    assert calls == []

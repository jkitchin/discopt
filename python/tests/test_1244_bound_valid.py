"""``SolveResult.bound_valid`` / ``bound_source`` (#1244).

A consumer that wants to say "globally certified to eps" has to know whether
``bound`` is a valid global dual bound. Until now the only public signal was
``gap_certified``, which answers a *narrower* question: whether the gap CLOSED.
A ``time_limit`` or ``node_limit`` exit never closes the gap, yet its bound is
usually perfectly valid -- the frontier minimum over a tree whose every node
bound entered with a soundness proof, or an independently proved root bound
that replaced a tainted one. Reading ``gap_certified`` throws all of those away;
reading ``bound`` without it risks using one that is not valid. The real flag
lived in ``solver.py`` internals (``_tree_bound_valid``).

The tests below are the acceptance criteria:

* a table over termination status x algorithm route, on instances with RECORDED
  optima, asserting the soundness invariant ``bound_valid => bound <= f* + tol``
  (``>=`` for a maximize) -- including ``node_limit=3`` stops, very small time
  limits, and convex fast-path instances;
* ``bound_valid`` is False whenever ``bound`` is None.

Soundness note: the invariant is asserted against the reference-optima registry
(``tests/data/known_optima.toml``), the same oracle the certification suites
use, not against discopt's own answer -- a bound checked against the solver that
produced it proves nothing.
"""

from __future__ import annotations

import math
from pathlib import Path

import discopt.modeling as dm
import pytest
from _optima import known_optimum, optima_registry
from discopt import Model
from discopt.modeling.core import BOUND_SOURCES, SolveResult

_CORPUS = Path(__file__).parent / "data" / "minlplib_nl"

#: Instances with a recorded optimum that are also vendored as ``.nl`` here.
_ORACLE_INSTANCES = sorted(n for n in optima_registry() if (_CORPUS / f"{n}.nl").exists())

#: Termination regimes to sweep. Each forces a different exit: a full solve, a
#: node-budget stop, and a wall-budget stop. ``node_limit=3`` and a very small
#: ``time_limit`` are named in the issue's acceptance criteria.
_REGIMES = {
    "full": {"time_limit": 20},
    "node_limit_3": {"time_limit": 20, "max_nodes": 3},
    "tiny_time_limit": {"time_limit": 0.2},
}


def _tol_for(optimum: float) -> float:
    """Absolute slack for the soundness comparison, scaled like the suites'."""
    return 1e-6 + 1e-6 * abs(float(optimum))


# ──────────────────────────────────────────────────────────────────────
# 1. The dataclass contract
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_bound_valid_is_false_whenever_bound_is_none():
    """The issue's second acceptance criterion, at the chokepoint that enforces it."""
    checked = 0
    for status in ("time_limit", "node_limit", "feasible", "infeasible", "unbounded"):
        r = SolveResult(status=status, objective=1.0, bound=None, bound_valid=True)
        assert r.bound_valid is False, status
        assert r.bound_source is None, status
        checked += 1
    assert checked == 5
    # A non-finite bound is no bound either.
    for bad in (float("inf"), float("-inf"), float("nan")):
        r = SolveResult(
            status="time_limit", objective=1.0, bound=bad, bound_valid=True, gap_certified=False
        )
        assert r.bound_valid is False


@pytest.mark.unit
def test_a_certified_gap_implies_a_valid_bound():
    """``gap_certified`` is the repo's certificate flag; it asserts bound validity.

    Deriving here rather than at each of the ~43 construction sites is what
    keeps a new return site from reporting "no claim" on a bound it certified.
    """
    r = SolveResult(status="optimal", objective=2.0, bound=2.0, gap_certified=True)
    assert r.bound_valid is True
    # ...but an infeasibility certificate is not a bound certificate.
    r2 = SolveResult(status="infeasible", bound=None, gap_certified=True)
    assert r2.bound_valid is False


@pytest.mark.unit
def test_a_local_status_may_not_claim_a_valid_bound():
    """A local solve makes no global claim, so it can carry no dual bound."""
    from discopt.status import is_local_status

    local = next(
        s for s in ("local_optimal", "local_feasible", "local_stationary") if is_local_status(s)
    )
    r = SolveResult(status=local, objective=1.0, bound=None, bound_valid=True)
    assert r.bound_valid is False and r.bound_source is None


@pytest.mark.unit
def test_an_unknown_bound_source_is_refused():
    """Closed vocabulary: a typo must not silently read as "some other source"."""
    with pytest.raises(ValueError, match="bound_source"):
        SolveResult(status="optimal", objective=1.0, bound=1.0, bound_source="bnb")
    for src in sorted(BOUND_SOURCES):
        r = SolveResult(status="optimal", objective=1.0, bound=1.0, bound_source=src)
        assert r.bound_source == src


@pytest.mark.unit
def test_the_claim_survives_a_serialization_round_trip():
    """``__post_init__`` can only DERIVE validity from ``gap_certified``.

    A stored result that drops ``bound_valid`` therefore reloads with every
    uncertified-but-valid bound downgraded to "no claim" — silently, and exactly
    the field a consumer reads the file for.
    """
    from discopt.result_io import deserialize_result, serialize_result

    original = SolveResult(
        status="time_limit",
        objective=1.0,
        bound=0.5,
        gap_certified=False,
        bound_valid=True,
        bound_source="bnb_tree",
    )
    assert original.bound_valid is True  # not clobbered on the way in
    restored = deserialize_result(serialize_result(original))
    assert restored.bound_valid is True
    assert restored.bound_source == "bnb_tree"


# ──────────────────────────────────────────────────────────────────────
# 2. The convex fast path
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_convex_fast_path_reports_a_valid_bound_named_by_its_proof():
    """``bound = objective`` there, valid because the convexity proof is rigorous."""
    m = Model()
    x = m.continuous("x", shape=3, lb=-5.0, ub=5.0)
    m.subject_to(dm.sum(x) >= 1.0)
    m.minimize(dm.sum(x * x))
    res = m.solve(time_limit=30)
    assert res.status == "optimal"
    assert res.bound_valid is True
    assert res.bound is not None
    # f* = 1/3 for min ||x||^2 s.t. sum(x) >= 1 in R^3.
    assert res.bound <= 1.0 / 3.0 + 1e-6
    if res.convex_fast_path:
        assert res.bound_source == "convex_proof"


# ──────────────────────────────────────────────────────────────────────
# 3. The acceptance table: status x route, against recorded optima
# ──────────────────────────────────────────────────────────────────────


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.parametrize("name", _ORACLE_INSTANCES)
@pytest.mark.parametrize("regime", sorted(_REGIMES))
def test_a_valid_bound_never_crosses_the_recorded_optimum(name, regime):
    """``bound_valid => bound <= f* + tol`` (minimize) on every status x route.

    This is the invariant the whole field exists to make safe to rely on: a
    consumer reading ``bound_valid`` and then using ``bound`` as a global bound
    must never be handed one that crosses the true optimum.
    """
    from discopt.modeling import from_nl

    entry = known_optimum(name, full=True)
    optimum = float(entry["optimum"])
    model = from_nl(str(_CORPUS / f"{name}.nl"))
    res = model.solve(**_REGIMES[regime])

    # Whatever happened, the pairing must be self-consistent.
    if res.bound is None:
        assert res.bound_valid is False
        assert res.bound_source is None
        pytest.skip(f"{name}/{regime}: no bound reported; nothing to check")
    assert res.bound_source is None or res.bound_source in BOUND_SOURCES

    if not res.bound_valid:
        pytest.skip(f"{name}/{regime}: bound reported but not claimed valid")

    assert math.isfinite(res.bound)
    from discopt.modeling.core import ObjectiveSense

    is_max = model._objective.sense == ObjectiveSense.MAXIMIZE
    tol = _tol_for(optimum)
    if is_max:
        assert res.bound >= optimum - tol, (
            f"{name}/{regime}: bound_valid bound {res.bound!r} is BELOW the "
            f"recorded optimum {optimum!r} for a maximize model "
            f"(status={res.status}, source={res.bound_source})"
        )
    else:
        assert res.bound <= optimum + tol, (
            f"{name}/{regime}: bound_valid bound {res.bound!r} EXCEEDS the "
            f"recorded optimum {optimum!r} (status={res.status}, "
            f"source={res.bound_source})"
        )
    # And the certificate invariant against this run's own incumbent.
    if res.objective is not None and math.isfinite(res.objective):
        if is_max:
            assert res.bound >= res.objective - tol
        else:
            assert res.bound <= res.objective + tol


@pytest.mark.slow
@pytest.mark.correctness
def test_the_table_actually_covered_budget_limited_exits():
    """Prove the sweep above is not vacuous (CLAUDE.md §6).

    A table whose every row certifies would assert nothing about the case the
    issue is really about — a bound reported at a budget stop. Count the
    regimes that produced a non-``optimal`` status carrying a valid bound, and
    fail if none did.
    """
    from discopt.modeling import from_nl

    budget_limited_with_valid_bound = 0
    rows = 0
    for name in _ORACLE_INSTANCES:
        model = from_nl(str(_CORPUS / f"{name}.nl"))
        res = model.solve(time_limit=20, max_nodes=3)
        rows += 1
        if res.status != "optimal" and res.bound_valid:
            budget_limited_with_valid_bound += 1
            optimum = float(known_optimum(name))
            from discopt.modeling.core import ObjectiveSense

            is_max = model._objective.sense == ObjectiveSense.MAXIMIZE
            tol = _tol_for(optimum)
            if is_max:
                assert res.bound >= optimum - tol, (name, res.bound, optimum)
            else:
                assert res.bound <= optimum + tol, (name, res.bound, optimum)
    assert rows == len(_ORACLE_INSTANCES)
    assert budget_limited_with_valid_bound > 0, (
        "no instance produced a budget-limited exit carrying a valid bound; the "
        "acceptance table proves nothing about the case #1244 exists for"
    )

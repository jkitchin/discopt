"""GAMS ``optca=0`` must not introduce an unrequested absolute stopping tolerance (#1889).

GAMS defines ``optca`` as ``|PB - DB| <= optca``, so ``0`` grants no absolute
allowance. The link used to map it to ``None``, which the solver resolves to its
own 1e-6 default: the pair ``PB=1e-7, DB=0`` at ``optcr=1e-4`` (relative gap 1,
absolute gap 1e-7) was declared converged although neither requested criterion
held.

Read literally, though, ``0`` cannot certify an optimum at exactly 0: rigorous
bounds carry a safety margin (HiGHS's LP route publishes ``-1e-12`` against
``0``) and the relative gap is undefined there, so GAMS's DEFAULT ``optca=0``
reported a one-line LP ``min x, x in [0,1]`` as merely feasible. The link
therefore reads ``0`` at floating-point resolution:
``|PB-DB| <= 1e-9 * max(1, |PB|, |DB|)`` or the ``optcr`` test.
"""

from types import SimpleNamespace

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt.gams.link import OPTCA_ZERO_RESOLUTION, _read_gams_options
from discopt.solver import (
    _gap_criterion,
    _gap_values_converged,
    _native_kernel_abs_tol,
    _recertify_gap_closed,
    _resolve_abs_gap_tolerance,
)


def _gams_options(optcr: float, optca: float) -> dict:
    values = {"ResLim": 10.0, "OptCR": optcr, "OptCA": optca}
    gev = SimpleNamespace(
        gevResLim="ResLim",
        gevOptCR="OptCR",
        gevOptCA="OptCA",
        gevNodeLim="NodeLim",
        gevGetDblOpt=lambda handle, name: values[name],
        gevGetIntOpt=lambda handle, name: 0,
        gevThreads=lambda handle: 1,
    )
    return _read_gams_options(gev, None)


@pytest.mark.unit
@pytest.mark.parametrize(
    "optca, expected_convergence",
    [(0.0, False), (1e-9, False), (1e-6, True)],
    ids=["zero-absolute-tolerance", "tight-positive", "loose-positive"],
)
def test_gams_absolute_tolerance_controls_near_zero_gap(optca, expected_convergence):
    # Feasible incumbent PB=1e-7 and valid DB=0: absolute gap 1e-7,
    # relative gap 1.  Neither requested criterion holds when optCA=0.
    options = _gams_options(1e-4, optca)
    absolute_tolerance = _resolve_abs_gap_tolerance(options["abs_gap"])
    converged = _gap_values_converged(1e-7, 0.0, options["gap"], absolute_tolerance)
    assert converged == expected_convergence, (
        f"GAMS optca={optca:g}, optcr={options['gap']:g}: "
        f"PB=1e-7, DB=0 was checked with effective absolute tolerance "
        f"{absolute_tolerance:g} and converged={converged}"
    )


@pytest.mark.unit
def test_omitted_abs_gap_keeps_the_discopt_default():
    """Non-GAMS callers that omit the option are untouched."""
    assert _resolve_abs_gap_tolerance(None) == 1e-6
    assert _gap_values_converged(1e-7, 0.0, 1e-4, _resolve_abs_gap_tolerance(None))


# (incumbent, bound) in the MINIMIZE orientation, the GAMS optcr, and whether
# ``|PB-DB| <= 0 OR rel <= optcr`` holds. The MAXIMIZE case mirrors each pair.
_CASES = [
    # near zero: only a gap at floating-point resolution terminates
    (1e-7, 0.0, 1e-4, False),
    (1e-7, 0.0, 0.0, False),
    (1e-12, 0.0, 1e-4, True),  # a rigorous bound's safety margin
    (1e-12, 0.0, 0.0, True),
    (0.0, 0.0, 1e-4, True),
    (0.0, 0.0, 0.0, True),
    # ordinary scale: the relative criterion still terminates
    (100.0 + 1e-3, 100.0, 1e-4, True),
    (100.0 + 1e-1, 100.0, 1e-4, False),
    (100.0 + 1e-3, 100.0, 0.0, False),
    (100.0 + 1e-8, 100.0, 0.0, True),  # resolution scales with |PB|
    (100.0, 100.0, 0.0, True),
]


@pytest.mark.unit
@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("inc, bnd, optcr, closed", _CASES)
def test_zero_optca_criterion_on_every_certification_predicate(sense, inc, bnd, optcr, closed):
    options = _gams_options(optcr, 0.0)
    assert options["abs_gap"] == OPTCA_ZERO_RESOLUTION
    abs_tol = _resolve_abs_gap_tolerance(options["abs_gap"])
    if sense == "max":
        inc, bnd = -inc, -bnd
    is_max = sense == "max"
    assert _recertify_gap_closed(inc, bnd, is_max, options["gap"], abs_tol) is closed
    hi, lo = (bnd, inc) if is_max else (inc, bnd)
    assert (_gap_criterion(hi, lo, options["gap"], abs_tol) is not None) is closed
    assert _gap_values_converged(hi, lo, options["gap"], abs_tol) is closed


@pytest.mark.unit
def test_native_kernel_keeps_its_relative_stop_under_zero_optca():
    """Zero switches the absolute arm off; ``min`` would leave only exact closure."""
    assert _native_kernel_abs_tol(1e-4, 0.0) == 1e-4
    assert _native_kernel_abs_tol(1e-4, 1e-9) == 1e-9
    assert _native_kernel_abs_tol(1e-4, None) == 1e-4


def _bilinear(sense: str) -> Model:
    m = Model()
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    if sense == "min":
        m.minimize(x * y - x)
    else:
        m.maximize(x - x * y)
    return m


def _knapsack(sense: str) -> Model:
    rng = np.random.default_rng(3)
    w = rng.integers(20, 90, 12).astype(float)
    v = w + rng.integers(0, 12, 12)
    m = Model()
    x = m.binary("x", shape=12)
    m.subject_to(dm.sum(w * x) <= float(w.sum() * 0.37))
    if sense == "min":
        m.minimize(-dm.sum(v * x))
    else:
        m.maximize(dm.sum(v * x))
    return m


def _assert_published_pair_meets_zero_optca(r, optcr: float) -> None:
    assert r.status == "optimal", (r.status, r.algorithm_route)
    abs_gap = abs(r.objective - r.bound)
    rel_gap = abs_gap / max(abs(r.objective), abs(r.bound), 1e-10)
    resolution = OPTCA_ZERO_RESOLUTION * max(1.0, abs(r.objective), abs(r.bound))
    assert abs_gap <= resolution or rel_gap <= optcr, (r.objective, r.bound, r.algorithm_route)


@pytest.mark.slow
@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("native", ["1", "0"], ids=["native-kernel", "python-tree"])
def test_zero_optca_solves_on_spatial_routes(monkeypatch, sense, native):
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", native)
    options = _gams_options(1e-4, 0.0)
    r = _bilinear(sense).solve(
        time_limit=60, gap_tolerance=options["gap"], abs_gap_tolerance=options["abs_gap"]
    )
    _assert_published_pair_meets_zero_optca(r, options["gap"])
    assert r.objective == pytest.approx(-1.0 if sense == "min" else 1.0, abs=1e-6)


@pytest.mark.slow
@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("backend", ["highs", "rust"])
def test_zero_optca_solves_on_milp_routes(monkeypatch, sense, backend):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    options = _gams_options(1e-4, 0.0)
    r = _knapsack(sense).solve(
        time_limit=60, gap_tolerance=options["gap"], abs_gap_tolerance=options["abs_gap"]
    )
    _assert_published_pair_meets_zero_optca(r, options["gap"])
    assert abs(r.objective) == pytest.approx(241.0)


@pytest.mark.slow
@pytest.mark.parametrize("sense", ["min", "max"])
@pytest.mark.parametrize("backend", ["highs", "rust"])
def test_zero_optcr_and_optca_still_certify(monkeypatch, sense, backend):
    """``optcr=0, optca=0`` must certify a solved MILP, at resolution, not at 1e-6."""
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", backend)
    options = _gams_options(0.0, 0.0)
    r = _knapsack(sense).solve(
        time_limit=60, gap_tolerance=options["gap"], abs_gap_tolerance=options["abs_gap"]
    )
    _assert_published_pair_meets_zero_optca(r, options["gap"])
    assert abs(r.objective) == pytest.approx(241.0)


@pytest.mark.slow
@pytest.mark.parametrize("sense", ["min", "max"])
def test_zero_optimum_lp_certifies_under_gams_defaults(sense):
    """GAMS defaults (optcr=1e-4, optca=0): ``min x, x in [0,1]`` is Optimal.

    Measured: with optca=0 forwarded literally the published pair was
    ``(0, -1e-12)`` and the certificate was withdrawn.
    """
    options = _gams_options(1e-4, 0.0)
    m = Model()
    x = m.continuous("x", lb=0.0, ub=1.0)
    if sense == "min":
        m.minimize(x)
    else:
        m.maximize(-x)
    r = m.solve(gap_tolerance=options["gap"], abs_gap_tolerance=options["abs_gap"])
    _assert_published_pair_meets_zero_optca(r, options["gap"])
    assert r.gap_certified

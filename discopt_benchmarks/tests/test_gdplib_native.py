"""Tests for the native-discopt GDPlib models (issue #823).

Unlike :mod:`test_gdplib` (which needs pyomo + gdplib for the bridge), these
exercise discopt's *own* disjunction machinery on transcribed models, so they need
only discopt. Each native builder must reach the SCIP-certified optimum its Pyomo
counterpart does — a native encoding that lands elsewhere is a porting bug.
"""

from __future__ import annotations

import pytest

from benchmarks import gdplib_native as gn
from benchmarks.gdplib_runner import reference_optima
from benchmarks.metrics import SolveStatus

# (name, certified optimum, abs tolerance) — optima from reference_optima().
_CERTIFIED = [
    ("jobshop", 11.0, 1e-3),
    ("ex1_linan_2023", -0.9996, 1e-3),
    ("small_batch", 167427.6515668371, 1.0),
]


def test_registry_matches_builders():
    assert set(gn.NATIVE_BUILDERS) == {"jobshop", "ex1_linan_2023", "small_batch"}
    for name in gn.NATIVE_BUILDERS:
        assert name in reference_optima(), f"{name} lacks a certified reference optimum"


@pytest.mark.parametrize("name", sorted(gn.NATIVE_BUILDERS))
def test_builder_returns_solvable_model(name):
    """Every builder returns a discopt Model with an objective and disjunctions."""
    model = gn.NATIVE_BUILDERS[name]()
    # discopt Model exposes either_or/subject_to; a built model is non-empty.
    assert model is not None
    assert hasattr(model, "solve")


@pytest.mark.correctness
@pytest.mark.parametrize(("name", "opt", "tol"), _CERTIFIED)
def test_native_reaches_certified_optimum(name, opt, tol):
    """The native encoding certifies the same optimum as the Pyomo-bridged model."""
    run = gn.solve_native(name, time_limit=120)
    assert run.discopt.status == SolveStatus.OPTIMAL, run.note
    assert run.discopt.objective == pytest.approx(opt, abs=tol)
    # Soundness gate: no impossible incumbent, false optimum, or bound crossing.
    assert run.false_optimum is False, run.note
    assert run.bound_crosses is False, run.note
    assert run.oracle_source == "reference"


def test_solve_native_unknown_name_raises():
    with pytest.raises(KeyError):
        gn.solve_native("not_a_model")


def test_run_native_suite_no_violations():
    runs = gn.run_native_suite(time_limit=120)
    assert len(runs) == len(gn.NATIVE_BUILDERS)
    assert all(not r.false_optimum and not r.bound_crosses for r in runs)
    assert all(r.discopt.status == SolveStatus.OPTIMAL for r in runs)

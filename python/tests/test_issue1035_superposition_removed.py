"""#1035: the inert ``superposition`` knob is gone, and cannot come back silently.

The switch reached ``build_milp_relaxation`` as a parameter the function never
read — the #632 uniform-factorable cutover stopped consuming it, and its own
docstring listed it among the arguments "currently **IGNORED** on the default
path".  Everything downstream of it (``discopt._relax.superposition``) was
reachable only from its own tests.  A knob that is accepted and ignored is worse
than no knob: ``relaxation_arithmetic="superposition"`` silently returned the
plain McCormick relaxation, and a measurement taken with that arm set reads as a
superposition-cut measurement (one such row is retracted in
``docs/dev/performance-plan.md`` §6).

M8's tracking issue (#81) is closed, so there is no plan the parameter is holding
a place for.  #1035 deleted it.

These tests are the tripwire: each one *passed silently* before the deletion
(the argument was accepted, the module imported) and now pins the removal.  They
are deliberately about the seams a re-introduction would have to pass through,
not about the cut math, which no longer exists.
"""

import importlib

import numpy as np
import pytest
from discopt._relax.mccormick_lp import MccormickLPRelaxer, build_milp_relaxation
from discopt._relax.obbt import obbt_tighten_root
from discopt._relax.root_reduce import run_root_fixpoint


def test_the_cut_generator_module_is_gone():
    """It was importable only from its own tests; nothing in the solver used it."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("discopt._relax.superposition")


def test_the_relaxer_rejects_the_knob():
    """Accepting-and-ignoring is the defect; a loud ``TypeError`` is the fix."""
    import discopt.modeling as dm

    m = dm.Model("issue1035")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=0.0, ub=1.0)
    m.minimize(x * y)
    with pytest.raises(TypeError, match="superposition"):
        MccormickLPRelaxer(m, superposition=True)


def test_the_builder_rejects_the_knob():
    with pytest.raises(TypeError, match="superposition"):
        build_milp_relaxation(None, None, None, superposition=True)


@pytest.mark.parametrize("fn", [obbt_tighten_root, run_root_fixpoint])
def test_the_reduce_entry_points_reject_the_knob(fn):
    """The two root-reduction entry points threaded it down to the builder."""
    with pytest.raises(TypeError, match="superposition"):
        fn(None, np.zeros(1), np.ones(1), superposition=True)


def test_a_bilinear_model_still_solves_to_its_known_optimum():
    """The deletion is inert on the solve path — nothing was consuming it.

    ``min x*y`` on ``[0,1] x [-1,1]`` has optimum ``-1`` (at ``x=1, y=-1``), which
    the McCormick envelope attains exactly, so any change in the relaxation the
    removal accidentally caused would show here.
    """
    import discopt.modeling as dm

    m = dm.Model("issue1035_solve")
    x = m.continuous("x", lb=0.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(x * y)
    result = m.solve(time_limit=60.0, gap_tolerance=1e-6)
    assert result.status in ("optimal", "feasible")
    assert result.objective == pytest.approx(-1.0, abs=1e-6)
    assert result.bound is not None
    assert result.bound <= result.objective + 1e-6

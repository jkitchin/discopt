"""Regression tests for #517 candidate (A): in-house numerical-dual safe bound.

Root cause (``docs/dev/hda-no-bound-simplex-robustness-2026-07-16.md``): on the
hda-class ill-conditioned flowsheet relaxations the in-house simplex's phase 1
finds a feasible basis but phase 2 drifts / breaks down (``Numerical``), so the
node certifies no dual bound and the tree never fathoms — hda has *no* dual bound
at all.

Fix (flag ``DISCOPT_NODE_NUMERICAL_DUAL_BOUND`` /
``SolverTuning.node_numerical_dual_bound``; shipped default OFF under the
bound-changing regime, graduated to default ON with #362 —
``DISCOPT_NODE_NUMERICAL_DUAL_BOUND=0`` restores the legacy no-rescue behavior):
export the Optimal-style dual candidate ``y = B⁻ᵀc_B`` from the broken basis and
attach the in-repo Neumaier–Shcherbina safe lower bound it yields. The NS bound is
valid for ANY multiplier vector, so a drifted-basis dual only *loosens* it — never
lifts it above the optimum — and it is reported as a bound-only node (no fabricated
incumbent), so it never fathoms on its own. No external solver is used.
"""

import math
import os

import discopt.modeling as dm
import pytest

_NL_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

_FLAG = "DISCOPT_NODE_NUMERICAL_DUAL_BOUND"
_HDA_OPT = -5964.534084  # published MINLPLib global optimum


def _hda_path():
    p = os.path.join(_NL_DATA, "hda.nl")
    if not os.path.exists(p):
        pytest.skip("hda.nl not vendored")
    return p


@pytest.mark.slow
def test_hda_gets_first_finite_dual_bound(monkeypatch):
    """With the flag ON, hda reports a *finite* dual bound (its first) that is a
    sound lower bound (never above the published optimum)."""
    monkeypatch.setenv(_FLAG, "1")
    r = dm.from_nl(_hda_path()).solve(time_limit=25)
    assert r.bound is not None, "hda should get its first finite dual bound with the flag ON"
    assert math.isfinite(r.bound), f"bound must be finite, got {r.bound}"
    # Soundness: a valid dual (lower) bound never crosses the true optimum.
    assert r.bound <= _HDA_OPT + 1e-2, f"UNSOUND: bound {r.bound:.6g} > opt {_HDA_OPT}"
    # Bound-only: the floor must not fabricate an incumbent or a false optimality.
    assert r.status != "optimal", "a loose dual floor must not claim optimality"


@pytest.mark.slow
def test_hda_flag_disabled_restores_the_legacy_baseline(monkeypatch):
    """Flag disabled (=0, the graduation escape hatch): hda has no dual bound —
    the legacy no-rescue baseline is reachable and untouched."""
    monkeypatch.setenv(_FLAG, "0")
    r = dm.from_nl(_hda_path()).solve(time_limit=25)
    assert r.bound is None, f"flag disabled must be the no-bound baseline, got {r.bound}"


@pytest.mark.slow
@pytest.mark.parametrize("name", ["alan", "ex1221"])
def test_inert_on_cleanly_certifying_instances(name, monkeypatch):
    """Instances whose node LPs solve cleanly (no numerical breakdown) are
    byte-identical with the flag ON: the floor fires only on a failed node LP, so
    a well-conditioned certifying instance never triggers it."""
    path = os.path.join(_NL_DATA, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name}.nl not vendored")

    monkeypatch.setenv(_FLAG, "0")
    off = dm.from_nl(path).solve(time_limit=20)
    monkeypatch.setenv(_FLAG, "1")
    on = dm.from_nl(path).solve(time_limit=20)

    assert off.status == on.status, f"{name}: status changed {off.status} -> {on.status}"
    assert off.objective == on.objective, f"{name}: objective drifted with the flag"
    assert off.bound == on.bound, f"{name}: bound drifted with the flag ({off.bound} -> {on.bound})"


def test_flag_defaults_on(monkeypatch):
    """The tuning flag is default-ON (graduated with #362; ``=0`` restores the
    legacy no-rescue behavior).

    Check the *code* default in the absence of the env override (a CI shell that
    exports the flag must not distort the default), and the escape hatch."""
    monkeypatch.delenv(_FLAG, raising=False)
    from discopt.solver_tuning import SolverTuning

    assert SolverTuning().node_numerical_dual_bound is True
    monkeypatch.setenv(_FLAG, "0")
    assert SolverTuning().node_numerical_dual_bound is False

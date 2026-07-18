"""Regression tests for the #671 float64-intractable-row filter.

hda-class relaxations emit a small set of rows whose coefficients span more
orders than float64 can resolve at the LP feasibility tolerance (hda: 130 rows,
raw spread 2.837e26). Those rows made every float64 LP engine false-fail while
contributing zero root tightness (measured:
``docs/dev/hda-certification-rowfilter-entry-2026-07-18.md``).

Fix (flag ``DISCOPT_RELAX_ROW_FILTER`` / ``SolverTuning.relax_row_filter``,
default OFF): **failure-triggered** — only when a node LP breaks down without a
certified verdict (``numerical``, or a spurious ``infeasible`` with no Farkas
proof) does ``mccormick_lp._solve_at_node_impl`` drop such rows and re-solve
once. Sound by construction — removing relaxation rows yields a superset
feasible region (a valid, weaker outer approximation), so the bound can only
loosen, never falsify. Firing only on failure keeps every already-solving node
byte-identical (its LP is optimal/Farkas-infeasible, so the filter never runs).
"""

import math
import os

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp

_NL_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")
_FLAG = "DISCOPT_RELAX_ROW_FILTER"
_HDA_OPT = -5964.534084  # published MINLPLib global optimum


def test_flag_defaults_off(monkeypatch):
    """Bound-changing lever ships default OFF (Dev-Philosophy #5); ``=1`` enables."""
    monkeypatch.delenv(_FLAG, raising=False)
    from discopt.solver_tuning import SolverTuning

    assert SolverTuning().relax_row_filter is False
    monkeypatch.setenv(_FLAG, "1")
    assert SolverTuning().relax_row_filter is True


class _FakeMilp:
    def __init__(self, a_ub, b_ub):
        self._A_ub = a_ub
        self._b_ub = b_ub


@pytest.mark.parametrize("sparse", [False, True], ids=["dense", "sparse"])
def test_filter_drops_wide_rows_and_keeps_normal_ones(sparse):
    """The helper drops exactly the float64-intractable rows (ratio > 1e6 or a
    coefficient outside [1e-8, 1e8]), preserves normal and empty rows, and keeps
    the container kind."""
    from discopt._jax.milp_relaxation import _filter_unresolvable_rows

    a = np.array(
        [
            [1.0, 2.0, 0.0],  # normal — keep
            [6.3e10, -1.0, 0.0],  # |a| > 1e8 (hda Arrhenius row) — drop
            [1.0, -6.2e-10, 0.0],  # |a| < 1e-8 — drop
            [1e5, 1e-3, 0.0],  # ratio 1e8 > 1e6 — drop
            [0.0, 0.0, 0.0],  # empty — keep (may encode infeasibility)
            [-3.0, 0.5, 7.0],  # normal — keep
        ]
    )
    b = np.array([1.0, 0.0, -6.6e-11, 2.0, -1.0, 3.0])
    milp = _FakeMilp(sp.csr_matrix(a) if sparse else a, b)
    dropped = _filter_unresolvable_rows(milp)
    assert dropped == 3
    kept = sp.csr_matrix(milp._A_ub).toarray()
    assert kept.shape == (3, 3)
    np.testing.assert_array_equal(kept[0], a[0])
    np.testing.assert_array_equal(kept[1], a[4])
    np.testing.assert_array_equal(kept[2], a[5])
    np.testing.assert_array_equal(milp._b_ub, [1.0, -1.0, 3.0])
    assert sp.issparse(milp._A_ub) == sparse, "container kind must be preserved"


def test_filter_noop_on_well_conditioned_matrix():
    """A relaxation with no wide rows is untouched (byte-identical object data)."""
    from discopt._jax.milp_relaxation import _filter_unresolvable_rows

    a = np.array([[1.0, -2.0], [3.5, 0.25]])
    b = np.array([1.0, 2.0])
    milp = _FakeMilp(a.copy(), b.copy())
    assert _filter_unresolvable_rows(milp) == 0
    np.testing.assert_array_equal(np.asarray(milp._A_ub), a)
    np.testing.assert_array_equal(milp._b_ub, b)


@pytest.mark.slow
def test_hda_certifies_a_tight_bound_with_the_filter(monkeypatch):
    """End-to-end: with the filter ON, hda's node LPs solve cleanly and the
    reported dual bound is the tight root-relaxation value (≈ −6.47e4 or better
    via branching) instead of candidate A's −1.80e10 — while staying sound."""
    monkeypatch.setenv(_FLAG, "1")
    r = dm.from_nl(os.path.join(_NL_DATA, "hda.nl")).solve(time_limit=60)
    assert r.bound is not None and math.isfinite(r.bound), f"no finite bound: {r.bound}"
    # Sound: never above the published optimum.
    assert r.bound <= _HDA_OPT + 1e-2, f"UNSOUND: bound {r.bound:.6g} > opt {_HDA_OPT}"
    # Tight: at or above the true root McCormick value (−64675.25 − slack); far
    # above candidate A's −1.80e10 floor.
    assert r.bound >= -7e4, f"bound {r.bound:.6g} not the tight root value"


@pytest.mark.slow
@pytest.mark.parametrize(
    "name",
    # alan/ex1221: no wide rows. nvs09/bchoco07/beuster/casctanks: the always-on
    # build-time filter LOOSENED these (nvs09 lost its `optimal` certificate) —
    # the failure-triggered filter must be byte-identical on all of them, since
    # their node LPs solve cleanly and the filter never fires.
    ["alan", "ex1221", "nvs09", "bchoco07", "beuster", "casctanks"],
)
def test_failure_triggered_is_byte_identical_on_solving_instances(name, monkeypatch):
    """The failure-triggered filter is byte-identical ON vs OFF on every
    already-solving instance: the un-filtered node LP is optimal/Farkas-infeasible,
    so the filter never fires (it only re-solves a numerically-failed node)."""
    path = os.path.join(_NL_DATA, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name}.nl not vendored")

    monkeypatch.setenv(_FLAG, "0")
    off = dm.from_nl(path).solve(time_limit=20)
    monkeypatch.setenv(_FLAG, "1")
    on = dm.from_nl(path).solve(time_limit=20)

    assert off.status == on.status, f"{name}: status changed {off.status} -> {on.status}"
    assert off.objective == on.objective, f"{name}: objective drifted with the flag"
    assert off.bound == on.bound, f"{name}: bound drifted ({off.bound} -> {on.bound})"

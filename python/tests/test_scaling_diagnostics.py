"""The scaling dynamic ranges must reach a caller.

``compute_equilibration`` has computed ``worst_row_dynamic_range`` and
``worst_col_dynamic_range`` since the Curtis-Reid pass was written, and until
this change nothing carried them anywhere: the ``ScalingPass`` adapter kept
only ``linear_rows_sampled`` (as ``work_units``) and dropped both ranges, so no
caller -- Rust or Python -- could observe the one number that answers "is this
model badly scaled?".

These tests pin the three links in the chain that now exists: the pass delta,
the direct ``PyModelRepr`` accessor, and the once-per-solve user warning. Five
of the seven fail on the tree before the fix. The two that do not are both
"stays quiet" guards — ``test_solve_is_silent_on_a_well_scaled_model`` (a
warning that always fires carries no information, so its complement has to be
pinned too) and ``test_opt_out_silences_the_warning`` — and both pass
vacuously before the change, because no warning exists at all to suppress.
They still belong here: they are what catches the warning becoming
unconditional, or the opt-out being dropped, later.
"""

import warnings

import discopt.modeling as dm
import pytest
from discopt._relax.presolve_pipeline import run_root_presolve
from discopt._rust import model_to_repr
from discopt.solver import _SCALING_WARN_THRESHOLD


def _badly_scaled_model():
    """One row spanning 1e12: ``1e6 x + 1e-6 y <= 1e6``."""
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=1e6)
    y = m.continuous("y", lb=0, ub=1e6)
    m.subject_to(1e6 * x + 1e-6 * y <= 1e6, name="badly_scaled_row")
    m.minimize(-x - y)
    return m


def _well_scaled_model():
    m = dm.Model()
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.subject_to(2.0 * x + 3.0 * y <= 12.0, name="fine_row")
    m.minimize(-x - y)
    return m


def _repr_of(model):
    return model_to_repr(model, getattr(model, "_builder", None))


def test_scaling_pass_delta_carries_the_dynamic_ranges():
    """The presolve pipeline surfaces both ranges, not just the row count."""
    _, stats = run_root_presolve(_repr_of(_badly_scaled_model()), scaling=True)
    sc = stats["scaling"]
    assert sc["linear_rows_sampled"] == 1
    assert sc["worst_row_dynamic_range"] == pytest.approx(1e12, rel=1e-9)
    # Each variable appears in exactly one row, so no column spans anything.
    assert sc["worst_col_dynamic_range"] == pytest.approx(1.0, rel=1e-9)


def test_accessor_reports_ranges_and_names_the_offender():
    diag = _repr_of(_badly_scaled_model()).scaling_diagnostics()
    assert diag["linear_rows_sampled"] == 1
    assert diag["worst_row_dynamic_range"] == pytest.approx(1e12, rel=1e-9)
    assert diag["worst_row_index"] == 0
    # A range without a name is not actionable; the name is the deliverable.
    assert diag["worst_row_name"] == "badly_scaled_row"
    assert diag["worst_col_name"] == "x"


def test_accessor_is_quiet_on_a_well_scaled_model():
    diag = _repr_of(_well_scaled_model()).scaling_diagnostics()
    assert diag["linear_rows_sampled"] == 1
    assert diag["worst_row_dynamic_range"] < _SCALING_WARN_THRESHOLD
    assert diag["worst_col_dynamic_range"] < _SCALING_WARN_THRESHOLD


def test_accessor_does_not_mutate_the_model():
    """It is a diagnostic: the bounds it read must be untouched afterwards."""
    model = _badly_scaled_model()
    rep = _repr_of(model)
    before = [(v.lb.copy(), v.ub.copy()) for v in model._variables]
    rep.scaling_diagnostics()
    after = [(v.lb.copy(), v.ub.copy()) for v in model._variables]
    assert len(before) == len(after) == 2
    for (lo0, hi0), (lo1, hi1) in zip(before, after):
        assert (lo0 == lo1).all()
        assert (hi0 == hi1).all()


@pytest.mark.smoke
def test_solve_warns_about_a_badly_scaled_model():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _badly_scaled_model().solve(time_limit=10)
    hits = [str(w.message) for w in caught if "Badly scaled model" in str(w.message)]
    assert hits, f"no scaling warning; got {[str(w.message) for w in caught]}"
    # The warning has to name the row and quote the number, or the user cannot
    # act on it.
    assert "badly_scaled_row" in hits[0]
    assert "1e+12" in hits[0]


@pytest.mark.smoke
def test_opt_out_silences_the_warning(monkeypatch):
    """The documented ``=0`` escape hatch actually escapes.

    The check pays a whole extra ``model_to_repr`` (it runs before any repr
    exists), so the opt-out is the one thing standing between a caller in a loop
    and a cost they did not ask for. An opt-out that does not opt out is worse
    than none, because the docstring promises it.
    """
    monkeypatch.setenv("DISCOPT_SCALING_DIAGNOSTICS", "0")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _badly_scaled_model().solve(time_limit=10)
    hits = [str(w.message) for w in caught if "Badly scaled model" in str(w.message)]
    assert not hits, f"opt-out ignored: {hits}"


@pytest.mark.smoke
def test_solve_is_silent_on_a_well_scaled_model():
    """The complement: a warning that always fires carries no information."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _well_scaled_model().solve(time_limit=10)
    hits = [str(w.message) for w in caught if "Badly scaled model" in str(w.message)]
    assert not hits, f"spurious scaling warning: {hits}"

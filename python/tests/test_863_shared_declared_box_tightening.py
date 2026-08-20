"""#863: the declared-box nonlinear tightening must run ONCE per solve, not twice.

``solve_model`` calls ``_check_finite_bounds`` and
``_detect_nonlinear_bound_infeasibility`` back-to-back on an *unmodified* model, and
each used to run the whole ``tighten_nonlinear_bounds`` pass over the same
``flat_variable_bounds(model)`` box. Measured on ``watercontamination0202``
(106,711 vars / 107,209 rows) the two runs took 39.98 s and 39.78 s and produced
bit-identical ``tightened_lb`` / ``tightened_ub`` / ``stats`` — 40 s of pure repeat
work against a 30 s budget.

Two things are tested, because sharing the result is only sound if the pass really is
repeatable on an unmodified model:

1. *Repeatability* — back-to-back runs on the same model agree bit-for-bit
   (i.e. no rule mutates the model or its cached metadata). This is the premise; if it
   ever fails, sharing is wrong and the caller must go back to two calls.
2. *Sharing* — a solve runs it exactly once on the declared box.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

from pathlib import Path  # noqa: E402

import discopt._relax.nonlinear_bound_tightening as nbt  # noqa: E402
import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt.solver import (  # noqa: E402
    _check_finite_bounds,
    _declared_box_tightening,
    _detect_nonlinear_bound_infeasibility,
)

_NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"


def _unbounded_model():
    """A model whose declared box has infinite bounds, so ``_check_finite_bounds``
    does NOT take its early return and really does want the tightening result."""
    m = dm.Model("unbounded")
    x = m.continuous("x")  # no lb/ub => infinite declared bounds
    y = m.continuous("y", lb=0.0, ub=4.0)
    b = m.binary("b")
    m.minimize(x * x + y)
    m.subject_to(x - y - b == 0)
    m.subject_to(y + b <= 4)
    return m


def test_declared_box_tightening_is_repeatable_on_an_unmodified_model():
    """The soundness premise for sharing: the pass does not mutate the model."""
    m = _unbounded_model()
    lb, ub = flat_variable_bounds(m)
    first = _declared_box_tightening(m)
    assert first is not None
    # The model must be unchanged, so re-reading the box gives the same input.
    lb2, ub2 = flat_variable_bounds(m)
    assert np.array_equal(lb, lb2, equal_nan=True)
    assert np.array_equal(ub, ub2, equal_nan=True)
    second = _declared_box_tightening(m)
    assert second is not None
    assert np.array_equal(first[0], second[0], equal_nan=True)
    assert np.array_equal(first[1], second[1], equal_nan=True)
    assert first[2] == second[2]


@pytest.mark.parametrize("name", ["alan", "ex1221", "nvs01", "st_miqp1", "st_e11"])
def test_declared_box_tightening_is_repeatable_on_corpus_instances(name):
    """Same premise, on real .nl instances rather than a hand-built model."""
    path = _NL_DIR / f"{name}.nl"
    if not path.exists():
        pytest.skip(f"{name}.nl not in the in-repo corpus")
    m = dm.from_nl(str(path))
    first = _declared_box_tightening(m)
    second = _declared_box_tightening(m)
    assert (first is None) == (second is None)
    if first is None:
        return
    assert np.array_equal(first[0], second[0], equal_nan=True)
    assert np.array_equal(first[1], second[1], equal_nan=True)
    assert first[2] == second[2]


def _count_declared_box_runs(monkeypatch, model):
    """Wrap ``tighten_nonlinear_bounds`` and count the runs whose input box is the
    model's *declared* box — the ones the two solve_model helpers issue."""
    declared_lb, declared_ub = flat_variable_bounds(model)
    real = nbt.tighten_nonlinear_bounds
    calls: list[int] = []

    def _counting(m, flat_lb, flat_ub, *args, **kwargs):
        lb = np.asarray(flat_lb, dtype=np.float64)
        ub = np.asarray(flat_ub, dtype=np.float64)
        if (
            lb.shape == declared_lb.shape
            and np.array_equal(lb, declared_lb, equal_nan=True)
            and np.array_equal(ub, declared_ub, equal_nan=True)
        ):
            calls.append(1)
        return real(m, flat_lb, flat_ub, *args, **kwargs)

    monkeypatch.setattr(nbt, "tighten_nonlinear_bounds", _counting)
    return calls


def test_both_consumers_share_a_single_pass(monkeypatch):
    """The fix: one pass feeds both helpers.

    Before #863 this was 2 — ``_check_finite_bounds`` and
    ``_detect_nonlinear_bound_infeasibility`` each ran the pass themselves.
    """
    m = _unbounded_model()
    calls = _count_declared_box_runs(monkeypatch, m)

    shared = _declared_box_tightening(m)
    with pytest.warns(UserWarning, match="very large or infinite declared bounds"):
        _check_finite_bounds(m, shared)
    assert _detect_nonlinear_bound_infeasibility(m, shared) is None

    assert len(calls) == 1, (
        f"the declared-box tightening ran {len(calls)} times for the two consumers; "
        "it must run once and be shared"
    )


def test_helpers_still_work_standalone(monkeypatch):
    """Passing no tightening keeps each helper self-sufficient (they are public-ish
    entry points used outside solve_model, e.g. test_amp_integration)."""
    m = _unbounded_model()
    calls = _count_declared_box_runs(monkeypatch, m)
    with pytest.warns(UserWarning, match="very large or infinite declared bounds"):
        _check_finite_bounds(m)
    assert len(calls) == 1
    assert _detect_nonlinear_bound_infeasibility(m) is None
    assert len(calls) == 2


def test_solve_runs_the_declared_box_pass_once(monkeypatch):
    """End to end: a whole solve builds the shared declared-box tightening once.

    Counted on ``_declared_box_tightening`` itself rather than on
    ``tighten_nonlinear_bounds``, because ``solve_model`` legitimately runs the pass
    a second time with a RESTRICTED rule set (``PeriodicVariableBoundRule`` +
    ``FunctionDomainBoundRule``) on the same box, and that one is not duplicate work
    — it computes something different and writes it back to the model.

    If either consumer stops taking the shared result this count goes to 2 or 3.
    """
    import discopt.solver as _solver

    m = _unbounded_model()
    real = _solver._declared_box_tightening
    calls: list[int] = []

    # ``*a, **kw`` rather than ``(model)``: ``solve_model`` now also passes the pass's
    # ``deadline`` (#875), and a fixed-arity spy would turn a plumbing change into a
    # TypeError inside the solve rather than a count mismatch here.
    def _counting(*a, **kw):
        calls.append(1)
        return real(*a, **kw)

    monkeypatch.setattr(_solver, "_declared_box_tightening", _counting)
    # A bare ``.solve()`` again, and deliberately so. This was briefly pinned to
    # ``solver="bb"`` because the pass sat downstream of the solver-family
    # dispatch and the #1059 auto-route therefore skipped it entirely -- the
    # count was 0, not 2. The pass now runs *before* the dispatch, so every
    # family gets it and the shared-once property this test pins is once again a
    # property of ``solve()`` rather than of one path through it.
    with pytest.warns(UserWarning, match="very large or infinite declared bounds"):
        m.solve(time_limit=30)
    assert len(calls) == 1, (
        f"solve() built the declared-box tightening {len(calls)} times (expected 1)"
    )


def test_infeasibility_proof_is_still_reported(monkeypatch):
    """The shared result must still carry an infeasibility proof through to
    ``_detect_nonlinear_bound_infeasibility`` — the reason that helper exists."""
    m = _unbounded_model()

    class _Stats:
        infeasible = True
        infeasibility_reason = "synthetic empty interval"
        n_tightened = 0
        applied_rules = ()

    lb, ub = flat_variable_bounds(m)
    monkeypatch.setattr(nbt, "tighten_nonlinear_bounds", lambda *a, **k: (lb, ub, _Stats()))
    shared = _declared_box_tightening(m)
    assert _detect_nonlinear_bound_infeasibility(m, shared) == "synthetic empty interval"
    # and _check_finite_bounds must stay silent rather than warn about the box
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _check_finite_bounds(m, shared)


def test_a_raising_pass_degrades_to_no_information(monkeypatch):
    """A failing tightening must not break the solve: both helpers treat ``None``
    as 'no information', which is what each of their own ``except`` blocks did."""
    m = _unbounded_model()

    def _boom(*a, **k):
        raise RuntimeError("synthetic tightening failure")

    monkeypatch.setattr(nbt, "tighten_nonlinear_bounds", _boom)
    assert _declared_box_tightening(m) is None
    assert _detect_nonlinear_bound_infeasibility(m, None) is None
    with pytest.warns(UserWarning, match="very large or infinite declared bounds"):
        _check_finite_bounds(m, None)

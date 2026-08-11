"""#917: the #844 LP-spatial reserve is no longer forfeited when the primary finds
an incumbent.

``Model.solve`` deducts ``0.35 * time_limit`` from the caller's budget for every
model the #844 no-incumbent fallback could serve, and spends it in exactly one
case — the primary returned nothing. A primary that *does* find an incumbent and
then hits its reduced deadline therefore forfeits 35% of the stated limit: nobody
spends it, and the caller is told ``time_limit`` at 65% of what they asked for.

Measured on the in-repo corpus at a 60 s budget (19 in-scope instances, isolated
subprocesses, ``scratchpad/issue917_entry_panel_T60.json``): 15 certify inside the
reduced budget, 1 (nvs24) is the #844 no-incumbent case the reserve exists for, and
3 (nvs17/nvs19/nvs23) stop at ~39 s holding an incumbent with 21 s discarded.
``nvs18`` certifies at 38.9 s of the 39 s primary budget — 0.1 s of margin — so the
reserve is a latent certification regression on the family, not only lost wall.

The fix hands the slice back to the primary as an extension it may take **only
while holding an incumbent**. That state is precisely the one in which the fallback
has nothing to contribute, so the #844 path keeps its exact budgets and ordering.
"""

from __future__ import annotations

import os
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.deadline import deadline_scope, get_deadline  # noqa: E402
from discopt.constants import SENTINEL_THRESHOLD  # noqa: E402
from discopt.modeling.core import _lp_spatial_reserve_extension_enabled  # noqa: E402
from discopt.solver import (  # noqa: E402
    _extend_budget_for_incumbent,
    _tree_has_incumbent,
    solve_model_accepted_kwargs,
)

_SENTINEL = 10 * SENTINEL_THRESHOLD  # a sentinel-magnitude "value", not a real primal


class _FakeTree:
    """Minimal stand-in exposing only the ``stats()`` surface these guards read."""

    def __init__(self, incumbent_value):
        self._v = incumbent_value

    def stats(self):
        return {} if self._v is None else {"incumbent_value": self._v}


class _FakeModel:
    pass


@pytest.fixture
def ext(monkeypatch):
    def _set(value):
        if value is None:
            monkeypatch.delenv("DISCOPT_LP_SPATIAL_RESERVE_EXTENSION", raising=False)
        else:
            monkeypatch.setenv("DISCOPT_LP_SPATIAL_RESERVE_EXTENSION", value)

    return _set


# ── the flag ────────────────────────────────────────────────────────────────


def test_flag_defaults_on(ext):
    """Graduated, retracted, then RE-graduated — default ON.

    The first panel's net-positive case was three cells, and #919 (native spatial
    kernel default-ON) voided it: the kernel certified all three in 2.4-5.4 s, so the
    extension never fired and a re-run showed 0 firings in 133 cells. The kernel then
    got its own reclaim point (``SpatialTreeConfig.incumbent_time_extension``), and the
    same 133-cell panel re-run passes both bars far more broadly: 27 firings,
    ``cert_regressions=0 lost_incumbents=0 lost_bound=0 looser_bound=0 unsound=0``, and
    **all 27 firing cells improve the bound** — six from no bound at all. nvs17@60 s
    closes onto its own incumbent; nvs19/nvs23/nvs24 at 60 s tighten 19-40%.

    ``=0`` remains the opt-out."""
    ext(None)
    assert _lp_spatial_reserve_extension_enabled() is True


@pytest.mark.parametrize(
    "value,expected", [("1", True), ("0", False), ("off", False), ("", False), ("true", True)]
)
def test_flag_parses(ext, value, expected):
    ext(value)
    assert _lp_spatial_reserve_extension_enabled() is expected


# ── "does the tree hold an incumbent" ───────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        (None, False),  # key absent entirely
        (float("inf"), False),  # the tree's "no incumbent yet" value
        (float("nan"), False),
        (_SENTINEL, False),  # sentinel magnitude is not a real incumbent
        (-_SENTINEL, False),
        (0.0, True),
        (-585.2, True),
    ],
)
def test_tree_has_incumbent(value, expected):
    """A bare ``is not None`` would read an empty tree (+inf) as holding a primal and
    hand it the extension the #844 fallback was owed."""
    assert _tree_has_incumbent(_FakeTree(value)) is expected


# ── the extension itself ────────────────────────────────────────────────────


def test_no_extension_without_an_incumbent():
    """The whole safety argument: with no incumbent the search must stop at its
    reduced deadline so the #844 fallback still gets its full reserve."""
    model = _FakeModel()
    assert (
        _extend_budget_for_incumbent(
            model,
            _FakeTree(float("inf")),
            time_limit=39.0,
            extension_s=21.0,
            elapsed=39.0,
            t_start=0.0,
        )
        is None
    )
    assert not hasattr(model, "_solve_deadline"), "declined extension must not touch the deadline"


def test_no_extension_when_nothing_was_reserved():
    """``extension_s=0.0`` is the default on every path, so an untouched caller keeps
    the pre-#917 deadline exactly."""
    assert (
        _extend_budget_for_incumbent(
            _FakeModel(),
            _FakeTree(-585.2),
            time_limit=39.0,
            extension_s=0.0,
            elapsed=39.0,
            t_start=0.0,
        )
        is None
    )


def test_extension_with_an_incumbent_returns_the_full_budget():
    model = _FakeModel()
    extended = _extend_budget_for_incumbent(
        model,
        _FakeTree(-585.2),
        time_limit=39.0,
        extension_s=21.0,
        elapsed=39.0,
        t_start=100.0,
    )
    assert extended == pytest.approx(60.0)
    # #654: the phase-gating stash must move with the budget or the extension buys
    # wall time every bound-strengthening phase then refuses to use.
    assert model._solve_deadline == pytest.approx(160.0)


def test_extension_moves_the_process_global_deadline():
    """#80/#844: the JAX-compiled LP/NLP loops poll the process-global deadline. Left
    at the reduced value it reads as expired for the whole extension — the same stale
    -deadline failure that silently degraded the #844 fallback to its cold path.

    The deadline must be re-armed to the budget REMAINING (``extended - elapsed``)
    measured from now, not shifted by the extension: at the moment it is taken the
    search has already spent ``elapsed`` of the extended limit."""
    with deadline_scope(39.0):
        _extend_budget_for_incumbent(
            _FakeModel(),
            _FakeTree(-585.2),
            time_limit=39.0,
            extension_s=21.0,
            elapsed=39.0,
            t_start=0.0,
        )
        after = get_deadline()
        assert after is not None
        assert after - time.monotonic() == pytest.approx(21.0, abs=0.5)
    assert get_deadline() is None, "the scope must still restore on exit"


def test_extension_leaves_an_unset_global_deadline_unset():
    """A direct ``solve_model`` call installs no global deadline; the extension must
    not invent one (a deadline where there was none can only cut a solve short)."""
    assert get_deadline() is None
    _extend_budget_for_incumbent(
        _FakeModel(),
        _FakeTree(-585.2),
        time_limit=39.0,
        extension_s=21.0,
        elapsed=39.0,
        t_start=0.0,
    )
    assert get_deadline() is None


# ── plumbing through Model.solve ────────────────────────────────────────────


def _pure_integer_min() -> dm.Model:
    """In scope for the #844 fallback: all-integer, MINIMIZE, has a constraint row."""
    m = dm.Model("pure_int_min")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.minimize(3 * x + 5 * y)
    m.subject_to(x + y >= 2)
    return m


def _pure_continuous() -> dm.Model:
    """Out of scope: no integer variable, so no reserve is taken and none is handed
    back."""
    m = dm.Model("pure_cont")
    a = m.continuous("a", lb=0, ub=5)
    b = m.continuous("b", lb=0, ub=5)
    m.minimize(3 * a + b)
    m.subject_to(a + b >= 2)
    return m


def test_solve_model_accepts_the_keyword():
    """``Model.solve`` rejects unknown keywords (M6), so the name must be in the
    accepted set or the forward below would raise."""
    assert "incumbent_time_extension" in solve_model_accepted_kwargs()


@pytest.fixture
def captured(monkeypatch):
    """Capture the kwargs ``Model.solve`` forwards, without running a solve."""
    seen = {}
    import discopt.solver as _solver

    real = _solver.solve_model

    def _spy(model, **kwargs):
        seen.update(kwargs)
        return real(model, **kwargs)

    monkeypatch.setattr(_solver, "solve_model", _spy)
    return seen


def test_in_scope_solve_hands_the_reserve_back_when_enabled(ext, captured):
    ext("1")
    _pure_integer_min().solve(time_limit=40)
    assert captured["time_limit"] == pytest.approx(26.0)  # 65% -- unchanged
    assert captured["incumbent_time_extension"] == pytest.approx(14.0)  # the 35% reserve


def test_in_scope_solve_forfeits_the_reserve_when_disabled(ext, captured):
    """Flag OFF must reproduce the pre-#917 call exactly: 65% primary, no extension."""
    ext("0")
    _pure_integer_min().solve(time_limit=40)
    assert captured["time_limit"] == pytest.approx(26.0)
    assert captured["incumbent_time_extension"] == 0.0


@pytest.mark.parametrize("flag", ["0", "1"])
def test_out_of_scope_solve_never_gets_an_extension(ext, captured, flag):
    """No reserve was taken, so there is nothing to hand back — under either flag."""
    ext(flag)
    _pure_continuous().solve(time_limit=40)
    assert captured["time_limit"] == pytest.approx(40.0)
    assert captured["incumbent_time_extension"] == 0.0


@pytest.mark.parametrize("flag", ["0", "1"])
def test_result_is_unchanged_under_either_flag(ext, flag):
    """The extension only ever adds time to a search that already holds an incumbent,
    so it can never weaken a result. Optimum 6 at x=2, y=0."""
    ext(flag)
    r = _pure_integer_min().solve(time_limit=60)
    assert r.objective == pytest.approx(6.0, abs=1e-6)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-6, "UNSOUND: bound above incumbent"

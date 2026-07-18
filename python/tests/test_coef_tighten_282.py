"""Soundness + behaviour tests for big-M coefficient tightening (issue #282).

The transformation (``DISCOPT_COEF_TIGHTEN``) shrinks big-M coefficients on
binary-indicator rows toward the FBBT activity slack. It must:

  * remove **no** integer-feasible point (feasible-point sampling, both
    directions) — soundness is non-negotiable (CLAUDE.md §1/§5);
  * leave the optimum unchanged (differential);
  * be completely inert when the flag is off (default-OFF opt-in);
  * actually fire (reduce a slack big-M) when the flag is on.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from discopt import Model
from discopt.solvers._root_presolve import (
    coef_tighten_enabled,
    tighten_bigm_coefficients,
)


def _flag(monkeypatch, on: bool) -> None:
    if on:
        monkeypatch.setenv("DISCOPT_COEF_TIGHTEN", "1")
    else:
        monkeypatch.delenv("DISCOPT_COEF_TIGHTEN", raising=False)


def _linear_rows(model: Model):
    """Yield ``(coeffs, const, sense, rhs)`` for every linear constraint."""
    from discopt._jax.problem_classifier import (
        _extract_linear_coefficients,
        _NotLinearError,
    )

    n = len(model._variables)
    for con in model._constraints:
        try:
            coeffs, const = _extract_linear_coefficients(con.body, model, n)
        except _NotLinearError:  # pragma: no cover - all rows here are linear
            continue
        yield np.asarray(coeffs, float), float(const), con.sense, float(con.rhs)


def _feasible(rows, point: np.ndarray, tol: float = 1e-6) -> bool:
    for coeffs, const, sense, rhs in rows:
        act = float(coeffs @ point) + const
        if sense == "<=" and act > rhs + tol:
            return False
        if sense == ">=" and act < rhs - tol:
            return False
        if sense == "==" and abs(act - rhs) > tol:
            return False
    return True


def _build_fixed_charge() -> Model:
    """A small fixed-charge network: 2 flows, 2 openings, slack big-Ms.

    ``flow_i <= 5`` caps each flow, so FBBT bounds ``flow_i`` at 5 while the
    big-M is 40 — a 8x slack that coefficient tightening must reclaim.
    """
    m = Model("fc")
    f0 = m.continuous("f0", lb=0.0, ub=float("inf"))
    f1 = m.continuous("f1", lb=0.0, ub=float("inf"))
    y0 = m.binary("y0")
    y1 = m.binary("y1")
    m.subject_to(f0 - 40.0 * y0 <= 0.0)  # f0 <= 40 y0  (slack big-M)
    m.subject_to(f1 - 40.0 * y1 <= 0.0)  # f1 <= 40 y1
    m.subject_to(f0 <= 5.0)  # capacity => FBBT ub(f0)=5
    m.subject_to(f1 <= 5.0)
    m.subject_to(f0 + f1 >= 3.0)  # demand
    m.maximize(f0 + f1 - 2.0 * y0 - 2.0 * y1)
    return m


def test_flag_default_off(monkeypatch):
    _flag(monkeypatch, False)
    assert coef_tighten_enabled() is False
    m = _build_fixed_charge()
    bodies_before = [c.body for c in m._constraints]
    rhs_before = [c.rhs for c in m._constraints]
    assert tighten_bigm_coefficients(m) == 0
    # Model untouched: same body objects, same rhs.
    assert [c.body for c in m._constraints] == bodies_before
    assert [c.rhs for c in m._constraints] == rhs_before


def test_fires_and_reduces_bigm(monkeypatch):
    _flag(monkeypatch, True)
    assert coef_tighten_enabled() is True
    m = _build_fixed_charge()
    n = tighten_bigm_coefficients(m)
    assert n >= 2  # both fixed-charge rows tightened
    # The big-M on each fixed-charge row must have shrunk from 40 toward 5.
    max_bigm = 0.0
    for coeffs, _const, sense, rhs in _linear_rows(m):
        # rows with a binary coeff of large magnitude
        for c in coeffs:
            if abs(c) > 6.0:  # anything still near 40 would fail
                max_bigm = max(max_bigm, abs(c))
    assert max_bigm <= 6.0 + 1e-6, f"a big-M coefficient survived: {max_bigm}"


def test_no_integer_feasible_point_removed(monkeypatch):
    """Feasible-point sampling: the integer-feasible set is preserved exactly."""
    orig = _build_fixed_charge()
    orig_rows = list(_linear_rows(orig))

    _flag(monkeypatch, True)
    tight = _build_fixed_charge()
    assert tighten_bigm_coefficients(tight) >= 2
    tight_rows = list(_linear_rows(tight))

    # Enumerate binary corners and a fine grid on the two continuous flows.
    grid = np.linspace(0.0, 6.0, 13)  # spans past the ub=5 cap deliberately
    removed = 0
    added = 0
    for y0, y1 in itertools.product((0.0, 1.0), repeat=2):
        for f0 in grid:
            for f1 in grid:
                pt = np.array([f0, f1, y0, y1], float)
                fo = _feasible(orig_rows, pt)
                ft = _feasible(tight_rows, pt)
                if fo and not ft:
                    removed += 1  # a feasible point was cut — UNSOUND
                if ft and not fo:
                    added += 1  # tightened admits a point the original rejected
    assert removed == 0, f"{removed} integer-feasible points removed (unsound)"
    # The tightened rows are a subset of the original feasible region at every
    # integer corner, so no new points either.
    assert added == 0, f"{added} points newly admitted (transformation not equivalent)"


def test_optimum_unchanged(monkeypatch):
    """The strengthened model has the same optimal objective."""
    _flag(monkeypatch, False)
    base = _build_fixed_charge().solve(time_limit=20, gap_tolerance=1e-6)

    _flag(monkeypatch, True)
    m = _build_fixed_charge()
    tighten_bigm_coefficients(m)
    _flag(monkeypatch, False)  # solve itself shouldn't re-tighten
    tuned = m.solve(time_limit=20, gap_tolerance=1e-6)

    assert base.objective is not None and tuned.objective is not None
    assert tuned.objective == pytest.approx(base.objective, abs=1e-4, rel=1e-4)


def test_unbounded_activity_row_skipped(monkeypatch):
    """A row whose non-binary activity is unbounded is left untouched."""
    _flag(monkeypatch, True)
    m = Model("unb")
    x = m.continuous("x", lb=0.0, ub=float("inf"))  # never bounded above
    y = m.binary("y")
    m.subject_to(x - 40.0 * y <= 0.0)  # only bound on x is via this row
    m.maximize(y)
    # FBBT derives ub(x)=40 from the row itself, so M==cap -> no slack to remove.
    n = tighten_bigm_coefficients(m)
    assert n == 0

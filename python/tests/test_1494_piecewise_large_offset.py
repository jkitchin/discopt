"""#1494: ``Model.piecewise`` at a large breakpoint offset.

The sos2/log/disaggregated lowerings used to write the convex-combination row
uncentred, ``x == sum_j b_j lambda_j``. A MILP solver satisfies
``sum(lambda) = 1`` (or ``z in {0, 1}``) only to a tolerance ``eps``; with
``b_0 = 1e6`` that lets ``x`` drift by ``1e6 * eps`` while every row still passes
its feasibility test. Measured on ``6596922``: 6 of 12 fixed-input solves over
breakpoints ``[1e6, 1e6+1, 1e6+2, 1e6+3]`` came back with a certified wrong
optimum or a certified "infeasible". The rows are now centred on ``(b_0, v_0)``.

The oracle is ``np.interp``: with the input fixed at ``x0``, both ``min y`` and
``max y`` must equal ``f(x0)`` exactly (the table *defines* ``f``).
"""

from __future__ import annotations

import re

import numpy as np
import pytest
from discopt import Model

METHODS = ("incremental", "log", "disaggregated", "sos2")

# Shape of the table: a zig-zag, so neither sense is decided by an endpoint.
_SHAPE = np.array([0.0, 5.0, -5.0, 0.0, 3.0])
# Where inside the span the input is fixed, as a fraction of one spacing past a
# breakpoint -- off-breakpoint on purpose, so the weights must straddle a segment.
_X_FRACS = (0.5, 1.5, 2.5, 3.25)


def _fixed_input_value(method, b, v, x0, sense):
    m = Model("pwl_offset")
    x = m.continuous("x", lb=float(b[0]), ub=float(b[-1]))
    y = m.piecewise(x, b, v, method=method, name="y")
    m.subject_to(x == x0)
    (m.minimize if sense == "min" else m.maximize)(y)
    return m.solve(time_limit=30)


def _check(r, expected, method, x0, sense, tol):
    assert r.status == "optimal", (
        f"{method} {sense} at x0={x0!r}: status {r.status} (gap_certified="
        f"{r.gap_certified}), expected optimal {expected!r}"
    )
    assert abs(r.objective - expected) <= tol, (
        f"{method} {sense} at x0={x0!r}: certified={r.gap_certified} objective "
        f"{r.objective!r}, expected {expected!r} (np.interp)"
    )


@pytest.mark.parametrize("method", METHODS)
def test_issue_repro_offset_1e6(method):
    """The exact table from the issue."""
    b = [1e6, 1e6 + 1, 1e6 + 2, 1e6 + 3]
    v = [0.0, 5.0, -5.0, 0.0]
    n = 0
    for x0, sense in [(1e6 + 0.5, "min"), (1e6 + 1.5, "max"), (1e6 + 2.5, "max")]:
        r = _fixed_input_value(method, b, v, x0, sense)
        _check(r, float(np.interp(x0, b, v)), method, x0, sense, 1e-5)
        n += 1
    assert n == 3


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("offset", [1e3, 1e4, 1e5, 1e6, 1e7])
def test_offset_sweep_against_interp(method, offset):
    """Offsets 1e3..1e7 x spacings x min/max x off-breakpoint inputs."""
    n = 0
    for h in (0.3, 1.0, 3.0, 10.0):
        b = offset + h * np.arange(_SHAPE.size)
        for frac in _X_FRACS:
            x0 = float(offset + frac * h)
            expected = float(np.interp(x0, b, _SHAPE))
            for sense in ("min", "max"):
                r = _fixed_input_value(method, b, _SHAPE, x0, sense)
                _check(r, expected, method, x0, sense, 1e-5)
                n += 1
    assert n == 4 * len(_X_FRACS) * 2


@pytest.mark.parametrize("method", METHODS)
def test_value_offset_is_centred_too(method):
    """Large *values* as well as large breakpoints: the y row is centred on v_0."""
    offset = 1e6
    b = offset + np.arange(_SHAPE.size)
    v = 1e6 + _SHAPE
    n = 0
    for frac in _X_FRACS:
        x0 = float(offset + frac)
        expected = float(np.interp(x0, b, v))
        for sense in ("min", "max"):
            r = _fixed_input_value(method, b, v, x0, sense)
            _check(r, expected, method, x0, sense, 1e-5 + 1e-10 * abs(expected))
            n += 1
    assert n == len(_X_FRACS) * 2


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("sense", ["min", "max"])
def test_free_input_extremes(method, sense):
    """Input free on the span: the optimum is the table's extreme value."""
    b = 1e6 + 0.3 * np.arange(_SHAPE.size)
    m = Model("pwl_offset_free")
    x = m.continuous("x", lb=float(b[0]), ub=float(b[-1]))
    y = m.piecewise(x, b, _SHAPE, method=method, name="y")
    (m.minimize if sense == "min" else m.maximize)(y)
    r = m.solve(time_limit=30)
    expected = float(_SHAPE.min() if sense == "min" else _SHAPE.max())
    assert r.status == "optimal"
    assert abs(r.objective - expected) <= 1e-5
    # The reported argmax/argmin must reproduce the objective through the table.
    xv = float(np.asarray(r.x["x"]).ravel()[0])
    assert abs(float(np.interp(xv, b, _SHAPE)) - r.objective) <= 1e-4


def test_lowered_rows_are_centred():
    """Structural check: no lowered row carries a breakpoint-sized coefficient."""
    offset = 1e6
    b = offset + 0.5 * np.arange(_SHAPE.size)  # span 2; 0.5 so no coefficient is 1
    n = 0
    for method in METHODS:
        m = Model("pwl_rows")
        x = m.continuous("x", lb=float(b[0]), ub=float(b[-1]))
        m.piecewise(x, b, _SHAPE, method=method, name="y")
        for con in m._constraints:
            if not str(getattr(con, "name", "")).endswith("_x"):
                continue
            s = str(con.body)
            coeffs = [abs(float(t)) for t in re.findall(r"(-?[0-9.]+(?:e[+-]?[0-9]+)?) \* ", s)]
            # Every weight coefficient is a breakpoint *difference* (<= span = 2).
            assert coeffs and max(coeffs) <= 2.0, (method, s)
            n += 1
    assert n == len(METHODS)


def test_value_reference_avoids_rounding_sized_coefficients():
    """Centring must not turn two samples equal up to rounding into a ~1e-16
    coefficient (HiGHS drops it and the verified route refuses the model)."""
    from discopt.modeling._piecewise import _TINY_COEF, _value_reference

    n = 0
    for v in (
        [0.1 + 0.2, 1.0, 0.3],  # v_0 and v_2 differ by 5.6e-17
        [np.cos(-2.0), np.cos(-1.0), 1.0, np.cos(1.0), np.cos(2.0 + 1e-15)],
        list(1e6 + np.array([0.0, 5.0, -5.0, 0.0, 3.0])),
    ):
        c = _value_reference(v)
        d = np.asarray(v) - c
        assert np.all((d == 0.0) | (np.abs(d) >= _TINY_COEF)), (v, c, d)
        n += 1
    # Every sample has a distinct near-duplicate: no sample qualifies, so the row
    # stays uncentred (c = 0), exactly the pre-#1494 row.
    assert _value_reference([0.3, 0.1 + 0.2, 0.3, 0.1 + 0.2]) == 0.0
    assert n == 3


@pytest.mark.parametrize("method", METHODS)
def test_near_duplicate_values_still_solve(method):
    """A table whose first and last values agree up to rounding solves (it used
    to, uncentred; naive centring on v_0 made the HiGHS route refuse it)."""
    b = [0.0, 1.0, 2.0, 3.0]
    v = [0.1 + 0.2, 1.0, -1.0, 0.3]
    n = 0
    for x0, sense in [(0.5, "min"), (2.5, "max")]:
        r = _fixed_input_value(method, b, v, x0, sense)
        _check(r, float(np.interp(x0, b, v)), method, x0, sense, 1e-6)
        n += 1
    assert n == 2

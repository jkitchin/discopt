"""``x**0.5`` must recover the same vertical-tangent facet as ``dm.sqrt(x)`` (#1118).

The two spellings are the same function with the same singularity at ``t = 0``, but
the ``singular_tangent`` recovery of #1111/#1115 could act only on the named atom. The
power path built its derivative by bare exponentiation, and ``p * (0.0 ** (p - 1.0))``
RAISES ``ZeroDivisionError`` for ``p < 1`` — an ``ArithmeticError``, caught by
``_emit_1d._tangent_row`` one branch ABOVE the recovery — so the facet stayed dropped
and the flag's behavior depended on how the user spelled the square root.

``_pow_deriv`` evaluates the ``t == 0`` limit explicitly (``+inf`` for ``0 < p < 1``,
the shape ``_dsqrt`` already had) so control reaches the recovery unchanged.

What this file pins, in the order CLAUDE.md §5 asks for on a bound-CHANGING path:

1. **Spelling parity** — ``x**0.5`` and ``dm.sqrt(x)`` produce identical rows in BOTH
   flag states, which is the issue's actual complaint.
2. **Flag-OFF byte identity** and non-interference with the powers that have no
   vertical tangent (integer ``p``, ``p > 1``, ``p <= 0``).
3. **Soundness** — no point of the graph is cut, sampled densely to the singularity.
4. **Differential bound** — ON >= OFF and ON <= the true box optimum, over a sweep of
   objective directions, with a counter proving the facet actually moved something.
5. The **lazy** placement (#1115, the default inside the flag) registers a spec for a
   power atom too, not just for the named ones.
6. The native spatial kernel keeps DECLINING a zero-touching fractional-power box, in
   both flag states — ``mccormick_patch::univariate_rows`` regenerates an endpoint
   tangent unconditionally and would write an ``inf`` slope into a node LP.
"""

from __future__ import annotations

import math
import os

import discopt.modeling as dm
import discopt.solver_tuning as solver_tuning
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.spatial_producer import build_spatial_kernel_spec
from discopt._relax.uniform_relax import _pow_deriv, build_uniform_relaxation
from scipy.optimize import linprog

pytestmark = [pytest.mark.relaxation]

FLAG = "DISCOPT_SINGULAR_TANGENT"
LAZY = "DISCOPT_SINGULAR_TANGENT_LAZY"

#: Fractional powers whose derivative diverges at 0, on a box that reaches it.
#: ``p`` spans an order of magnitude on both sides of ``0.5`` so nothing here can be
#: a sqrt special case, and two box widths pin scale-freeness.
SINGULAR_POWERS = [
    ("x**0.5 [0,4]", 0.5, 0.0, 4.0),
    ("x**0.25 [0,4]", 0.25, 0.0, 4.0),
    ("x**0.75 [0,1e-3]", 0.75, 0.0, 1e-3),
    ("x**(1/3) [0,8]", 1.0 / 3.0, 0.0, 8.0),
]

#: Powers with NO vertical tangent on a zero-touching box: ``f'`` is finite (``p > 1``)
#: or ``f`` itself diverges (``p <= 0``), so the recovery must decline both.
REGULAR_POWERS = [
    ("x**2 [0,4]", 2.0, 0.0, 4.0),
    ("x**3 [0,4]", 3.0, 0.0, 4.0),
    ("x**1.5 [0,4]", 1.5, 0.0, 4.0),
    ("x**-1 [0,4]", -1.0, 0.0, 4.0),
    ("x**0.5 [1,4]", 0.5, 1.0, 4.0),  # fractional, but away from the singularity
]


@pytest.fixture
def flag():
    """Set/restore the flag in EAGER mode (see the #1111 file's fixture)."""
    prev, prev_lazy = os.environ.get(FLAG), os.environ.get(LAZY)

    def _set(value: str | None):
        if value is None:
            os.environ.pop(FLAG, None)
            os.environ.pop(LAZY, None)
        else:
            os.environ[FLAG] = value
            os.environ[LAZY] = "0"

    yield _set
    for key, val in ((FLAG, prev), (LAZY, prev_lazy)):
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _pow_model(p: float, lo: float, hi: float, obj_coeffs=(0.0, -1.0)):
    """``y == x**p`` over ``x in [lo,hi]``, minimizing ``a*x + b*y``."""
    m = dm.Model()
    span = max(abs(lo), abs(hi), 1.0)
    x = m.continuous("x", lb=lo, ub=hi)
    y = m.continuous("y", lb=-1e3 * span, ub=1e3 * span)
    m.subject_to(y == x**p)
    m.minimize(obj_coeffs[0] * x + obj_coeffs[1] * y)
    return m


def _sqrt_model(lo: float, hi: float, obj_coeffs=(0.0, -1.0)):
    m = dm.Model()
    span = max(abs(lo), abs(hi), 1.0)
    x = m.continuous("x", lb=lo, ub=hi)
    y = m.continuous("y", lb=-1e3 * span, ub=1e3 * span)
    m.subject_to(y == dm.sqrt(x))
    m.minimize(obj_coeffs[0] * x + obj_coeffs[1] * y)
    return m


def _rows(model):
    rel = build_uniform_relaxation(model)
    A = sp.csr_matrix(rel.model._A_ub, dtype=float)
    A.sort_indices()
    return rel, A.toarray(), np.asarray(rel.model._b_ub, dtype=float).ravel()


def _lp_bound(model):
    rel = build_uniform_relaxation(model)
    M = rel.model
    bnds = [
        (float(lo) if np.isfinite(lo) else None, float(hi) if np.isfinite(hi) else None)
        for lo, hi in np.asarray(M._bounds, dtype=float)
    ]
    res = linprog(
        np.asarray(M._c, dtype=float).ravel(),
        A_ub=sp.csr_matrix(M._A_ub),
        b_ub=np.asarray(M._b_ub, dtype=float).ravel(),
        bounds=bnds,
        method="highs",
    )
    assert res.status == 0, res.message
    return float(res.fun)


def _graph_point(rel, t: float, fval: float) -> np.ndarray:
    """Lifted point for the graph point ``(t, f(t))`` of a single-atom model.

    The power path registers no ``univariate_atom_specs`` entry, so the aux column is
    located structurally instead: a one-atom ``y == f(x)`` model lifts to ``[x, y,
    atom_aux, residual_aux]`` (verified against the identical ``sqrt`` build), and the
    residual aux is pinned to 0 by its own two rows."""
    z = np.zeros(len(rel.model._bounds), dtype=float)
    assert z.shape[0] == 4, f"unexpected lifted layout with {z.shape[0]} columns"
    z[0] = t
    z[1] = fval
    z[2] = fval
    return z


# --------------------------------------------------------------------------- #
# 0. The derivative itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("p", [0.5, 0.25, 0.75, 1.0 / 3.0])
def test_pow_deriv_returns_the_limit_at_zero_instead_of_raising(p):
    assert _pow_deriv(p)(0.0) == math.inf
    # ...and is bit-identical to the bare exponentiation everywhere else.
    for t in (1e-8, 1e-3, 0.25, 1.0, 3.0, 1e6):
        assert _pow_deriv(p)(t) == p * (t ** (p - 1.0))


@pytest.mark.parametrize("p", [-1.0, -0.5, 0.0])
def test_pow_deriv_declines_a_nonfinite_f(p):
    """``p <= 0``: ``f(0)`` itself diverges, so the recovery must NOT be offered a
    finite ``f`` with a divergent ``f'`` — ``nan`` keeps ``_finite(g)`` the decliner."""
    assert math.isnan(_pow_deriv(p)(0.0))


# --------------------------------------------------------------------------- #
# 1. Spelling parity — the issue's actual complaint
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state", ["0", "1"])
def test_pow_and_sqrt_are_the_same_relaxation(flag, state):
    flag(state)
    _rp, a_pow, b_pow = _rows(_pow_model(0.5, 0.0, 4.0))
    _rs, a_sqrt, b_sqrt = _rows(_sqrt_model(0.0, 4.0))
    assert a_pow.shape == a_sqrt.shape, (
        f"x**0.5 emitted {a_pow.shape[0]} rows vs sqrt(x) {a_sqrt.shape[0]} with the "
        f"flag {state} — the facet depends on how the user spelled it (#1118)"
    )
    # Same rows, not the same bits: ``np.sqrt(t)`` and ``t**0.5`` differ by an ULP
    # at some points, so the two builds' coefficients agree to rounding, and the
    # sparsity pattern agrees exactly.
    assert np.array_equal(a_pow != 0.0, a_sqrt != 0.0)
    assert np.allclose(a_pow, a_sqrt, rtol=1e-12, atol=1e-12)
    assert np.allclose(b_pow, b_sqrt, rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# 2. Flag-OFF identity and the facet count with the flag ON
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,p,lo,hi", SINGULAR_POWERS + REGULAR_POWERS)
def test_flag_off_is_unchanged_by_the_new_code(flag, label, p, lo, hi):
    flag(None)
    _r, a_unset, b_unset = _rows(_pow_model(p, lo, hi))
    flag("0")
    _r, a_off, b_off = _rows(_pow_model(p, lo, hi))
    assert np.array_equal(a_unset, a_off), label
    assert np.array_equal(b_unset, b_off), label


@pytest.mark.parametrize("label,p,lo,hi", SINGULAR_POWERS)
def test_singular_facet_is_recovered(flag, label, p, lo, hi):
    flag("0")
    _r0, a0, _b0 = _rows(_pow_model(p, lo, hi))
    flag("1")
    _r1, a1, _b1 = _rows(_pow_model(p, lo, hi))
    assert a1.shape[0] == a0.shape[0] + 1, f"{label}: {a0.shape[0]} -> {a1.shape[0]} rows"
    assert np.isfinite(a1).all(), f"{label}: non-finite coefficient in the recovered row"


@pytest.mark.parametrize("label,p,lo,hi", REGULAR_POWERS)
def test_powers_without_a_vertical_tangent_are_untouched(flag, label, p, lo, hi):
    """``p > 1`` (finite ``f'(0)``), integer powers, ``p <= 0`` (``f`` itself
    divergent) and a fractional power away from 0 must be byte-identical ON vs OFF —
    the recovery is for a finite ``f`` with a divergent ``f'`` only."""
    flag("0")
    _r0, a0, b0 = _rows(_pow_model(p, lo, hi))
    flag("1")
    _r1, a1, b1 = _rows(_pow_model(p, lo, hi))
    assert np.array_equal(a0, a1), f"{label}: the flag changed a non-singular power"
    assert np.array_equal(b0, b1), label


# --------------------------------------------------------------------------- #
# 3. Soundness — no graph point is cut
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,p,lo,hi", SINGULAR_POWERS)
def test_no_graph_point_is_cut(flag, label, p, lo, hi):
    flag("1")
    rel = build_uniform_relaxation(_pow_model(p, lo, hi))
    A = sp.csr_matrix(rel.model._A_ub, dtype=float)
    b = np.asarray(rel.model._b_ub, dtype=float).ravel()

    ts = [lo, hi, 0.5 * (lo + hi)]
    ts += list(np.linspace(lo, hi, 200))
    for k in range(1, 25):  # cluster at the singular endpoint, where the facet lives
        d = 0.5 * 2.0**-k
        ts += [lo + d * (hi - lo), hi - d * (hi - lo)]
    checked = 0
    for t in ts:
        if not (lo <= t <= hi):
            continue
        fval = float(t) ** p
        if not np.isfinite(fval):
            continue
        z = _graph_point(rel, t, fval)
        resid = A @ z - b
        viol = float(np.max(resid / np.maximum(1.0, np.abs(b))))
        assert viol <= 1e-9, f"{label}: graph point t={t!r} cut by {viol:.3e}"
        checked += 1
    assert checked >= 200, f"{label}: only {checked} graph points evaluated"


# --------------------------------------------------------------------------- #
# 4. Differential bound test on fixed boxes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,p,lo,hi", SINGULAR_POWERS)
def test_bound_tightens_and_never_crosses(flag, label, p, lo, hi):
    grid = np.linspace(lo, hi, 200001)
    fv = grid**p
    xs = max(hi - lo, 1e-300)
    fs = max(abs(float(fv[-1]) - float(fv[0])), 1e-300)

    compared = improved = 0
    for k in range(32):
        th = 2.0 * np.pi * k / 32.0
        cx, cy = float(np.cos(th)) / xs, float(np.sin(th)) / fs
        flag("0")
        off = _lp_bound(_pow_model(p, lo, hi, (cx, cy)))
        flag("1")
        on = _lp_bound(_pow_model(p, lo, hi, (cx, cy)))
        true_min = float(np.min(cx * grid + cy * fv))
        assert on >= off - 1e-9 * (1.0 + abs(off)), (
            f"{label} dir={k}: bound LOOSENED {off:.12g} -> {on:.12g}"
        )
        assert on <= true_min + 1e-6 * (1.0 + abs(true_min)), (
            f"{label} dir={k}: bound {on:.12g} crossed the true optimum {true_min:.12g}"
        )
        compared += 1
        if on > off + 1e-9 * (1.0 + abs(off)):
            improved += 1

    assert compared == 32, f"{label}: only {compared} directions compared"
    assert improved > 0, f"{label}: the recovered facet never moved the bound (no-op)"


# --------------------------------------------------------------------------- #
# 5. The lazy placement (#1115) sees power atoms too
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,p,lo,hi", SINGULAR_POWERS)
def test_lazy_placement_registers_a_spec_for_a_power_atom(label, p, lo, hi):
    tuning = solver_tuning.current().replace(singular_tangent=True, singular_tangent_lazy=True)
    token = solver_tuning.set_current(tuning)
    try:
        rel = build_uniform_relaxation(_pow_model(p, lo, hi))
        specs = list(rel.singular_tangent_specs)
    finally:
        solver_tuning.reset_current(token)
    assert len(specs) == 1, f"{label}: lazy mode registered {len(specs)} specs, expected 1"
    assert specs[0].edge == -1, f"{label}: the singular endpoint is lo, so edge = -1"
    assert not math.isfinite(specs[0].fp(lo)), f"{label}: the spec's f' must diverge at lo"


# --------------------------------------------------------------------------- #
# 6. The native kernel must keep declining a zero-touching fractional power
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state", ["0", "1"])
def test_native_kernel_declines_zero_touching_power(flag, state):
    """The Rust ``mccormick_patch::univariate_rows`` regenerates an endpoint tangent
    unconditionally; an ``inf`` slope in a node LP is the landmine #1113 closed for
    ``sqrt``. A fractional power reaches the kernel through no term family at all, so
    the producer must decline it — asserted rather than assumed, because the recovery
    now emits a row on this path where it used to emit none."""
    flag(state)
    assert build_spatial_kernel_spec(_pow_model(0.5, 0.0, 4.0)) is None
    assert build_spatial_kernel_spec(_pow_model(0.5, 1.0, 4.0)) is None

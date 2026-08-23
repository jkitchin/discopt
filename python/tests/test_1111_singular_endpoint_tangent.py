"""Vertical-tangent recovery in the 1-D envelope (issue #1111).

``uniform_relax._emit_1d`` places tangents at ``lo``, the box midpoint and ``hi``.
Where ``f`` is finite at an endpoint but ``f'`` DIVERGES there — ``sqrt`` at 0,
``asin``/``acos`` at ±1, ``acosh`` at 1 — ``_tangent_row`` used to return without
emitting, silently dropping the facet and leaving the envelope one-sided on that
side. Sound, but loose, and ``t = 0`` is often the interesting point rather than
an edge case (the conical-intersection radical of #1111 vanishes exactly at the
solution).

``DISCOPT_SINGULAR_TANGENT=1`` re-anchors the dropped facet at an interior point
``lo + delta*width`` chosen from a geometric ladder, capped relative to the box's
own slope scale. This file locks the four properties that gate a bound-CHANGING
change (CLAUDE.md §5):

1. **Flag-OFF is byte-identical.** Both on a box that never hits the singular
   case and on one that does.
2. **Soundness — no feasible point is cut.** Every row of the flag-ON relaxation
   holds at the exact lifted graph point ``(t, f(t))``, densely sampled over the
   box including the singular endpoint itself.
3. **Differential bound.** The flag-ON LP bound is ``>=`` the flag-OFF bound
   (structural: the path only ADDS rows) and still ``<=`` the true optimum over
   the same fixed box.
4. **No spurious ``RuntimeWarning``.** The old ``0.5/np.sqrt(t)`` raised
   ``divide by zero encountered in scalar divide`` on every affected solve.

Plus the native-kernel guard: ``mccormick_patch::univariate_rows`` regenerates
the sqrt tangent at ``t_lo`` unconditionally, which is ``inf``/``NaN`` at
``t_lo == 0``, so ``build_spatial_kernel_spec`` must decline a zero-touching sqrt
box in BOTH flag states.
"""

from __future__ import annotations

import os
import warnings

import discopt.modeling as dm
import discopt.solver_tuning as solver_tuning
import numpy as np
import pytest
import scipy.sparse as sp
from discopt._relax.spatial_producer import build_spatial_kernel_spec
from discopt._relax.uniform_relax import build_uniform_relaxation
from scipy.optimize import linprog

# ``relaxation`` (not ``correctness``): these are theorem-style property tests over
# the envelope rows themselves, fast, and NOT deselected by the default addopts —
# so the soundness and differential-bound checks run on every PR.
pytestmark = [pytest.mark.relaxation]

FLAG = "DISCOPT_SINGULAR_TANGENT"
LAZY = "DISCOPT_SINGULAR_TANGENT_LAZY"

#: ``(label, modeling atom, numpy f, lo, hi)`` — every ``_UNIVARIATE_FN`` entry
#: whose derivative diverges at an endpoint the domain guard admits, on a box that
#: reaches the singularity. Two ``sqrt`` boxes six orders of magnitude apart pin
#: the scale-freeness of the ladder. Generic atoms/boxes, no named instance.
SINGULAR_CASES = [
    ("sqrt[0,4]", dm.sqrt, np.sqrt, 0.0, 4.0),
    ("sqrt[0,1e-3]", dm.sqrt, np.sqrt, 0.0, 1e-3),
    ("sqrt[0,1e6]", dm.sqrt, np.sqrt, 0.0, 1e6),
    ("asin[0.2,1]", dm.asin, np.arcsin, 0.2, 1.0),
    ("acos[0.2,1]", dm.acos, np.arccos, 0.2, 1.0),
]

#: Boxes where no endpoint derivative diverges — the flag must be a no-op here.
REGULAR_CASES = [
    ("sqrt[1,4]", dm.sqrt, np.sqrt, 1.0, 4.0),
    ("asin[0.1,0.9]", dm.asin, np.arcsin, 0.1, 0.9),
    ("log[0.5,3]", dm.log, np.log, 0.5, 3.0),
]


@pytest.fixture
def flag():
    """Set/restore ``DISCOPT_SINGULAR_TANGENT`` around a test, in EAGER mode.

    This file tests the #1111 eager anchor: the facet emitted as a relaxation row
    at a fixed geometric ladder point. #1115 made LP-point separation the default
    placement, so enabling recovery here must also pin
    ``DISCOPT_SINGULAR_TANGENT_LAZY=0`` — otherwise these tests would look for a
    row that the lazy path deliberately does not emit until the LP asks for it,
    and would report the eager anchor as broken when it is merely not selected.
    """
    prev = os.environ.get(FLAG)
    prev_lazy = os.environ.get(LAZY)

    def _set(value: str | None):
        if value is None:
            os.environ.pop(FLAG, None)
            os.environ.pop(LAZY, None)
        else:
            os.environ[FLAG] = value
            os.environ[LAZY] = "0"

    yield _set
    if prev is None:
        os.environ.pop(FLAG, None)
    else:
        os.environ[FLAG] = prev
    if prev_lazy is None:
        os.environ.pop(LAZY, None)
    else:
        os.environ[LAZY] = prev_lazy


def _atom_model(atom, lo, hi, obj_coeffs=(0.0, -1.0)):
    """``y == atom(x)`` over ``x in [lo,hi]``, minimizing ``a*x + b*y``."""
    m = dm.Model()
    span = max(abs(lo), abs(hi), 1.0)
    x = m.continuous("x", lb=lo, ub=hi)
    y = m.continuous("y", lb=-1e3 * span, ub=1e3 * span)
    m.subject_to(y == atom(x))
    m.minimize(obj_coeffs[0] * x + obj_coeffs[1] * y)
    return m


def _relax(model, lo, hi):
    return build_uniform_relaxation(model, box=(np.array([lo, -1e12]), np.array([hi, 1e12])))


def _rows(model, lo, hi):
    """``(A, b)`` of the relaxation LP, densified for exact comparison."""
    rel = build_uniform_relaxation(model)
    A = sp.csr_matrix(rel.model._A_ub, dtype=float)
    A.sort_indices()
    return rel, A.toarray(), np.asarray(rel.model._b_ub, dtype=float).ravel()


def _lp_bound(model):
    """Root LP bound of the uniform relaxation over the model's declared box."""
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
    """Exact lifted point for the graph point ``(t, f(t))``: originals, then the
    univariate aux column set to its true value. Any other aux column would make
    the sample meaningless, so assert there is exactly one."""
    specs = list(rel.univariate_atom_specs)
    assert len(specs) == 1, f"expected a single univariate atom, got {specs}"
    _fname, w, _var, _coeff, _cst = specs[0]
    z = np.zeros(len(rel.model._bounds), dtype=float)
    z[0] = t
    z[1] = fval
    z[int(w)] = fval
    return z


# --------------------------------------------------------------------------- #
# 1. Flag-OFF byte identity, and the flag being a no-op on a regular box
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,atom,fnp,lo,hi", SINGULAR_CASES + REGULAR_CASES)
def test_flag_off_is_unchanged_by_the_new_code(flag, label, atom, fnp, lo, hi):
    """With the flag unset the emitted rows are exactly those with it set to 0 —
    i.e. the new code path is entered only through the flag."""
    flag(None)
    _r0, a0, b0 = _rows(_atom_model(atom, lo, hi), lo, hi)
    flag("0")
    _r1, a1, b1 = _rows(_atom_model(atom, lo, hi), lo, hi)
    assert np.array_equal(a0, a1) and np.array_equal(b0, b1)


@pytest.mark.parametrize("label,atom,fnp,lo,hi", REGULAR_CASES)
def test_regular_box_is_byte_identical_on_and_off(flag, label, atom, fnp, lo, hi):
    """No endpoint derivative diverges, so the ladder never runs: the polytope is
    bit-for-bit the same. This is what makes the flag free on the corpus."""
    flag("0")
    _r0, a0, b0 = _rows(_atom_model(atom, lo, hi), lo, hi)
    flag("1")
    _r1, a1, b1 = _rows(_atom_model(atom, lo, hi), lo, hi)
    assert np.array_equal(a0, a1) and np.array_equal(b0, b1)


# --------------------------------------------------------------------------- #
# 2. The facet is actually recovered
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,atom,fnp,lo,hi", SINGULAR_CASES)
def test_singular_facet_is_recovered(flag, label, atom, fnp, lo, hi):
    """Flag ON emits exactly one more row than flag OFF on a box whose endpoint
    derivative diverges — the facet that used to be dropped."""
    flag("0")
    _r0, a0, _b0 = _rows(_atom_model(atom, lo, hi), lo, hi)
    flag("1")
    _r1, a1, _b1 = _rows(_atom_model(atom, lo, hi), lo, hi)
    assert a1.shape[0] == a0.shape[0] + 1, f"{label}: {a0.shape[0]} -> {a1.shape[0]} rows"
    assert np.isfinite(a1).all(), f"{label}: non-finite coefficient in the recovered row"


# --------------------------------------------------------------------------- #
# 3. Soundness — the recovered facet cuts no point of the graph
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,atom,fnp,lo,hi", SINGULAR_CASES)
def test_no_graph_point_is_cut(flag, label, atom, fnp, lo, hi):
    """Every row of the flag-ON relaxation holds at ``(t, f(t))`` for a dense
    sample of the box, INCLUDING the singular endpoint itself. A tangent taken on
    the wrong side of the curvature, or at a point outside the box, fails here."""
    flag("1")
    model = _atom_model(atom, lo, hi)
    rel = build_uniform_relaxation(model)
    A = sp.csr_matrix(rel.model._A_ub, dtype=float)
    b = np.asarray(rel.model._b_ub, dtype=float).ravel()

    # Endpoints, a uniform grid, and a geometric cluster hugging each endpoint —
    # the recovered facet lives within ~1e-4 of the corner, so a uniform grid alone
    # would never probe where it is tightest.
    ts = [lo, hi, 0.5 * (lo + hi)]
    ts += list(np.linspace(lo, hi, 200))
    for k in range(1, 25):
        d = 0.5 * 2.0**-k
        ts += [lo + d * (hi - lo), hi - d * (hi - lo)]
    checked = 0
    worst = -np.inf
    for t in ts:
        if not (lo <= t <= hi):
            continue
        fval = float(fnp(t))
        if not np.isfinite(fval):
            continue
        z = _graph_point(rel, t, fval)
        resid = A @ z - b
        scale = np.maximum(1.0, np.abs(b))
        viol = float(np.max(resid / scale))
        worst = max(worst, viol)
        assert viol <= 1e-9, f"{label}: graph point t={t!r} cut by {viol:.3e}"
        checked += 1
    assert checked >= 200, f"{label}: only {checked} graph points evaluated"


# --------------------------------------------------------------------------- #
# 4. Differential bound test on fixed boxes
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,atom,fnp,lo,hi", SINGULAR_CASES)
def test_bound_tightens_and_never_crosses(flag, label, atom, fnp, lo, hi):
    """``bound_ON >= bound_OFF`` AND ``bound_ON <= true box optimum``, swept over
    the whole circle of linear objective directions.

    Directions rather than a hand-picked objective: which one the recovered facet
    binds under depends on the atom's curvature and on which endpoint is singular,
    and picking one per atom would be tuning the test to the answer. Coefficients
    are normalized by the box's own ``x``- and ``f``-ranges so the sweep means the
    same thing on a box of width 1e-3 and one of width 1e6.

    The true optimum comes from a dense grid ON THE GRAPH, which is an UPPER bound
    on the true minimum — so ``bound <= grid_min`` is the conservative form of the
    no-crossing check.

    The sweep also has to show the change is not a no-op: at least one direction
    per case must move the bound STRICTLY. Without that counter a do-nothing
    implementation passes every other assertion in this file.
    """
    grid = np.linspace(lo, hi, 200001)
    fv = fnp(grid)
    xs = max(hi - lo, 1e-300)
    fs = max(abs(float(fv[-1]) - float(fv[0])), 1e-300)

    compared = 0
    improved = 0
    for k in range(32):
        th = 2.0 * np.pi * k / 32.0
        cx, cy = float(np.cos(th)) / xs, float(np.sin(th)) / fs
        flag("0")
        off = _lp_bound(_atom_model(atom, lo, hi, (cx, cy)))
        flag("1")
        on = _lp_bound(_atom_model(atom, lo, hi, (cx, cy)))
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
# 5. No spurious RuntimeWarning (issue #1111 motivation 1)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("label,atom,fnp,lo,hi", SINGULAR_CASES)
@pytest.mark.parametrize("state", ["0", "1"])
def test_no_divide_by_zero_warning(flag, label, atom, fnp, lo, hi, state):
    """Building the envelope on a box that reaches the vertical tangent must not
    emit ``RuntimeWarning: divide by zero``. Fails before #1111 in BOTH flag
    states — the warning came from the ``_UNIVARIATE_FN`` derivative itself."""
    flag(state)
    model = _atom_model(atom, lo, hi)
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        build_uniform_relaxation(model)
    bad = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert not bad, f"{label}: {[str(w.message) for w in bad]}"


# --------------------------------------------------------------------------- #
# 6. Native-kernel guard
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("state", ["0", "1"])
def test_native_kernel_declines_zero_touching_sqrt(flag, state):
    """``mccormick_patch::univariate_rows`` regenerates the tangent at ``t_lo``
    unconditionally, so at ``t_lo == 0`` it would write ``f'(0) = inf`` and
    ``intercept = NaN`` into the node LP. The producer must decline that box in
    both flag states — and must still ADMIT an equivalent box away from zero, or
    the guard is over-broad."""
    m0 = _atom_model(dm.sqrt, 0.0, 4.0)
    m1 = _atom_model(dm.sqrt, 1.0, 4.0)
    flag(state)
    assert build_spatial_kernel_spec(m0) is None, "zero-touching sqrt box must decline"
    assert build_spatial_kernel_spec(m1) is not None, "guard must not reject a regular box"


# --------------------------------------------------------------------------- #
# 7. The flag is a SolverTuning field, not a bare environ read
# --------------------------------------------------------------------------- #
def test_programmatic_tuning_enables_recovery_without_the_env_var(flag):
    """``SolverTuning(singular_tangent=True)`` must enable the facet with the
    environment variable UNSET.

    This is the whole point of the migration: a bare ``os.environ`` read is
    invisible to a programmatic tuning and cannot be pinned for the duration of a
    solve. Asserting the env var absent is what makes the test meaningful — with
    the pre-migration code the ON arm below is byte-identical to the OFF arm and
    the row-count assertion fails."""
    flag(None)
    assert FLAG not in os.environ, "the env var must be unset for this test to mean anything"

    model = _atom_model(dm.sqrt, 0.0, 4.0)
    _, a_off, b_off = _rows(model, 0.0, 4.0)

    tuning = solver_tuning.current().replace(singular_tangent=True, singular_tangent_lazy=False)
    token = solver_tuning.set_current(tuning)
    try:
        assert solver_tuning.current().singular_tangent is True
        _, a_on, b_on = _rows(model, 0.0, 4.0)
    finally:
        solver_tuning.reset_current(token)

    assert a_on.shape[0] == a_off.shape[0] + 1, (
        f"programmatic tuning emitted {a_on.shape[0]} rows vs {a_off.shape[0]} off — "
        "the recovered facet is exactly one added row"
    )
    assert b_on.shape[0] == a_on.shape[0]
    # And the context is restored: back to the OFF row count.
    _, a_after, _ = _rows(model, 0.0, 4.0)
    assert a_after.shape[0] == a_off.shape[0]


@pytest.mark.parametrize("bad", [0.0, -1.0, float("inf"), float("nan")])
def test_kappa_must_be_finite_and_positive(bad):
    """An invalid kappa RAISES rather than being silently swapped for the default.

    The pre-migration helper caught ``ValueError`` and returned 100.0, so a typo in
    a measurement sweep produced a run at the default cap that read as a run at the
    requested one (§3: refuse loudly; §7: never swallow in an instrument)."""
    with pytest.raises(ValueError, match="singular_tangent_kappa"):
        solver_tuning.current().replace(singular_tangent_kappa=bad)


def test_kappa_flows_from_tuning_to_the_ladder(flag):
    """The ladder must read kappa from the active tuning, not a module constant.

    A kappa small enough to reject every ladder point leaves the facet dropped, so
    the ON arm collapses to the OFF row count — an observable that can only move if
    the value is genuinely threaded through."""
    flag(None)
    model = _atom_model(dm.sqrt, 0.0, 4.0)
    _, a_off, _ = _rows(model, 0.0, 4.0)

    counts = {}
    for kappa in (100.0, 1e-9):
        tuning = solver_tuning.current().replace(
            singular_tangent=True, singular_tangent_lazy=False, singular_tangent_kappa=kappa
        )
        token = solver_tuning.set_current(tuning)
        try:
            _, a_on, _ = _rows(model, 0.0, 4.0)
        finally:
            solver_tuning.reset_current(token)
        counts[kappa] = a_on.shape[0]

    assert counts[100.0] == a_off.shape[0] + 1, "default kappa must admit the facet"
    assert counts[1e-9] == a_off.shape[0], (
        f"kappa=1e-9 must reject every ladder point (got {counts[1e-9]} rows vs "
        f"{a_off.shape[0]} off) — if it does not, kappa is not being read from the tuning"
    )


def _endpoint_binding_sqrt_model(a: float):
    """``min a*x - sqrt(x)`` over ``x in [0,4]`` — a box whose LP optimum sits AT the
    singular endpoint, so the recovered facet is actually load-bearing.

    The generic single-atom models above put their optimum at the far end of the box,
    where the dropped facet never binds and every cap reads as neutral. Generic atom,
    generic box, no named instance."""
    m = dm.Model()
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=-1e3, ub=1e3)
    m.subject_to(y == dm.sqrt(x))
    m.minimize(a * x - y)
    return m


@pytest.mark.parametrize("a", [3.0, 1.0])
def test_an_uncapped_anchor_emits_a_row_that_buys_nothing(flag, a):
    """The slope cap is load-bearing: without it the facet is worthless (#1111).

    The ladder walks from its SMALLEST ``delta`` outward, so an effectively infinite
    ``kappa`` accepts the first rung, ``0.5*8**-20 ~ 5.5e-19``. That anchor is
    numerically degenerate — the row is still emitted, but it constrains nothing, so
    the feature would pay a row on every singular atom for no bound.

    This is the cap's real justification, and it replaces the one originally written
    into ``SolverTuning.singular_tangent_kappa`` (outward-rounding slack), which is
    ~1e-14 here and six orders too small to explain anything. Asserting the ROW IS
    STILL THERE matters as much as the bound: it distinguishes "the anchor degenerated"
    from "the ladder declined", which would produce the same bound for a different
    reason (§6)."""
    flag(None)
    model = _endpoint_binding_sqrt_model(a)
    off = _lp_bound(model)

    def _at(kappa):
        tuning = solver_tuning.current().replace(
            singular_tangent=True, singular_tangent_lazy=False, singular_tangent_kappa=kappa
        )
        token = solver_tuning.set_current(tuning)
        try:
            rel = build_uniform_relaxation(model)
            n_rows = sp.csr_matrix(rel.model._A_ub).shape[0]
            return _lp_bound(model), n_rows
        finally:
            solver_tuning.reset_current(token)

    n_off = sp.csr_matrix(_relax_rows(model)).shape[0]
    uncapped, n_uncapped = _at(1e12)
    capped, n_capped = _at(100.0)

    assert n_uncapped == n_off + 1, (
        f"kappa=1e12 must still EMIT the facet (got {n_uncapped} rows vs {n_off} off) — "
        "otherwise this test is measuring a declined ladder, not a degenerate anchor"
    )
    assert n_capped == n_off + 1, f"kappa=100 must emit the facet ({n_capped} vs {n_off})"
    assert uncapped - off < 1e-6, (
        f"an uncapped anchor bought {uncapped - off:.3g} of root bound; it is supposed to "
        "be degenerate. If this fires the ladder's smallest rung is no longer degenerate "
        "and the cap's justification needs re-measuring"
    )
    assert capped - off > 1e-3, (
        f"kappa=100 bought only {capped - off:.3g}; the capped anchor is supposed to be "
        "worth orders more than the uncapped one"
    )
    assert capped > uncapped, f"capped {capped} must beat uncapped {uncapped}"


def _relax_rows(model):
    """Flag-OFF constraint matrix of ``model``'s relaxation (row count reference)."""
    token = solver_tuning.set_current(solver_tuning.current().replace(singular_tangent=False))
    try:
        return build_uniform_relaxation(model).model._A_ub
    finally:
        solver_tuning.reset_current(token)

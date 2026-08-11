"""#861 — the monomial envelope no longer declines when the root box spans zero.

``IncrementalMcCormickLP`` gated EVERY monomial ``x_i**p`` on a sign-definite root
box, so a model whose integers straddle zero (ball_mk2_30: 30 integers, one
sign-mixed "thin shell" row) declined the whole incremental structure — and with it
the cuts, the feasibility pump and, under ``require_incremental=True``, the entire
LP-per-node solve, which then returned no incumbent at all.

The gate was wider than the mathematics. ``x**p`` for EVEN ``p`` has
``f'' = p(p-1)x**(p-2) >= 0`` on all of R, so it is convex across a sign change and
the cold build emits the *same* 4-row secant/tangent envelope in every sign regime —
measured on ``build_milp_relaxation`` before this change:

    p=2 box=[-2,3]  -> 4 rows      p=2 box=[1,3] -> 4 rows    p=2 box=[-3,-1] -> 4 rows
    p=4 box=[-2,3]  -> 4 rows      p=4 box=[1,3] -> 4 rows    p=4 box=[-3,-1] -> 4 rows
    p=3 box=[-2,3]  -> 2 rows      p=3 box=[1,3] -> 4 rows    p=3 box=[-3,-1] -> 4 rows

Only the ODD powers change facet COUNT across the sign change (the S-shaped atom's
2-facet hull vs the 4-row envelope), and only those are still unmappable by a fixed
sparsity pattern. So even powers are admitted on any root box; odd powers keep the
sign-definite requirement.

Two things had to be generalized before the gate could move, and both are pinned
here: the aux-column enclosure (the old endpoint ``min``/``max`` assumed monotonicity
and would have FLOORED ``x**2`` above zero on a straddling box — cutting off the true
point ``x=0``), and the validation gate's row comparison (a *pinned* box, reached
whenever integer branching fixes a variable, gets no envelope rows from the cold
build but four exactly-tight — hence vacuous — rows from the fixed-pattern patch).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.incremental_mccormick import IncrementalMcCormickLP, _monomial_aux_bounds
from discopt._relax.lp_spatial_bb import solve_lp_spatial_bb
from discopt._relax.term_classifier import classify_nonlinear_terms


def _monomial_model(p: int, lo: float, hi: float, n: int = 3):
    """``n`` integers over ``[lo,hi]`` carrying bare ``x_i**p`` monomials."""
    m = dm.Model(f"mono_p{p}")
    xs = [m.integer(f"x{i}", lb=lo, ub=hi) for i in range(n)]
    m.minimize(sum(x**p for x in xs))
    m.subject_to(sum(x**p for x in xs) >= 1)
    return m


def _ball_mk2_class(n: int = 30):
    """The ball_mk2_30 class: ``n`` integers whose root boxes straddle zero, a single
    sign-mixed row over all of them, MINIMIZE, optimum 0 at the origin. Named
    instances are gate probes only — this is the *shape* that #861 declined."""
    m = dm.Model("ball_mk2_class")
    xs = [m.integer(f"x{i}", lb=-1, ub=1) for i in range(n)]
    m.minimize(sum(x * x for x in xs))
    shell = sum((-1.0) ** i * x for i, x in enumerate(xs))
    m.subject_to(shell <= 1)
    m.subject_to(shell >= -1)
    return m


def _ball_mk2_real(n: int = 30):
    """Faithful reconstruction of MINLPLib's ``ball_mk2_30`` — the instance #861 was
    actually filed against:

        min  -Σ xᵢ      s.t.  Σ (xᵢ² - 0.995825·xᵢ) ≤ 0,   xᵢ ∈ {-1,0,1}

    This is NOT the same problem as :func:`_ball_mk2_class`, and the difference is the
    point. Every term ``x² - 0.995825x`` is ≥ 0 at an integer ``x`` with equality only
    at ``x=0``, so the origin is the *only* feasible integer point and the optimum is
    0.0 (matching ``minlplib.solu``) — but the objective ``-Σ xᵢ`` pulls the
    relaxation the other way, to fractional ``xᵢ ≈ 0.995825`` where the shell is
    slack. That is the "thin shell": the LP optimum is nowhere near the integer one,
    which is what makes the primal hard here and trivial in ``_ball_mk2_class``
    (whose optimum sits at the origin and is found by the root LP).

    Reconstructing it matters because ``_ball_mk2_class`` alone cannot detect the real
    failure — the #727 RLT lesson in CLAUDE.md: a mechanism validated only on a
    synthetic proxy can be a no-op on the real class.
    """
    m = dm.Model("ball_mk2_real")
    xs = [m.integer(f"x{i}", lb=-1, ub=1) for i in range(n)]
    m.minimize(-sum(xs))
    m.subject_to(sum(x * x - 0.995825 * x for x in xs) <= 0.0)
    return m


def _structure(model):
    return IncrementalMcCormickLP(model, classify_nonlinear_terms(model))


# --------------------------------------------------------------------------- #
# The gate itself
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p", [2, 4, 6])
def test_even_power_monomial_maps_on_a_root_box_spanning_zero(p):
    """The #861 regression: before the fix this raised ``monomial x_0^{p}: root box
    spans zero (unmappable)`` and left ``ok=False``."""
    assert _structure(_monomial_model(p, -2, 2)).ok


@pytest.mark.parametrize("p", [2, 3, 4, 5])
@pytest.mark.parametrize("box", [(0, 3), (-3, 0)])
def test_sign_definite_root_still_maps_for_every_power(p, box):
    """Unchanged behaviour on the regime that already worked."""
    assert _structure(_monomial_model(p, *box)).ok


@pytest.mark.parametrize("p", [3, 5])
def test_odd_power_monomial_still_declines_on_a_root_box_spanning_zero(p):
    """An odd power's envelope switches between the 4-row secant/tangent hull and the
    2-facet S-hull across zero — a facet-COUNT change the fixed sparsity pattern
    cannot express, so it must keep declining (soundly: the caller cold-builds)."""
    assert not _structure(_monomial_model(p, -2, 2)).ok


# --------------------------------------------------------------------------- #
# Aux-column enclosure parity (the soundness prerequisite)
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p", [2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize(
    "box",
    [
        (-2.0, 3.0),
        (1.0, 3.0),
        (-3.0, -1.0),
        (0.0, 2.5),
        (-2.5, 0.0),
        (-1.5, -1.5),
        # Half-infinite and doubly-infinite boxes. These are where a hand-rolled
        # closed form diverges from Interval: `0 * ±inf` is NaN in IEEE, and
        # Interval's per-step outward rounding turns an exact 0 endpoint into a
        # denormal that then multiplies to ±inf rather than to a NaN-to-zero corner.
        # `_monomial_aux_bounds` delegates here rather than reproducing it.
        (-np.inf, 0.0),
        (0.0, np.inf),
        (-np.inf, np.inf),
        (-np.inf, -1.0),
        (1.0, np.inf),
        (-2.0, np.inf),
        (-np.inf, 2.0),
    ],
)
def test_monomial_aux_bounds_match_interval_pow(p, box):
    """``_monomial_aux_bounds`` must reproduce the enclosure the COLD build takes from
    ``Interval.__pow__`` — not merely a sound one — or the two paths describe
    different polytopes. Pins the parity so a change to the interval arithmetic
    surfaces here rather than as a silent bound difference."""
    from discopt._relax.convexity.interval import Interval

    lo, hi = box
    ref = Interval.from_bounds(np.array([lo]), np.array([hi])) ** p
    got = _monomial_aux_bounds(lo, hi, p)
    assert not any(g != g for g in got), f"NaN enclosure on {box}, p={p}: {got}"
    for g, t in zip(got, (float(ref.lo[0]), float(ref.hi[0]))):
        if g == t:  # covers the ±inf endpoints, which approx() cannot compare
            continue
        assert g == pytest.approx(t, rel=1e-9, abs=1e-9)


@pytest.mark.parametrize("p", [3, 4, 5, 6])
def test_unbounded_box_puts_no_nan_anywhere_in_the_node_lp(p):
    """Regression for the NaN family on an unbounded box (caught in review of this
    PR): ``(nan, nan)`` from the ``p >= 3`` aux enclosure, and NaN coefficients from
    the envelope rows.

    The variable below is sign-DEFINITE (``ub <= 0`` → ``_root_sign = -1``), so the
    structure was admitted before *and* after #861 and all six validation boxes are
    finite — the gate never sees the unbounded node box. NaN reaching the LP is worse
    than a loose bound: every comparison against it is ``False``, so ``NaN <=
    incumbent`` can silently disable fathoming. Assert on the WHOLE assembled LP, not
    just the aux bounds — the aux fix alone left 4 NaNs in ``A``/``b``.
    """
    m = dm.Model("halfinf")
    x = m.integer("x", lb=-np.inf, ub=0)
    y = m.integer("y", lb=-5, ub=0)
    m.minimize(x**p + y**p)
    m.subject_to(x + y >= -8)
    inc = _structure(m)
    assert inc.ok
    A, b, bounds = inc.assemble(np.array([-np.inf, -5.0]), np.array([0.0, 0.0]))
    assert not np.isnan(bounds).any(), f"NaN in aux bounds for p={p}: {bounds}"
    assert not np.isnan(A.data).any(), f"NaN in constraint matrix for p={p}"
    assert not np.isnan(b).any(), f"NaN in rhs for p={p}"


@pytest.mark.parametrize("p", [2, 3, 4])
def test_unbounded_box_envelope_matches_the_cold_build_emptiness(p):
    """On a non-finite box the cold build emits NO envelope rows (``_emit_1d`` bails
    under ``_finite``), leaving the aux interval bound as the entire relaxation. The
    fixed-pattern patch cannot delete rows, so it must fill them with VACUOUS ones —
    which describes the same polytope. Pins that they carry no coefficients at all."""
    from discopt._relax.incremental_mccormick import _monomial_rows

    for box in [(-np.inf, 0.0), (0.0, np.inf), (-np.inf, np.inf), (1.0, np.inf)]:
        rows = _monomial_rows(box[0], box[1], p)
        assert len(rows) == 4
        for cx, cs, rhs in rows:
            assert (cx, cs, rhs) == (0.0, 0.0, 0.0), f"non-vacuous row on {box}: {rows}"


def test_even_power_aux_floor_admits_the_origin_on_a_straddling_box():
    """The old endpoint-``min`` form returned ``min(li^2, ui^2) = 1`` on ``[-1, 3]``,
    which floors ``x**2`` above its true value at ``x=0`` and cuts off a feasible
    point. The enclosure must reach 0 wherever the box contains 0."""
    lo, hi = _monomial_aux_bounds(-1.0, 3.0, 2)
    assert lo <= 0.0 <= hi
    assert hi == pytest.approx(9.0)


# --------------------------------------------------------------------------- #
# Soundness of the admitted envelope
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("p", [2, 4])
def test_spanning_monomial_envelope_cuts_no_feasible_point(p):
    """Feasible-point sampling: every ``(x_i, x_i**p)`` with ``x_i`` in the node box
    must satisfy every patched envelope row. Run over sub-boxes that straddle zero,
    sit on either side of it, and pin the variable."""
    inc = _structure(_monomial_model(p, -2, 2))
    assert inc.ok
    boxes = [(-2.0, 2.0), (-2.0, 0.0), (0.0, 2.0), (-1.0, 1.0), (1.0, 1.0), (-2.0, -2.0)]
    checked = 0
    for lo, hi in boxes:
        lb = np.full(inc.n, lo)
        ub = np.full(inc.n, hi)
        A, b, bounds = inc.assemble(lb, ub)
        A = A.tocsr()
        for (i, a, pw), rows in inc.mono_rows.items():
            aux_lo, aux_hi = bounds[a]
            for v in np.linspace(lo, hi, 9):
                s = v**pw
                assert aux_lo - 1e-9 <= s <= aux_hi + 1e-9, f"aux bound cuts x={v}"
                for k in rows:
                    lhs = 0.0
                    for t in range(A.indptr[k], A.indptr[k + 1]):
                        col = int(A.indices[t])
                        if col == i:
                            lhs += float(A.data[t]) * v
                        elif col == a:
                            lhs += float(A.data[t]) * s
                    assert lhs <= float(b[k]) + 1e-9, f"row {k} cuts x={v} (p={pw})"
                    checked += 1
    # Guard against the loops silently emptying (a changed mono_rows key, a box list
    # that stops admitting): 6 boxes x 3 monomials x 9 samples x 4 rows.
    assert checked == 648, f"expected 648 row assertions, executed {checked}"


@pytest.mark.parametrize("p", [2, 4])
def test_spanning_monomial_patch_is_bound_neutral_against_the_cold_build(p):
    """The incremental path may change speed, never the bound: the patched LP's
    optimal value must equal the value of the same-flag cold build on the same box,
    including boxes that straddle zero."""
    inc = _structure(_monomial_model(p, -2, 2))
    assert inc.ok
    rng = np.random.default_rng(861)
    compared = 0
    for _ in range(10):
        lb = rng.integers(-2, 3, size=inc.n).astype(float)
        ub = np.array([rng.integers(int(v), 3) for v in lb], dtype=float)
        patched = inc.solve_assembled(*inc.assemble(lb, ub))[0]
        Af, bf, bdf, _, _, _ = inc._full_build(lb, ub)
        cold = inc.solve_assembled(Af, bf, bdf)[0]
        if patched is None or cold is None:
            continue
        assert patched == pytest.approx(cold, rel=1e-9, abs=1e-9)
        compared += 1
    assert compared >= 5, "too few comparable boxes to call this a check"


@pytest.mark.parametrize("p", [2, 4])
def test_spanning_monomial_bound_never_exceeds_the_box_optimum(p):
    """The relaxation is a valid lower bound: on a fixed box its LP value must not
    exceed the true (brute-forced) integer optimum over that box."""
    lo, hi = -2, 2
    n = 3
    model = _monomial_model(p, lo, hi, n=n)
    inc = _structure(model)
    assert inc.ok
    compared = 0
    for box in [(-2, 2), (-2, 0), (0, 2), (-1, 1)]:
        lb = np.full(n, float(box[0]))
        ub = np.full(n, float(box[1]))
        bound = inc.solve_assembled(*inc.assemble(lb, ub))[0]
        if bound is None:
            continue
        true = None
        for pt in itertools.product(range(box[0], box[1] + 1), repeat=n):
            v = np.array(pt, dtype=float)
            if float(np.sum(v**p)) >= 1.0 - 1e-9:
                obj = float(np.sum(v**p))
                true = obj if true is None else min(true, obj)
        if true is not None:
            assert bound <= true + 1e-6, f"bound {bound} above box optimum {true}"
            compared += 1
    # Without this the test degrades to a no-op if every LP returns None or every box
    # turns out infeasible (the sibling bound-neutrality test guards the same way).
    assert compared == 4, (
        f"expected 4 boxes compared against a brute-forced optimum, got {compared}"
    )


# --------------------------------------------------------------------------- #
# The validation gate's vacuous-row filter
# --------------------------------------------------------------------------- #


def test_rowset_drops_only_rows_that_cannot_cut_the_box():
    """The filter that lets a *pinned* box validate must drop exactly the rows whose
    maximum over the box already satisfies them, and keep every row that bites."""
    import scipy.sparse as sp

    A = sp.csr_matrix(np.array([[1.0, 1.0], [1.0, 0.0]]))
    b = np.array([10.0, 0.5])  # row 0 vacuous over the box below, row 1 cuts it
    bounds = np.array([[0.0, 1.0], [0.0, 2.0]])
    unfiltered = IncrementalMcCormickLP._rowset(A, b)
    filtered = IncrementalMcCormickLP._rowset(A, b, bounds)
    assert len(unfiltered) == 2
    assert filtered == [(((0, 1.0),), 0.5)]


def test_pinned_variable_box_validates_for_a_spanning_root():
    """``_validation_boxes`` drives every spanning var through a degenerate
    (``lb==ub``) trial — reachable whenever integer branching fixes a variable. The
    cold build emits no envelope rows at zero width while the fixed pattern must fill
    its four reserved rows, so the structure only validates because those rows are
    exactly tight (vacuous) there. Assert the regime is actually exercised."""
    inc = _structure(_monomial_model(2, -2, 2))
    assert inc.ok
    assert {"span", "degen", "neg"} <= inc._validated_regimes


# --------------------------------------------------------------------------- #
# Gate probe: the class the issue was filed against
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("n", [10, 30])
def test_ball_mk2_class_is_admitted_and_solved_under_require_incremental(n):
    """Before #861 this returned ``None``: the structure declined on
    ``monomial x_0^2: root box spans zero`` and ``require_incremental=True`` (PR #858)
    turned that into "no incumbent". It must now certify the optimum, 0.0."""
    result = solve_lp_spatial_bb(_ball_mk2_class(n), time_limit=60.0, require_incremental=True)
    assert result is not None, "engine still declines the ball_mk2_30 class"
    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.0, abs=1e-6)
    assert result.bound <= result.objective + 1e-6  # certificate invariant


def test_real_ball_mk2_30_relaxation_is_admitted():
    """The REAL instance's relaxation coverage — the gap #861 reports — is closed.

    Fails before the change (``monomial x_0^2: root box spans zero (unmappable)``
    → ``ok=False``), passes after, with all 30 monomials mapped.
    """
    inc = _structure(_ball_mk2_real(30))
    assert inc.ok, "the real ball_mk2_30 relaxation still cannot be built"
    assert len(inc.monomial) == 30
    assert all(inc._root_sign[i] == 0 for (i, _p) in inc.monomial), "expected straddling roots"


def test_real_ball_mk2_30_bound_is_sound_and_no_false_certificate():
    """On the real instance the engine now produces a *bound* where it previously
    produced nothing — and that bound must never cross the 0.0 oracle
    (``minlplib.solu``).

    Deliberately NOT asserted: that an incumbent is found. It is not, at any budget
    tried (see ``test_real_ball_mk2_30_still_finds_no_incumbent``), so #861's stated
    symptom is only partly addressed — this PR closes the relaxation-coverage half.
    The assertions below are written to keep holding if a later primal fix lands.
    """
    result = solve_lp_spatial_bb(_ball_mk2_real(30), time_limit=20.0, require_incremental=True)
    assert result is not None, "engine declines the real ball_mk2_30"
    assert result.bound is not None
    assert result.bound <= 0.0 + 1e-6, f"dual bound {result.bound} crossed the 0.0 oracle"
    if result.objective is not None:
        # If a primal ever appears it must be feasible-valued and certificate-consistent.
        assert result.objective >= 0.0 - 1e-6, "incumbent below the true optimum"
        assert result.bound <= result.objective + 1e-6


@pytest.mark.slow
def test_real_ball_mk2_30_still_finds_no_incumbent():
    """Pins the RESIDUAL so it cannot be mistaken for fixed, and so the day it stops
    being true someone is told.

    Admitting the model was necessary but not sufficient: the relaxation now builds
    and the engine explores thousands of nodes inside its budget, but the thin shell
    means no node LP rounds to the origin, so ``objective`` stays ``None`` — #861's
    "returns no incumbent" persists. Measured on this reconstruction: 791 nodes /
    bound -27.88 at a 20 s budget, 12236 nodes / bound -26.89 at 60 s, both budgets
    honoured. That is primal work (the #844 family), not relaxation coverage.

    XFAIL-shaped on purpose: it PASSES while the residual exists and XPASSes loudly
    if a primal fix lands, at which point delete it and assert the optimum instead.
    """
    result = solve_lp_spatial_bb(_ball_mk2_real(30), time_limit=20.0, require_incremental=True)
    assert result is not None
    if result.objective is not None:  # pragma: no cover - the day this fires, celebrate
        pytest.fail(
            f"ball_mk2_30 now yields an incumbent ({result.objective}) — the #861 "
            "residual is closed; replace this test with an optimality assertion."
        )
    assert result.status == "time_limit"
    assert result.node_count > 0, "admitted but explored no nodes — budget spent elsewhere"


# --------------------------------------------------------------------------- #
# The #844 fallback's dual bound is no longer discarded (raised in review of #873)
# --------------------------------------------------------------------------- #


@pytest.mark.slow
def test_no_incumbent_fallback_still_reports_its_dual_bound():
    """A spent fallback reserve must buy a BOUND even when it buys no primal.

    #861 admits ball_mk2_30 into the LP-per-node engine, so ``Model.solve``'s
    no-incumbent fallback (#844) now runs it instead of declining under
    ``require_incremental``. It spends the whole reserve and finds no incumbent — and
    before this fix the fallback's result was merged into ``result`` only when it
    carried an objective, so a *sound dual bound* was computed and then thrown away:
    the solve reported ``bound=None`` while the fallback had proved ``bound=-27.88``.
    That, not the admission itself, was the real cost behind the review's
    "spends the budget for nothing".

    Asserts properties rather than values, since the bound tightens with the budget:
    a bound exists, it never crosses the 0.0 oracle (``minlplib.solu``), and no
    certificate is claimed without an incumbent. Fails before the fix (``bound is
    None``), passes after.
    """
    result = _ball_mk2_real(30).solve(time_limit=6.0)
    assert result.bound is not None, "the fallback's dual bound was discarded again"
    assert result.bound <= 0.0 + 1e-6, f"dual bound {result.bound} crossed the 0.0 oracle"
    # No incumbent -> no certificate, whatever the bound says.
    if result.objective is None:
        assert not getattr(result, "gap_certified", False)
    else:  # a primal fix landed: the certificate invariant must still hold
        assert result.bound <= result.objective + 1e-6

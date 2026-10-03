"""#1537: a certified answer must not change under a change of variables that leaves
the model identical -- translation ``x = y - c`` and multiplying a row or the
objective by a positive constant.

This file is the standing seeded panel (workstream B) plus the regressions for the
two classes it found on ``1a6556b``:

* **rows x 1e6 withdrew 12/12 correct MILP certificates.** The HiGHS route's own unit
  logical ``a x + s = b`` sat at 2.5e-7 of its row, below the #1295 cap. HiGHS really
  does mis-prune with that column (the #1509 refutation caught it on the #1295-class
  panel), so the logical is now rescaled by an exact power of two instead of excused
  (``lp_milp_highs.logical_column_scales``).
* **the QP route published its bound from the expanded polynomial.** At ``c = 1e6``
  ``(y - c)**2 - 3 (y - c)`` had bound -2.2501220703125 (true -2.25, the ulp of
  1e12); at 3.3e6 / 7.7e6 the certificate was withdrawn. The objective-as-bound is
  now evaluated in declared form (``solver._qp_objective_at_point``).

Truth is exhaustive enumeration over the integer box, independent of discopt. False
certificates and lost certificates are counted separately, as the issue asks: a
lost certificate is not a soundness failure, but it is the measurement that showed
the #1295 guard was scale-variant.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest

NI = 4  # integers in [0, 3]


def _data(seed: int):
    rng = np.random.default_rng(seed)
    A = rng.integers(-4, 5, size=(3, NI)).astype(float)  # noqa: N806
    b = rng.integers(2, 10, size=3).astype(float)
    c = rng.integers(-5, 6, size=NI).astype(float)
    Q = rng.integers(-3, 4, size=(NI, NI)).astype(float)  # noqa: N806
    P = rng.integers(-2, 3, size=NI).astype(float)  # noqa: N806
    return A, b, c, Q, P


def _f(kind: str, x, c, Q, P):  # noqa: N803
    v = sum(c[j] * x[j] for j in range(NI))
    if kind in ("bilinear", "poly"):
        v = v + sum(Q[i, j] * x[i] * x[j] for i in range(NI) for j in range(i + 1, NI))
    if kind == "poly":
        v = v + sum(P[i] * x[i] ** 3 - 2 * x[i] ** 2 for i in range(NI))
    return v


def _build(kind: str, seed: int, off: float = 0.0, rowscale: float = 1.0, objscale: float = 1.0):
    """The same model written in ``y = x + off``, with every row times ``rowscale`` and
    the objective times ``objscale``. All three are exact changes of representation."""
    A, b, c, Q, P = _data(seed)  # noqa: N806
    m = dm.Model(f"{kind}{seed}")
    ys = [m.integer(f"i{k}", lb=off, ub=3 + off) for k in range(NI)]
    x = [v - off for v in ys]
    for r in range(3):
        m.subject_to(rowscale * sum(A[r, j] * x[j] for j in range(NI)) <= rowscale * b[r])
    if kind != "linear":
        m.subject_to(rowscale * (x[0] * x[1] - x[2]) <= rowscale * 4)
    m.minimize(objscale * _f(kind, x, c, Q, P))
    return m


def _oracle(kind: str, seed: int) -> float:
    A, b, c, Q, P = _data(seed)  # noqa: N806
    best = np.inf
    for ii in itertools.product(range(4), repeat=NI):
        x = np.array(ii, dtype=float)
        if np.all(A @ x <= b) and (kind == "linear" or x[0] * x[1] - x[2] <= 4):
            best = min(best, float(_f(kind, x, c, Q, P)))
    return best


TRANSFORMS = {
    "shift1e3": dict(off=1e3),
    "shift1e6": dict(off=1e6),
    "rows1e6": dict(rowscale=1e6),
    "rows1e-3": dict(rowscale=1e-3),
    "obj1e6": dict(objscale=1e6),
}


def _verdict(r, truth: float) -> str:
    tol = 1e-6 + 1e-4 * abs(truth)
    if r.gap_certified:
        if abs(r.objective - truth) > tol or r.bound > truth + tol:
            return "false"
        return "ok"
    return "lost"


@pytest.mark.parametrize("kind", ["linear", "bilinear", "poly"])
def test_invariance_panel(kind):
    """Every transform of every seeded model certifies the enumerated optimum.

    The issue's 12-seed measurement on ``1a6556b``: ``linear`` x ``rows1e6`` lost 12/12, every
    other cell 0 false / 0 lost. This test runs the first 6 seeds: 0 / 0 everywhere. Unmarked on
    purpose -- ~15 s for all three kinds -- so it runs on every PR (workstream B)."""
    compared = 0
    bad = []
    for seed in range(6):
        truth = _oracle(kind, seed)
        assert np.isfinite(truth), "the generator's box always contains a feasible point"
        for name, kw in TRANSFORMS.items():
            scale = kw.get("objscale", 1.0)
            r = _build(kind, seed, **kw).solve(time_limit=30)
            compared += 1
            v = _verdict(r, truth * scale)
            if v != "ok":
                bad.append((seed, name, v, r.status, r.objective, r.bound, truth * scale))
    assert compared == 6 * len(TRANSFORMS), f"panel compared only {compared} models"
    assert not bad, bad


# ── rows x 1e6: the route's own logical column ─────────────────────────────────


def test_row_scaled_milp_keeps_certificate():
    """The issue's lost-certificate cell: seed 7 under rows x 1e6 came back
    ``feasible`` with bound -16.8 (truth -12) because the unit slack tripped #1295."""
    truth = _oracle("linear", 7)
    r = _build("linear", 7, rowscale=1e6).solve(time_limit=30)
    st = r.solver_stats or {}
    assert st.get("route/lp_milp_backend") == 1.0, "HiGHS MILP route not taken"
    assert st.get("milp/logicals_rescaled", 0) >= 1, "the logical was not rescaled"
    assert "milp/decertified_unscalable" not in st
    assert "milp/certificate_refuted" not in st
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(truth, abs=1e-6)


def test_unscaled_milp_is_handed_to_highs_unchanged():
    """A model with no under-scaled logical takes exactly the old path."""
    r = _build("linear", 7).solve(time_limit=30)
    assert "milp/logicals_rescaled" not in (r.solver_stats or {})
    assert r.gap_certified


def test_users_own_tiny_column_still_decertifies():
    """Rescaling is for the route's OWN logicals. A user column with a tiny entry is
    the #1295 class (HiGHS was measured mis-pruning it) and keeps decertifying."""
    m = dm.Model("user_slack")
    x = [m.integer(f"x{j}", lb=0, ub=3) for j in range(3)]
    s = m.continuous("s", lb=0)
    m.subject_to(3 * x[0] + 5 * x[1] + 2 * x[2] + 1e-8 * s == 7)
    m.minimize(-4 * x[0] - 6 * x[1] - 3 * x[2])
    r = m.solve(time_limit=30)
    st = r.solver_stats or {}
    assert st.get("route/lp_milp_backend") == 1.0
    assert "milp/logicals_rescaled" not in st
    assert st.get("milp/decertified_unscalable") == 1.0
    assert not r.gap_certified


def _sf(A, b, c, xl, xu, int_idx=()):  # noqa: N803
    from discopt.solvers.lp_milp_highs import StdForm

    return StdForm.from_arrays(
        np.asarray(c, float),
        np.asarray(A, float),
        np.asarray(b, float),
        np.asarray(xl, float),
        np.asarray(xu, float),
        int_idx=list(int_idx),
    )


def test_logical_column_scales_rules():
    from discopt.solvers.lp_milp_highs import INF, UNSCALABLE_OPEN_RATIO, logical_column_scales

    big = 3e6
    # cols: x0 (struct, int), x1 (struct), s0 logical row0, s1 logical row1 (well scaled),
    # s2 logical row2 with cost, s3 integer logical row3
    A = [  # noqa: N806
        [big, 2 * big, 1.0, 0.0, 0.0, 0.0],
        [1.0, 2.0, 0.0, 1.0, 0.0, 0.0],
        [big, 0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, big, 0.0, 0.0, 0.0, 1.0],
    ]
    c = [1.0, 1.0, 0.0, 0.0, 5.0, 0.0]
    xl = [0, 0, 0, 0, 0, 0]
    xu = [3, 3, INF, INF, INF, INF]
    sf = _sf(A, [1, 1, 1, 1], c, xl, xu, int_idx=[0, 5])
    assert logical_column_scales(sf, None) is None
    assert logical_column_scales(sf, 6) is None
    f = logical_column_scales(sf, 2)
    assert f is not None
    checked = 0
    for j in (0, 1, 3, 4, 5):  # structural, well scaled, costed, integer
        assert f[j] == 1.0, j
        checked += 1
    assert f[2] != 1.0
    assert np.log2(f[2]) == np.round(np.log2(f[2])), "not a power of two"
    assert 2 * big / 2 < f[2] * 1.0 <= 2 * big, "entry must land in (row_max/2, row_max]"
    assert 1.0 / (2 * big) < UNSCALABLE_OPEN_RATIO <= f[2] / (2 * big)
    assert checked == 5


def test_rescaled_solve_maps_back_to_the_callers_variables():
    """``solve_milp_std`` returns ``x`` in ``sf``'s own variables: the logical is
    ``(b - a x) / a_s`` at the original coefficient, not at the scaled one."""
    from discopt.solvers.lp_milp_highs import INF, solve_milp_std

    k = 4e6
    # max 3 x0 + 2 x1  s.t. k (x0 + x1) + s = 4.5 k,  x0, x1 in {0..3}
    A = [[k, k, 1.0]]  # noqa: N806
    sf = _sf(A, [4.5 * k], [-3.0, -2.0, 0.0], [0, 0, 0], [3, 3, INF], int_idx=[0, 1])
    out = solve_milp_std(sf, time_limit=30, gap_tolerance=1e-4, max_nodes=1000, n_struct=2)
    assert out.stats.get("milp/logicals_rescaled") == 1.0
    assert out.status == "optimal" and out.gap_certified
    assert out.objective == pytest.approx(-11.0)  # x = (3, 1)
    assert out.x is not None
    assert out.x[:2] == pytest.approx([3.0, 1.0])
    assert out.x[2] == pytest.approx(0.5 * k, rel=1e-12)
    assert float(np.asarray(sf.A @ out.x).ravel()[0]) == pytest.approx(4.5 * k, rel=1e-12)


# ── the QP route: objective-as-bound in declared form ──────────────────────────


@pytest.mark.parametrize("c0", [0.0, 1e5, 1e6, 1000001.48, 3.3e6, 7.7e6])
def test_translated_qp_bound_is_not_the_expanded_polynomial(c0):
    """``min (y - c0)**2 - 3 (y - c0)`` on ``[c0, c0 + 4]``; the optimum is -2.25.

    Before: bound -2.2501220703125 at 1e6 (error ~ c0**2 * eps) and the certificate
    withdrawn at 1000001.48, 3.3e6 and 7.7e6. The bound must now be accurate to the
    point's own optimality error, not to the expanded polynomial's ``c0**2 * eps``.
    Since #1596 it is a dual bound that charges the backend's unconverged
    complementarity rather than the objective itself, so it sits at or below the
    true optimum. Measured (#1605): -2.250000001002788 at every ``c0`` -- 1.0e-9
    below the optimum, the same at each translation, which is the invariance this
    file is about."""
    m = dm.Model("q")
    y = m.continuous("y", lb=c0, ub=c0 + 4)
    m.minimize((y - c0) ** 2 - 3 * (y - c0))
    r = m.solve(time_limit=20)
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert r.bound <= r.objective
    assert -2.25 - 2e-9 <= r.bound <= -2.25, r.bound
    assert r.objective == pytest.approx(-2.25, abs=1e-8)


def test_qp_declared_form_does_not_hide_a_reformulation_mismatch():
    """Agreement within the expanded form's rounding scale is the only case that
    swaps the value; a genuine difference is left for reconciliation to judge."""
    from discopt.solver import _qp_objective_at_point

    m = dm.Model("q")
    y = m.continuous("y", lb=0, ub=4)
    m.minimize((y - 1) ** 2)
    x = np.array([3.0])  # f = 4
    Q, c, k = np.array([[2.0]]), np.array([-2.0]), 1.0  # noqa: N806
    assert _qp_objective_at_point(m, x, 4.0 + 1e-15, Q, c, k) == 4.0
    assert _qp_objective_at_point(m, x, 4.5, Q, c, k) == 4.5


# ── MIQP-BB: POUNCE node labels and node bounds are not certificates ───────────

_MIQP_OPTIMA = {"st_miqp1": 281.0, "st_miqp2": 2.0, "st_miqp3": -6.0}  # minlplib.solu


@pytest.mark.parametrize("name", sorted(_MIQP_OPTIMA))
def test_translated_miqp_is_never_falsely_certified(tmp_path, name):
    """Found by the corpus invariance panel under ``x = y + 1e6``: ``st_miqp1`` and
    ``st_miqp2`` came back certified ``infeasible`` (POUNCE labelled the feasible
    root ``primal_infeasible`` and the label was pruned on), ``st_miqp3`` certified
    0.0 against -6 (a root bound of f(x) = 25.8 at a drifted IPM point pruned the
    optimum). Every answer must now be honest: no infeasibility claim, a bound on
    the right side of the optimum, and a certified objective only if it is right."""
    from pathlib import Path

    from discopt.modeling.core import from_nl
    from discopt.validation.nl_invariance import translate_nl

    src = Path(__file__).parent / "data" / "minlplib_nl" / f"{name}.nl"
    out = tmp_path / f"{name}_shift.nl"
    out.write_text(translate_nl(src.read_text(encoding="latin-1"), 1e6), encoding="latin-1")
    truth = _MIQP_OPTIMA[name]
    r = from_nl(str(out)).solve(time_limit=30)
    assert r.status != "infeasible", "a feasible model was certified infeasible"
    tol = 1e-6 + 1e-4 * abs(truth)
    if r.bound is not None:
        assert r.bound <= truth + tol, f"bound {r.bound} above the optimum {truth}"
    if r.gap_certified:
        assert r.objective == pytest.approx(truth, abs=tol)
    if r.objective is not None:
        assert r.objective >= truth - tol, f"incumbent {r.objective} below the optimum"


def test_linear_node_verified_empty_needs_a_proof():
    from discopt.solver import _linear_node_verified_empty

    G, h = np.array([[1.0, 1.0]]), np.array([-5.0])  # noqa: N806
    assert _linear_node_verified_empty(G, h, None, None, np.zeros(2), np.ones(2))
    # Feasible, and feasible far from the origin (the st_miqp1 shape): never "empty".
    assert not _linear_node_verified_empty(G, np.array([5.0]), None, None, np.zeros(2), np.ones(2))
    G2 = np.array([[-20.0, -12.0]])  # noqa: N806
    lo = np.full(2, 1e6)
    assert not _linear_node_verified_empty(G2, np.array([-30.0 - 32e6]), None, None, lo, lo + 1.0)


def test_convex_qp_node_lower_bound_is_rigorous_at_a_drifted_point():
    """At the true minimiser the bound is tight; at a drifted 'optimal' point it is
    still at or below the true minimum (f there is not)."""
    from discopt.solver import _convex_qp_node_lower_bound

    # min 6 (u - c)^2 - 3 (v - c)  s.t. -4 (u - c) + (v - c) <= 0, box [c, c+3] x [c, c+12]
    c0 = 1e6
    P = np.diag([12.0, 0.0])  # noqa: N806
    q = np.array([-12.0 * c0, -3.0])
    k = 6.0 * c0 * c0 + 3.0 * c0
    G, h = np.array([[-4.0, 1.0]]), np.array([-3.0 * c0])  # noqa: N806
    lo, hi = np.array([c0, c0]), np.array([c0 + 3.0, c0 + 12.0])
    true_min = -6.0
    exact = np.array([c0 + 1.0, c0 + 4.0])
    b_exact = _convex_qp_node_lower_bound(P, q, k, G, h, None, None, lo, hi, exact, z=[3.0])
    assert b_exact <= true_min + 1e-6
    # Tight up to the rounding margin of the expanded form, whose terms are ~3e13
    # here (the reason workstream D wants affine-argument bounds instead).
    assert b_exact == pytest.approx(true_min, abs=0.1)
    drifted = np.array([c0 + 2.4388538761, c0 + 3.296923424])  # st_miqp3's root point
    f_drift = 0.5 * drifted @ P @ drifted + q @ drifted + k
    assert f_drift > true_min + 10.0, "the drifted point must be visibly non-optimal"
    for z in (None, [0.0], [1e-9]):
        b = _convex_qp_node_lower_bound(P, q, k, G, h, None, None, lo, hi, drifted, z=z)
        assert b <= true_min + 1e-6, (z, b)


# ── nonlinear FBBT: an emptiness proof must survive the arithmetic ─────────────


def test_monotone_equality_touching_ranges_is_not_a_proof():
    """``sqrt(a) == y - c`` with ``a in [d**2, 4]`` (sqrt range [d, 2]) and
    ``y in [c, c + d]``: the ranges touch at ``d = 0.07``. At ``c = 1e6`` the
    computed ``(c + d) - c`` is 0.06999999994877726, 5e-11 short -- the bare 1e-12
    slack read that as an empty intersection and proved the model infeasible (hda's
    shape under the corpus panel's ``x = y + 1e6``)."""
    from discopt._relax.nonlinear_bound_tightening import tighten_nonlinear_bounds
    from discopt.solver import _extract_variable_info

    checked = 0
    for c0 in (0.0, 1e6):
        m = dm.Model("touch")
        d = 0.07
        a = m.continuous("a", lb=d * d, ub=4.0)
        y = m.continuous("y", lb=c0, ub=c0 + d)
        m.subject_to(y - c0 - dm.sqrt(a) == 0)
        m.minimize(y)
        _, lb, ub, _, _ = _extract_variable_info(m)
        _, _, stats = tighten_nonlinear_bounds(m, lb, ub)
        assert not stats.infeasible, (c0, stats.infeasibility_reason)
        checked += 1
    assert checked == 2


@pytest.mark.slow
def test_translated_hda_is_not_certified_infeasible(tmp_path):
    """The corpus panel's hda finding: certified ``infeasible`` under x = y + 1e6
    (MINLPLib optimum -5964.5), from the monotone-equality rule above."""
    from pathlib import Path

    from discopt.modeling.core import from_nl
    from discopt.validation.nl_invariance import translate_nl

    src = Path(__file__).parent / "data" / "minlplib_nl" / "hda.nl"
    out = tmp_path / "hda_shift.nl"
    out.write_text(translate_nl(src.read_text(encoding="latin-1"), 1e6), encoding="latin-1")
    # The contradiction appears only once the root FBBT has run to its full budget,
    # which scales with the time limit; 5 s stops short of it (measured).
    r = from_nl(str(out)).solve(time_limit=30)
    assert r.status != "infeasible", "a feasible model was certified infeasible"


@pytest.mark.slow
@pytest.mark.parametrize("shift", [1e3, 1e6])
def test_translated_nvs09_is_honest(tmp_path, shift):
    """The corpus panel's nvs09 cells raised ``RecursionError`` before #1549 (a product
    of ten shifted factors distributed into 1024 monomials). With #1549 the shifted
    model must answer honestly: never certify a wrong value, never a bound past the
    optimum (minlplib.solu: -43.134336918)."""
    from pathlib import Path

    from discopt.modeling.core import from_nl
    from discopt.validation.nl_invariance import translate_nl

    truth = -43.134336918
    src = Path(__file__).parent / "data" / "minlplib_nl" / "nvs09.nl"
    out = tmp_path / "nvs09_shift.nl"
    out.write_text(translate_nl(src.read_text(encoding="latin-1"), shift), encoding="latin-1")
    r = from_nl(str(out)).solve(time_limit=30)
    tol = 1e-6 + 1e-4 * abs(truth)
    assert r.status != "infeasible"
    if r.bound is not None:
        assert r.bound <= truth + tol
    if r.gap_certified:
        assert r.objective == pytest.approx(truth, abs=tol)


def test_rescaled_logical_point_is_reverified_on_the_callers_form():
    """#1414's class through the rescaling: ``min -x + 3 z, x <= M z`` with M = 1e8.
    The unit slack is rescaled by ~2**26, so x = 10, z = 0 (a 10-unit violation) is
    1.5e-7 in the scaled slack and passes its ``>= -tol`` bound. The mapped point
    must be re-verified on the original form and the rescaled answer discarded."""
    m = dm.Model("bigm")
    x = m.continuous("x", lb=0, ub=10)
    z = m.binary("z")
    m.subject_to(x - 1e8 * z <= 0)
    m.minimize(-x + 3 * z)
    r = m.solve(time_limit=30)
    st = r.solver_stats or {}
    assert st.get("route/lp_milp_backend") == 1.0
    assert st.get("milp/logicals_rescale_refused") == 1.0, st
    assert r.objective is None or r.objective >= -7.0 - 1e-6, (r.status, r.objective)

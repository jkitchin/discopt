"""Convex-quadratic (ellipsoid) bound tightening — issue #1193.

``nvs19`` (8 integer variables in [0, 200], 8 dense quadratic rows) stalled at
-1098.2 against the -1098.4 optimum and terminated on ``max_nodes``. The dual
side was the binding half: interval FBBT evaluates a row one term at a time, so
every cross term ``x_i x_j`` becomes the product of two independent intervals
and the coupling that actually bounds the set is discarded. Four of nvs19's rows
are positive definite — each is an *ellipsoid* whose coordinate ranges have a
closed form.

Measured on nvs19: the root box after Rust FBBT + OBBT is
``x_i <= [49, 55, 55, 54, 61, 58, 53, 58]``; one positive-definite row gives
``x_i <= [14, 19, 19, 21, 17, 15, 21, 15]``.

Every test here holds the rule to the two things bound tightening owes: the new
box is a subset of the old one, and it never cuts a feasible point.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt._relax.model_utils import flat_variable_bounds  # noqa: E402
from discopt._relax.nonlinear_bound_tightening import (  # noqa: E402
    _ELLIPSOID_ENV,
    _ELLIPSOID_MAX_SUPPORT,
    ConvexQuadraticEllipsoidRule,
    build_flat_variable_metadata,
    tighten_nonlinear_bounds,
)
from discopt._relax.quadratic_form import (  # noqa: E402
    extract_quadratic,
    extract_quadratic_support,
    polynomial_degree_bound,
)
from discopt.modeling.core import Model  # noqa: E402

pytestmark = pytest.mark.unit


@pytest.fixture
def ellipsoid_on(monkeypatch):
    """Enable the default-off rule for the duration of a test."""
    monkeypatch.setenv(_ELLIPSOID_ENV, "1")


def _tighten(model):
    flat_lb, flat_ub = flat_variable_bounds(model)
    lb, ub, stats = tighten_nonlinear_bounds(model, flat_lb, flat_ub)
    return flat_lb, flat_ub, lb, ub, stats


def _assert_subset_box(new_lb, new_ub, old_lb, old_ub):
    assert np.all(new_lb >= old_lb - 1e-12)
    assert np.all(new_ub <= old_ub + 1e-12)


# ---------------------------------------------------------------------------
# The support-restricted quadratic extraction the rule is built on
# ---------------------------------------------------------------------------


def _rotated_ellipse_model(lb=-10.0, ub=10.0, integer=False):
    """``2x^2 + 2y^2 + 2xy <= 1`` — positive definite WITH a cross term.

    ``Q = [[2, 1], [1, 2]]``, ``Q^-1 = [[2, -1], [-1, 2]] / 3``, ``r^2 = 1``, so
    the exact coordinate range is ``+-sqrt(2/3)`` on both axes. The separable
    rule abstains on this row because of the ``xy`` term.
    """
    m = Model("rotated_ellipse")
    kind = m.integer if integer else m.continuous
    x = kind("x", lb=lb, ub=ub)
    y = kind("y", lb=lb, ub=ub)
    m.subject_to(2.0 * x**2 + 2.0 * y**2 + 2.0 * x * y <= 1.0)
    m.minimize(x + y)
    return m


def test_extract_quadratic_support_scatters_to_the_dense_form():
    m = _rotated_ellipse_model()
    body = m._constraints[0].body
    reduced = extract_quadratic_support(body, m)
    assert reduced is not None
    support, Q_s, c_s, d_s = reduced
    assert support == (0, 1)

    dense = extract_quadratic(body, 2, m)
    assert dense is not None
    Q, c, d = dense
    idx = np.asarray(support, dtype=np.intp)
    # Bit-for-bit, not merely close: the dense form is now built by scattering
    # the reduced one, and a bound derived from either must be identical.
    assert np.array_equal(Q[np.ix_(idx, idx)], Q_s)
    assert np.array_equal(c[idx], c_s)
    assert d == d_s


def test_extract_quadratic_support_abstains_on_a_cubic():
    m = Model("cubic")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    m.subject_to(x**3 <= 1.0)
    m.minimize(x)
    assert extract_quadratic_support(m._constraints[0].body, m) is None


def test_flat_layouts_agree():
    """The rule indexes with the tightening metadata but extracts with the
    model's own prefix-sum layout; a mismatch would tighten the wrong variable."""
    m = Model("layout")
    m.continuous("a", shape=(3,), lb=0.0, ub=1.0)
    m.integer("b", lb=0, ub=5)
    m.continuous("c", shape=(2,), lb=-1.0, ub=1.0)
    m.minimize(m._variables[1])
    metadata = build_flat_variable_metadata(m)
    checks = 0
    for var in m._variables:
        assert metadata.base_offsets[id(var)] == m._flat_var_offset(var)
        checks += 1
    assert checks == 3


# ---------------------------------------------------------------------------
# The rule itself
# ---------------------------------------------------------------------------


def test_flag_off_is_a_noop():
    m = _rotated_ellipse_model()
    lb0, ub0, lb, ub, stats = _tighten(m)
    assert "convex_quadratic_ellipsoid" not in stats.applied_rules
    assert np.array_equal(lb, lb0)
    assert np.array_equal(ub, ub0)


def test_rotated_ellipse_bounds_are_exact(ellipsoid_on):
    m = _rotated_ellipse_model()
    lb0, ub0, lb, ub, stats = _tighten(m)
    assert not stats.infeasible
    assert "convex_quadratic_ellipsoid" in stats.applied_rules
    _assert_subset_box(lb, ub, lb0, ub0)
    exact = np.sqrt(2.0 / 3.0)
    assert ub == pytest.approx([exact, exact], rel=1e-5)
    assert lb == pytest.approx([-exact, -exact], rel=1e-5)
    # The margin is one-sided: never inside the true range.
    assert np.all(ub >= exact)
    assert np.all(lb <= -exact)


def test_off_center_ellipse_bounds_are_exact(ellipsoid_on):
    # (x-3)^2 + (y+1)^2 + (x-3)(y+1) <= 1, expanded, so the linear part is
    # non-zero and the centre must be recovered rather than assumed at 0.
    m = Model("offcenter")
    x = m.continuous("x", lb=-50.0, ub=50.0)
    y = m.continuous("y", lb=-50.0, ub=50.0)
    u, v = x - 3.0, y + 1.0
    m.subject_to(u * u + v * v + u * v <= 1.0)
    m.minimize(x + y)
    lb0, ub0, lb, ub, stats = _tighten(m)
    assert "convex_quadratic_ellipsoid" in stats.applied_rules
    _assert_subset_box(lb, ub, lb0, ub0)
    # Q = [[1, .5], [.5, 1]], Q^-1 = [[4, -2], [-2, 4]]/3, r^2 = 1.
    rad = np.sqrt(4.0 / 3.0)
    assert ub == pytest.approx([3.0 + rad, -1.0 + rad], rel=1e-5)
    assert lb == pytest.approx([3.0 - rad, -1.0 - rad], rel=1e-5)


@pytest.mark.parametrize("integer", [False, True])
def test_bounds_never_cut_a_feasible_point(ellipsoid_on, integer):
    m = _rotated_ellipse_model(integer=integer)
    lb0, ub0, lb, ub, stats = _tighten(m)
    _assert_subset_box(lb, ub, lb0, ub0)

    rng = np.random.default_rng(1193)
    checks = 0
    # Sample the box, keep the row-feasible points, and demand every one of them
    # survives the tightening.
    for _ in range(20000):
        pt = rng.uniform(lb0, ub0)
        if integer:
            pt = np.round(pt)
        if 2 * pt[0] ** 2 + 2 * pt[1] ** 2 + 2 * pt[0] * pt[1] > 1.0:
            continue
        assert np.all(pt >= lb - 1e-9) and np.all(pt <= ub + 1e-9), pt
        checks += 1
    assert checks > 0, "probe never saw a feasible point — it proved nothing"


def test_integer_bounds_are_rounded_inward(ellipsoid_on):
    m = _rotated_ellipse_model(integer=True)
    _lb0, _ub0, lb, ub, stats = _tighten(m)
    assert "convex_quadratic_ellipsoid" in stats.applied_rules
    # +-sqrt(2/3) = +-0.8165 rounds to the single integer 0 on both axes.
    assert list(lb) == [0.0, 0.0]
    assert list(ub) == [0.0, 0.0]


def test_ge_sense_uses_the_negated_form(ellipsoid_on):
    m = Model("ge")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.subject_to(1.0 - 2.0 * x**2 - 2.0 * y**2 - 2.0 * x * y >= 0.0)
    m.minimize(x + y)
    _lb0, _ub0, lb, ub, stats = _tighten(m)
    assert "convex_quadratic_ellipsoid" in stats.applied_rules
    exact = np.sqrt(2.0 / 3.0)
    assert ub == pytest.approx([exact, exact], rel=1e-5)


def test_eq_sense_uses_the_convex_direction(ellipsoid_on):
    m = Model("eq")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.subject_to(2.0 * x**2 + 2.0 * y**2 + 2.0 * x * y == 1.0)
    m.minimize(x + y)
    _lb0, _ub0, lb, ub, stats = _tighten(m)
    assert "convex_quadratic_ellipsoid" in stats.applied_rules
    exact = np.sqrt(2.0 / 3.0)
    assert ub == pytest.approx([exact, exact], rel=1e-5)
    assert lb == pytest.approx([-exact, -exact], rel=1e-5)


def test_indefinite_row_abstains(ellipsoid_on):
    m = Model("indefinite")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.subject_to(x**2 - y**2 + x * y <= 1.0)
    m.minimize(x + y)
    lb0, ub0, lb, ub, stats = _tighten(m)
    assert "convex_quadratic_ellipsoid" not in stats.applied_rules
    assert np.array_equal(ub, ub0) and np.array_equal(lb, lb0)


def test_wrong_sense_direction_abstains(ellipsoid_on):
    # 2x^2 + 2y^2 + 2xy >= 1 is the COMPLEMENT of an ellipsoid: unbounded.
    m = Model("wrong_direction")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.subject_to(2.0 * x**2 + 2.0 * y**2 + 2.0 * x * y >= 1.0)
    m.minimize(x + y)
    lb0, ub0, lb, ub, stats = _tighten(m)
    assert "convex_quadratic_ellipsoid" not in stats.applied_rules
    assert np.array_equal(ub, ub0) and np.array_equal(lb, lb0)


def test_empty_ellipsoid_abstains_rather_than_pruning(ellipsoid_on):
    # 2x^2 + 2y^2 + 2xy <= -1 has no solution. The rule declines to draw the
    # infeasibility rather than risk a false prune from round-off.
    m = Model("empty")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    y = m.continuous("y", lb=-10.0, ub=10.0)
    m.subject_to(2.0 * x**2 + 2.0 * y**2 + 2.0 * x * y <= -1.0)
    m.minimize(x + y)
    lb0, ub0, lb, ub, stats = _tighten(m)
    assert "convex_quadratic_ellipsoid" not in stats.applied_rules
    assert np.array_equal(ub, ub0) and np.array_equal(lb, lb0)


def test_oversized_support_is_skipped(ellipsoid_on):
    n = _ELLIPSOID_MAX_SUPPORT + 5
    m = Model("oversized")
    x = m.continuous("x", shape=(n,), lb=-10.0, ub=10.0)
    m.subject_to(dm.sum(x**2) + x[0] * x[1] <= 1.0)
    m.minimize(x[0])
    _lb0, _ub0, _lb, _ub, stats = _tighten(m)
    assert "convex_quadratic_ellipsoid" not in stats.applied_rules


def test_random_positive_definite_rows_never_cut_a_feasible_point(ellipsoid_on):
    """The soundness claim, over random ellipsoids rather than one hand-picked."""
    rng = np.random.default_rng(20260906)
    checks = 0
    for _ in range(12):
        n = int(rng.integers(2, 5))
        A = rng.normal(size=(n, n))
        Q = A @ A.T + n * np.eye(n)  # positive definite by construction
        c = rng.normal(size=n)
        m = Model("rand")
        x = m.continuous("x", shape=(n,), lb=-50.0, ub=50.0)
        body = sum(Q[i, j] * x[i] * x[j] for i in range(n) for j in range(n))
        body = body + sum(c[i] * x[i] for i in range(n))
        m.subject_to(body <= 10.0)
        m.minimize(dm.sum(x))
        lb0, ub0, lb, ub, stats = _tighten(m)
        _assert_subset_box(lb, ub, lb0, ub0)
        for _ in range(4000):
            pt = rng.uniform(lb0, ub0)
            if pt @ Q @ pt + c @ pt > 10.0:
                continue
            assert np.all(pt >= lb - 1e-9) and np.all(pt <= ub + 1e-9)
            checks += 1
    assert checks > 0, "probe never saw a feasible point — it proved nothing"


def test_rule_is_registered_in_the_default_list():
    from discopt._relax.nonlinear_bound_tightening import DEFAULT_NONLINEAR_BOUND_RULES

    assert any(
        isinstance(rule, ConvexQuadraticEllipsoidRule) for rule in DEFAULT_NONLINEAR_BOUND_RULES
    )


# ---------------------------------------------------------------------------
# Differential bound test on real corpus instances (CLAUDE.md §5)
# ---------------------------------------------------------------------------

_CORPUS = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

# The nvs family is the class #1193 came from: small integer programs whose
# rows are dense positive-definite quadratics. Oracles from minlplib.solu.
_NVS_ORACLES = {"nvs10": -310.8, "nvs11": -431.0, "nvs12": -481.2, "nvs13": -585.2}


@pytest.mark.parametrize("name,oracle", sorted(_NVS_ORACLES.items()))
def test_nvs_family_bound_is_tighter_and_never_invalid(monkeypatch, name, oracle):
    """Differential bound test: ON >= OFF (tighter) and ON <= the true optimum.

    A bound-tightening rule may only move the dual bound *up* (min sense) and
    may never move it past the optimum — that would be a false certificate.
    """
    path = os.path.join(_CORPUS, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"corpus instance {name} not present")
    from discopt.modeling.core import from_nl

    bounds = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(_ELLIPSOID_ENV, flag)
        r = from_nl(path).solve(max_nodes=2000, time_limit=120.0)
        bounds[flag] = (r.bound, r.objective)

    tol = 1e-4 * (1 + abs(oracle))
    off_bound, off_obj = bounds["0"]
    on_bound, on_obj = bounds["1"]
    # Soundness first: neither arm may claim a bound past the optimum.
    assert off_bound <= oracle + tol, f"OFF bound {off_bound} > oracle {oracle}"
    assert on_bound <= oracle + tol, f"ON bound {on_bound} > oracle {oracle} (UNSOUND)"
    # ...nor an incumbent better than it.
    assert on_obj >= oracle - tol, f"ON objective {on_obj} beats oracle {oracle}"
    # Then the differential: the rule may not loosen the bound.
    assert on_bound >= off_bound - tol, f"ON bound {on_bound} LOOSER than OFF {off_bound}"


def test_nvs_root_box_is_materially_tighter(monkeypatch):
    """The mechanism, measured: the ellipsoid box is far smaller than FBBT's.

    Recorded because it is the whole claim of #1193 — interval FBBT drops the
    cross-term coupling that bounds these rows, so its box is loose by
    construction, not by a tolerance.
    """
    path = os.path.join(_CORPUS, "nvs13.nl")
    if not os.path.exists(path):
        pytest.skip("corpus instance nvs13 not present")
    from discopt.modeling.core import from_nl

    m = from_nl(path)
    lb0, ub0 = flat_variable_bounds(m)
    widths = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(_ELLIPSOID_ENV, flag)
        lb, ub, _stats = tighten_nonlinear_bounds(m, lb0.copy(), ub0.copy())
        widths[flag] = ub - lb
    assert np.all(widths["1"] <= widths["0"] + 1e-12), "ON box is not a subset of OFF"
    # Measured 2026-09-06: geometric-mean width ratio 0.082 on nvs13.
    ratio = float(np.exp(np.mean(np.log(widths["1"] / widths["0"]))))
    assert ratio < 0.5, f"expected a materially tighter box, got width ratio {ratio:.3f}"


# ---------------------------------------------------------------------------
# The cheap degree pre-filter
# ---------------------------------------------------------------------------


def test_degree_bound_on_the_shapes_it_must_get_right():
    m = Model("deg")
    x = m.continuous("x", lb=-1.0, ub=1.0)
    y = m.continuous("y", lb=-1.0, ub=1.0)
    m.minimize(x)
    assert polynomial_degree_bound(x) == 1
    assert polynomial_degree_bound(x + y) == 1
    assert polynomial_degree_bound(x * y) == 2
    assert polynomial_degree_bound(x**2) == 2
    assert polynomial_degree_bound(-(x * y)) == 2
    assert polynomial_degree_bound(x / 2.0) == 1
    assert polynomial_degree_bound((x**2 + y) * (x + y) ** 2) == 4
    # Non-polynomial and unbounded shapes: unknown, so callers keep their path.
    assert polynomial_degree_bound(dm.exp(x)) is None
    assert polynomial_degree_bound(x**0.5) is None
    assert polynomial_degree_bound(1.0 / x) is None


def test_high_degree_row_is_rejected_without_expanding(ellipsoid_on, monkeypatch):
    """The pre-filter must actually short-circuit, not merely agree.

    st_e36 row 0 (a quadratic times a squared quadratic) cost 554 ms to expand
    before being thrown away. Assert ``extract_quadratic_support`` is never
    reached for a degree-6 row.
    """
    import discopt._relax.nonlinear_bound_tightening as nbt

    calls = {"n": 0}
    orig = nbt.extract_quadratic_support

    def counting(expr, model):
        calls["n"] += 1
        return orig(expr, model)

    monkeypatch.setattr(nbt, "extract_quadratic_support", counting)

    m = Model("deg6")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    y = m.continuous("y", lb=-5.0, ub=5.0)
    m.subject_to((x**2 + y) * (x + y) ** 2 * (x * y) <= 1.0)
    m.minimize(x + y)
    lb0, ub0, lb, ub, _stats = _tighten(m)
    assert calls["n"] == 0, "the degree pre-filter did not short-circuit"
    assert np.array_equal(lb, lb0) and np.array_equal(ub, ub0)

    # ...and a genuine quadratic row still reaches the extraction.
    m2 = _rotated_ellipse_model()
    _tighten(m2)
    assert calls["n"] > 0, "a quadratic row was skipped by the pre-filter"

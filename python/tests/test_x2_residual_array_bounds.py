"""X-2 residual (#413) — array-variable bounds must not collapse to element 0.

The X-2 shared root is the "variable block treated as a scalar" class. Its two
P0s (CORE-1=C-29 extractor array-as-sum, TG-1=C-31 FBBT per-element seeding) were
fixed earlier. This module pins the *residual* two consumers found by the
codebase sweep, both of which read an array variable's bounds through element 0
(``v.lb.flat[0]`` / ``arr.flat[0]``) and would silently drop or narrow the
heterogeneous per-element bounds:

1. ``export/gams.py`` — heterogeneous array bounds were emitted *only when
   uniform*, so a model with distinct per-element bounds exported with **no
   bounds at all** (EX-4). GAMS then solves a different model than discopt.
2. ``_jax/gdp_reformulate.py:_compute_big_m_lp`` — the LP-optimum big-M read
   element 0's bounds and stamped them onto every element, so a too-tight
   element-0 upper bound produced a **too-small big-M** that cuts feasible
   points of the inactive disjunct (the exact unsoundness the "exact oracle
   only" comment there guards against).

Each test fails on the pre-fix ``.flat[0]`` collapse and passes after.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.smoke


# ─────────────────────────────────────────────────────────────
# GAMS export: heterogeneous per-element bounds must survive
# ─────────────────────────────────────────────────────────────


def test_gams_export_preserves_heterogeneous_array_bounds(heterogeneous_array_bounds):
    """to_gams must emit every element's own bound, not drop them when non-uniform.

    Pre-fix: bounds were written only when ``np.all(arr == arr.flat[0])``, so a
    heterogeneous block produced ZERO bound lines. Post-fix: one ``.lo``/``.up``
    line per element at its 1-based GAMS label.
    """
    m, x, lb, ub = heterogeneous_array_bounds(shape=(3,), lb=[0.0, 2.0, 4.0], ub=[1.0, 5.0, 9.0])
    gams = m.to_gams()

    lo_lines = [ln for ln in gams.splitlines() if "x.lo(" in ln]
    up_lines = [ln for ln in gams.splitlines() if "x.up(" in ln]

    # The heterogeneous case must emit a bound for every element (not be dropped).
    assert len(lo_lines) == 3, f"expected 3 per-element .lo lines, got {lo_lines}"
    assert len(up_lines) == 3, f"expected 3 per-element .up lines, got {up_lines}"

    # Element 1 (lb=2, ub=5) must NOT be collapsed to element 0's (lb=0, ub=1).
    assert any("x.lo('2') = 2.0" in ln for ln in lo_lines), lo_lines
    assert any("x.up('2') = 5.0" in ln for ln in up_lines), up_lines
    assert any("x.up('3') = 9.0" in ln for ln in up_lines), up_lines


def test_gams_export_uniform_bounds_still_compact(heterogeneous_array_bounds):
    """A uniform block must still compact to one domain-wide assignment (no regression)."""
    from discopt import Model

    m = Model("uniform")
    u = m.continuous("u", shape=(3,), lb=0.0, ub=10.0)
    m.minimize(u[0])
    m.subject_to(u[0] >= 0.0)
    gams = m.to_gams()
    lo_lines = [ln for ln in gams.splitlines() if "u.lo" in ln]
    up_lines = [ln for ln in gams.splitlines() if "u.up" in ln]
    # Compact: exactly one .lo and one .up over the whole set domain.
    assert lo_lines == ["u.lo(s1) = 0.0;"], lo_lines
    assert up_lines == ["u.up(s1) = 10.0;"], up_lines


def test_gams_roundtrip_preserves_per_element_bounds():
    """from_gams must parse per-element ``x.lo('k')`` bounds back to the same array.

    Hand-written GAMS (scalar equation to avoid the separate, pre-existing
    concrete-label-in-equation-body from_gams limitation) so this isolates the
    bound parse/apply round-trip.
    """
    from discopt.modeling.core import from_gams

    gms = (
        "Set s1 / 1, 2, 3 /;\n"
        "Free Variables obj_var;\n"
        "Positive Variables x(s1);\n"
        "x.lo('1') = 0.0;\n"
        "x.lo('2') = 2.0;\n"
        "x.lo('3') = 4.0;\n"
        "x.up('1') = 1.0;\n"
        "x.up('2') = 5.0;\n"
        "x.up('3') = 9.0;\n"
        "Equations obj_eq;\n"
        "obj_eq.. obj_var =e= 3;\n"
        "Model model / all /;\n"
        "Solve model using LP minimizing obj_var;\n"
    )
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "rt.gms"
        p.write_text(gms)
        m = from_gams(str(p))

    x = next(v for v in m._variables if v.name == "x")
    np.testing.assert_allclose(np.asarray(x.lb).ravel(), [0.0, 2.0, 4.0])
    np.testing.assert_allclose(np.asarray(x.ub).ravel(), [1.0, 5.0, 9.0])


# ─────────────────────────────────────────────────────────────
# GDP big-M: LP-optimum M must use each element's own bounds
# ─────────────────────────────────────────────────────────────


def test_gdp_bigm_lp_uses_per_element_bounds():
    """_compute_big_m_lp must bound each array element by its own box, not element 0.

    Repro: x = [x0 in [0,1], x1 in [0,10]], body = x1 (sense <=). True big-M is
    max(x1) = 10 over the box. Pre-fix, element 0's ub=1 was stamped onto x1, so
    the LP came back with M ~= 1 — a too-small big-M that cuts feasible points of
    the inactive disjunct. Post-fix M ~= 10 (matching the sound interval path).
    """
    import discopt._jax.gdp_reformulate as g
    from discopt import Model
    from discopt.modeling.core import Constraint

    exact = None
    try:
        from discopt.solvers.lp_backend import get_exact_lp_solver

        exact = get_exact_lp_solver()
    except Exception:
        exact = None
    if exact is None:
        pytest.skip("no exact LP oracle available; _compute_big_m_lp falls back to interval")

    m = Model("gdp_bigm")
    x = m.continuous("x", shape=(2,), lb=[0.0, 0.0], ub=[1.0, 10.0])
    m.subject_to(x[0] + x[1] <= 100)  # a linear row so the LP relaxation is non-trivial
    con = Constraint(body=x[1], sense="<=", rhs=0.0, name="c")

    lp_data = g._precompute_lp_relaxation(m)
    assert lp_data is not None
    m_lp = g._compute_big_m_lp(con, m, lp_data)

    # The sound interval path (a known-correct oracle) gives ~10.1; the LP path
    # must agree, not collapse to element 0's ub=1.
    m_interval = g._compute_big_m(con, m)
    assert m_lp == pytest.approx(m_interval, rel=1e-6), (m_lp, m_interval)
    assert m_lp > 9.0, f"big-M collapsed to element 0's box: {m_lp}"


def test_gdp_bigm_lp_heterogeneous_fixture(heterogeneous_array_bounds):
    """The shared fixture's block must not collapse in the LP big-M path."""
    import discopt._jax.gdp_reformulate as g
    from discopt.modeling.core import Constraint

    try:
        from discopt.solvers.lp_backend import get_exact_lp_solver

        if get_exact_lp_solver() is None:
            pytest.skip("no exact LP oracle available")
    except Exception:
        pytest.skip("no exact LP oracle available")

    # x0 in [0,1] (tightest), x2 in [4,9]: element 2's true upper bound is 9.
    m, x, lb, ub = heterogeneous_array_bounds(shape=(3,), lb=[0.0, 2.0, 4.0], ub=[1.0, 5.0, 9.0])
    con = Constraint(body=x[2], sense="<=", rhs=0.0, name="c")
    lp_data = g._precompute_lp_relaxation(m)
    assert lp_data is not None
    m_lp = g._compute_big_m_lp(con, m, lp_data)
    # Must reach element 2's own ub (9), not element 0's (1).
    assert m_lp > 8.0, f"big-M collapsed to element 0's box: {m_lp}"

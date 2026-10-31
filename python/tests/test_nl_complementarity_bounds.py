"""`.nl` complementarity rows are read with AMPL's semantics (#2056).

A type-5 row ``5 flag i`` is the mixed complementarity relation
``body ⊥ x ∈ [lb, ub]`` with ``x`` the ``i``-th variable (1-based): at ``x = lb``
the body is ``>= 0``, at ``x = ub`` it is ``<= 0``, strictly between it is 0.
``flag`` records which of ``x``'s bounds are finite -- AMPL MP ``ComplInfo``:
``INF_UB = 1`` (finite variable lower bound) and ``INF_LB = 2`` (finite variable
upper bound); Pyomo's ``mpec4`` writer adds 1 for a lower and 2 for an upper bound.

``from_nl`` had both halves wrong: it read bit 0 as "body ``<= 0``" (the bits were
swapped) and complemented the body against ``x`` itself instead of against
``x - lb`` / ``ub - x``. Each wrong reading certified a wrong optimum.

Every fixture under ``data/nl_complementarity/`` is **AMPL output**, written from
the ``.mod`` beside it with ``option presolve 0; write g<name>;``. AMPL drops a body
variable's own bound when the relation implies it, and keeps a bound on the
complementary variable's open side only as a ``cap`` row, so a big-M lowering of the
file as written has no finite enclosure and refuses (loudly, correctly). The solve
tests therefore restore exactly the bounds the ``.mod`` declares -- each one is
implied by the file (a ``cap`` row or the relation itself), so the feasible set is
unchanged and neither reading of the flag is presupposed.
"""

from __future__ import annotations

from pathlib import Path

import discopt.modeling.core as dm
import pytest

pytestmark = pytest.mark.smoke

_DATA = Path(__file__).parent / "data" / "nl_complementarity"


def _load(name: str, bounds: dict[str, tuple[float, float]] | None = None) -> dm.Model:
    """Import ``ampl_<name>.nl`` and restore the ``.mod``'s declared bounds."""
    m = dm.from_nl(str(_DATA / f"ampl_{name}.nl"))
    by_name = {v.name: v for v in m._variables}
    for vname, (lb, ub) in (bounds or {}).items():
        by_name[vname].lb = lb
        by_name[vname].ub = ub
    return m


def _solve(m: dm.Model):
    res = m.solve(time_limit=60.0, gap_tolerance=1e-8)
    assert res.status == "optimal", res.status
    return res


def _relation(m: dm.Model):
    """The single relation the import recorded, evaluated as ``(a(x), b(x))``."""
    assert len(m._complementarities) == 1
    return m._complementarities[0]


# ── the relation is against the bound, not against 0 ─────────────────────────


def test_nonzero_lower_bound_is_the_complementarity_bound():
    """``3 <= x <= Infinity complements y``: ``x = 3`` leaves ``y`` free to be
    positive, so ``(3, 1)`` is feasible with objective 0.

    Before the fix: ``x * y == 0`` with ``y <= 0`` (both halves wrong) forced
    ``y = 0`` and certified ``optimal`` at 1.0."""
    m = _load("lo", {"x0": (3.0, 10.0), "x1": (-10.0, 10.0)})
    res = _solve(m)
    assert res.objective == pytest.approx(0.0, abs=1e-6)
    assert float(res.x["x0"]) == pytest.approx(3.0, abs=1e-5)
    assert float(res.x["x1"]) == pytest.approx(1.0, abs=1e-5)


def test_nonzero_upper_bound_is_the_complementarity_bound():
    """Mirror: ``-Infinity <= x <= -3 complements y``. At ``x = -3`` the body may
    be negative, so ``(-3, -1)`` is feasible with objective 0 (was 1.0)."""
    m = _load("up", {"x0": (-10.0, -3.0), "x1": (-10.0, 10.0)})
    res = _solve(m)
    assert res.objective == pytest.approx(0.0, abs=1e-6)
    assert float(res.x["x0"]) == pytest.approx(-3.0, abs=1e-5)
    assert float(res.x["x1"]) == pytest.approx(-1.0, abs=1e-5)


def test_flag_one_means_body_nonnegative():
    """The bit swap alone, with ``lb = 0``: ``0 <= x complements y`` forces
    ``y >= 0`` (AMPL omits ``y``'s ``-10`` bound as implied). Optimum 1 at
    ``(0, 0)``; the swapped reading allowed ``y = -1`` and certified 0 -- a dual
    bound below the true optimum."""
    m = _load("swap", {"x0": (0.0, 10.0), "x1": (-10.0, 10.0)})
    res = _solve(m)
    assert res.objective == pytest.approx(1.0, abs=1e-6)
    assert float(res.x["x1"]) >= -1e-6


def test_relation_is_measured_from_the_bound():
    """Structural check of the recorded pair, independent of any solve: for
    ``lo`` the pair is ``(y, x - 3)``; for ``up`` it is ``(-y, -3 - x)``."""
    import numpy as np
    from discopt._point_eval import evaluate_at

    checked = 0
    for name, want in (
        ("lo", lambda x, y: (y, x - 3.0)),
        ("up", lambda x, y: (-y, -3.0 - x)),
    ):
        m = _load(name)
        rel = _relation(m)
        for pt in ([3.0, 1.0], [5.0, -2.0], [-4.0, 0.5], [-3.0, 7.0]):
            x = np.asarray(pt)
            wa, wb = want(*pt)
            assert float(evaluate_at(rel.f, m, x)) == pytest.approx(wa), (name, pt)
            assert float(evaluate_at(rel.g, m, x)) == pytest.approx(wb), (name, pt)
            checked += 1
    assert checked == 8


# ── the well-formed baseline still matches the Python build ──────────────────


def test_zero_bound_ncp_matches_python_build():
    """``0 <= y <= Infinity complements x`` is the textbook NCP ``0 <= x ⊥ y >= 0``."""
    m_nl = _load("base", {"x0": (0.0, 10.0), "x1": (0.0, 10.0)})
    res_nl = _solve(m_nl)

    m_py = dm.Model("equiv")
    x = m_py.continuous("x", lb=0, ub=10)
    y = m_py.continuous("y", lb=0, ub=10)
    m_py.minimize((x - 1) ** 2 + (y - 1) ** 2)
    m_py.complementarity(x, y)
    res_py = _solve(m_py)

    assert res_nl.objective == pytest.approx(1.0, abs=1e-6)
    assert res_nl.objective == pytest.approx(res_py.objective, abs=1e-6)
    assert float(res_nl.x["x0"]) * float(res_nl.x["x1"]) == pytest.approx(0.0, abs=1e-6)


# ── refusals ─────────────────────────────────────────────────────────────────


def test_doubly_bounded_row_is_refused_not_misread():
    """``3 <= x <= 10 complements y`` (AMPL flag 3) is a box MCP. It needs the
    box-MCP lowering; until then the import refuses rather than approximates."""
    with pytest.raises(ValueError, match="doubly-bounded"):
        dm.from_nl(str(_DATA / "ampl_box.nl"))


@pytest.mark.parametrize("flag", [0, 2, 3])
def test_flag_inconsistent_with_variable_bounds_is_refused(tmp_path, flag):
    """``base`` complements ``y in [0, inf)``, so AMPL writes flag 1. Any other
    flag contradicts the bounds the same file declares; the intended relation is
    ambiguous and must not be guessed."""
    text = (_DATA / "ampl_base.nl").read_text()
    assert "\n5 1 2\n" in text
    path = tmp_path / "bad.nl"
    path.write_text(text.replace("\n5 1 2\n", f"\n5 {flag} 2\n"))
    with pytest.raises(ValueError, match="malformed"):
        dm.from_nl(str(path))


def test_hand_written_legacy_fixture_is_refused_as_malformed():
    """``mpcc_complementarity.nl`` (#658) was written for the swapped reading:
    flag 2 on a variable with both bounds finite, which AMPL would write as flag
    3. It is kept byte-for-byte because the serialization golden documents pin
    its digest; importing it now refuses instead of reading a relation the file
    does not state."""
    with pytest.raises(ValueError, match="malformed"):
        dm.from_nl(str(Path(__file__).parent / "data" / "mpcc_complementarity.nl"))

"""#1917: the seeded convex-quadratic GDP generator behind the
``DISCOPT_HULL_CONIC_QUADRATIC`` graduation panel.

Pinned: names resolve through ``gdplib_runner.discover_models``; a name always
builds the same model; and, the point of the corpus, the cone hull fires on the
generated families -- every disjunct ball row becomes a cone row ON, none does OFF.

The route family fires only partially (measured: 5 of 25 rows on
``route_t4k5d2_s1``). Its rows are ``((p - c) / a)**2``; ``_expr_to_polynomial``
abstains on a sum divided by a constant, so only the ellipsoid centred at the
origin (which Pyomo folds to ``(p / a)**2``) is recognized. Pinned as a strict
xfail so the test flips when the recognizer learns that form.
"""

from __future__ import annotations

import warnings

import pytest

from benchmarks import convex_gdp_generator as cg
from benchmarks import gdplib_runner as gr

pyo = pytest.importorskip("pyomo.environ")

FLAG = "DISCOPT_HULL_CONIC_QUADRATIC"


def _cone_rows(low) -> int:
    return sum("sqrt(" in str(c.body) and "/" not in str(c.body) for c in low._constraints)


def test_panel_names_resolve_and_parse():
    assert len(cg.PANEL) == 18
    assert {cg.parse(n)[0] for n in cg.PANEL} == {"clay", "cover", "route"}
    specs = gr.discover_models(include=list(cg.PANEL))
    assert sorted(s.name for s in specs) == sorted(cg.PANEL)
    assert not cg.is_name("cgdp.clay_r3_s1") and not cg.is_name("pyomo.simple1")


def test_a_name_builds_the_same_model():
    a = cg.clay_params(3, 3, 7)
    assert a == cg.clay_params(3, 3, 7) and a != cg.clay_params(3, 3, 8)


def _rows(monkeypatch, name: str) -> dict[str, int]:
    import discopt.transformations as dt
    from discopt.pyomo.gdp import convert_pyomo_gdp

    rows = {}
    for flag in ("0", "1"):
        monkeypatch.setenv(FLAG, flag)
        model = convert_pyomo_gdp(cg.builder(name)()).model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            rows[flag] = _cone_rows(dt.create_using("gdp.hull", model))
    return rows


@pytest.mark.parametrize(
    "name, n_rows",
    [("cgdp.clay_r3c2_s1", 24), ("cgdp.cover_n6k2_s1", 12)],
)
def test_the_cone_hull_fires_on_every_ball_row(monkeypatch, name, n_rows):
    assert _rows(monkeypatch, name) == {"0": 0, "1": n_rows}


@pytest.mark.xfail(
    strict=True,
    reason="_expr_to_polynomial abstains on ((p - c) / a)**2: only 5 of 25 rows cone",
)
def test_the_cone_hull_fires_on_every_ellipsoid_row(monkeypatch):
    assert _rows(monkeypatch, "cgdp.route_t4k5d2_s1") == {"0": 0, "1": 25}

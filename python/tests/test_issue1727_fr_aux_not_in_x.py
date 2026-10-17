"""#1727: the factorable lift's ``_fr_aux_*`` columns must not leak into ``x``."""

from __future__ import annotations

import discopt.modeling as dm
from discopt._relax.factorable_reform import collect_lift_aux_names


def _goldstein_price(extra_name: str | None = None):
    m = dm.Model("gp")
    x = m.continuous("x", lb=-2, ub=2)
    y = m.continuous("y", lb=-2, ub=2)
    if extra_name is not None:
        m.continuous(extra_name, lb=0, ub=1)
    f = (1 + (x + y + 1) ** 2 * (19 - 14 * x + 3 * x**2 - 14 * y + 6 * x * y + 3 * y**2)) * (
        30 + (2 * x - 3 * y) ** 2 * (18 - 32 * x + 12 * x**2 + 48 * y - 36 * x * y + 27 * y**2)
    )
    m.minimize(f)
    return m


def test_lift_columns_are_recorded_and_projected_out(monkeypatch):
    import discopt._relax.factorable_reform as fr

    seen: list[set[str]] = []
    real = fr.collect_lift_aux_names

    def spy():
        cm = real()
        names = cm.__enter__()
        seen.append(names)

        class _CM:
            def __enter__(self):
                return names

            def __exit__(self, *exc):
                return cm.__exit__(*exc)

        return _CM()

    monkeypatch.setattr(fr, "collect_lift_aux_names", spy)
    r = _goldstein_price().solve(time_limit=5)
    assert r.x is not None
    assert seen and seen[0], "the lift did not run; the probe measured nothing"
    assert set(r.x) == {"x", "y"}


def test_declared_variable_with_aux_like_name_is_kept():
    r = _goldstein_price(extra_name="_fr_aux_0").solve(time_limit=5)
    assert r.x is not None
    assert "_fr_aux_0" in r.x


def test_sink_is_scoped():
    with collect_lift_aux_names() as names:
        assert names == set()
    import discopt._relax.factorable_reform as fr

    assert fr._LIFT_AUX_SINK.get() is None

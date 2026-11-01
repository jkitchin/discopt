"""#1917: ``DISCOPT_OBBT_NS_STRICT`` tightens OBBT/DBBT only to a rigorous bound.

Today OBBT replaces the LP vertex objective ``raw`` by the Neumaier-Shcherbina
safe bound ``g`` only when ``raw > g + 1e-6 * (1 + |raw|)``. A vertex up to ~1e-6
above the true LP minimum is accepted as a bound. On the (retired) lifted hull of
GDPlib clay0203 a min probe returned raw 1.56e-7 against g = -1.35e-7, the guard
accepted it, and the solve certified 41709.77 for an optimum of 41573.26. DBBT sized
its gap from the raw ``z_lp`` as well.

No small model makes the simplex return a vertex above its own LP minimum, so the
tests replace the NS bound with ``g_true - 5e-7``: inside the legacy guard, so
legacy keeps ``raw`` (which is then *above* the reported rigorous bound) and strict
must take ``g``. Every strict test fails with the flag OFF.
"""

from __future__ import annotations

from types import SimpleNamespace

import discopt._relax.obbt as obbt_mod
import discopt.modeling as dm
import numpy as np
import pytest
from test_obbt_warm_probes import PANEL, _build_relaxation

FLAG = "DISCOPT_OBBT_NS_STRICT"
SHIFT = 5e-7  # inside the legacy 1e-6 guard


def _shifted_ns(monkeypatch, *, none: bool = False):
    """Replace the NS bound by ``g_true - SHIFT`` (or ``None``); record each value.

    Keys are ``(column, sign of c)`` for a single-column OBBT objective, else
    ``"obj"``."""
    real = obbt_mod._ns_safe_lp_lower_bound
    seen: dict = {}

    def fake(c, dual_values, A_ub, b_ub, lo, hi, **kw):
        g = real(c, dual_values, A_ub, b_ub, lo, hi, **kw)
        c = np.asarray(c)
        nz = np.flatnonzero(c)
        key = (int(nz[0]), int(np.sign(c[nz[0]]))) if nz.size == 1 else "obj"
        out = None if (none or g is None) else g - SHIFT
        seen[key] = out
        return out

    monkeypatch.setattr(obbt_mod, "_ns_safe_lp_lower_bound", fake)
    return seen


def _n_orig(model) -> int:
    return sum(v.size for v in model._variables)


def _base(rel, n):
    return (
        np.array([b[0] for b in rel._bounds[:n]], dtype=np.float64),
        np.array([b[1] for b in rel._bounds[:n]], dtype=np.float64),
    )


def test_relaxation_obbt_strict_takes_g(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    seen = _shifted_ns(monkeypatch)
    checked = 0
    for make in PANEL.values():
        seen.clear()
        model = make()
        n = _n_orig(model)
        rel = _build_relaxation(model)
        lb0, ub0 = _base(rel, n)
        res = obbt_mod.run_obbt_on_relaxation(rel, n_orig=n)
        for j in range(n):
            if res.tightened_lb[j] > lb0[j]:
                g = seen[(j, 1)]
                assert g is not None
                # min(raw, g) == g here, never the raw vertex above it.
                assert res.tightened_lb[j] == min(g, res.tightened_ub[j])
                checked += 1
            if res.tightened_ub[j] < ub0[j]:
                g = seen[(j, -1)]
                assert g is not None
                assert res.tightened_ub[j] == max(-g, res.tightened_lb[j])
                checked += 1
    assert checked > 0, "no bound tightened: the instrument measured nothing"
    print(f"EXECUTED {checked} strict-bound assertions")


def test_relaxation_obbt_legacy_keeps_raw_inside_guard(monkeypatch):
    """The defect, pinned: OFF keeps a vertex above the rigorous bound."""
    monkeypatch.setenv(FLAG, "0")
    seen = _shifted_ns(monkeypatch)
    above = 0
    for make in PANEL.values():
        seen.clear()
        model = make()
        n = _n_orig(model)
        rel = _build_relaxation(model)
        lb0, _ = _base(rel, n)
        res = obbt_mod.run_obbt_on_relaxation(rel, n_orig=n)
        for j in range(n):
            g = seen.get((j, 1))
            if res.tightened_lb[j] > lb0[j] and g is not None and res.tightened_lb[j] > g:
                above += 1
    assert above > 0


def test_relaxation_obbt_strict_refuses_without_g(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    _shifted_ns(monkeypatch, none=True)
    total = 0
    for make in PANEL.values():
        model = make()
        n = _n_orig(model)
        res = obbt_mod.run_obbt_on_relaxation(_build_relaxation(model), n_orig=n)
        assert res.n_lp_solves > 0
        assert res.n_tightened == 0
        total += res.n_lp_solves
    # OFF, the same probes tighten to the raw vertex.
    monkeypatch.setenv(FLAG, "0")
    off = sum(
        obbt_mod.run_obbt_on_relaxation(_build_relaxation(m), n_orig=_n_orig(m)).n_tightened
        for m in (make() for make in PANEL.values())
    )
    assert off > 0 and total > 0


def _linear_model():
    m = dm.Model("ns_strict_lin")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    m.subject_to(x + y <= 4.0)
    m.subject_to(x - y >= -1.0)
    m.minimize(x + y)
    return m


def test_model_obbt_strict_takes_g(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    seen = _shifted_ns(monkeypatch)
    res = obbt_mod.run_obbt(_linear_model())
    checked = 0
    for j in range(2):
        if res.tightened_ub[j] < 10.0:
            assert res.tightened_ub[j] == -seen[(j, -1)]
            checked += 1
        if res.tightened_lb[j] > 0.0:
            assert res.tightened_lb[j] == seen[(j, 1)]
            checked += 1
    assert checked > 0
    # The shift is visible: x <= 4 becomes x <= 4 + SHIFT, not the raw 4.
    assert res.tightened_ub[0] == pytest.approx(4.0 + SHIFT, abs=1e-9)


def test_model_obbt_strict_refuses_without_g(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    _shifted_ns(monkeypatch, none=True)
    res = obbt_mod.run_obbt(_linear_model())
    assert res.n_lp_solves > 0 and res.n_tightened == 0
    monkeypatch.setenv(FLAG, "0")
    assert obbt_mod.run_obbt(_linear_model()).n_tightened > 0


def _dbbt_rel():
    # min x + y s.t. x >= 2 over [0,10]^2: LP min 2 at (2, 0), reduced cost of y is 1.
    return SimpleNamespace(
        _bounds=[(0.0, 10.0), (0.0, 10.0)],
        _c=np.array([1.0, 1.0]),
        _A_ub=np.array([[-1.0, 0.0]]),
        _b_ub=np.array([-2.0]),
        _obj_offset=0.0,
    )


def test_dbbt_strict_uses_safe_z(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    seen = _shifted_ns(monkeypatch)
    cutoff = 5.0
    res = obbt_mod.dbbt_on_relaxation(_dbbt_rel(), 2, cutoff)
    g = seen["obj"]
    assert g is not None and g < 2.0
    # y <= 0 + (cutoff - g + slack) / d_y, with d_y = 1 recomputed from the same y.
    expected = (cutoff - g) + 1e-6 * (1.0 + cutoff)
    assert res.n_tightened >= 1
    assert res.tightened_ub[1] >= expected - 1e-12  # never tighter than the safe z allows
    assert res.tightened_ub[1] == pytest.approx(expected, abs=1e-12)
    # Every feasible point with objective <= cutoff survives.
    for x in np.linspace(2.0, 5.0, 31):
        assert cutoff - x <= res.tightened_ub[1] + 1e-12


def test_dbbt_strict_refuses_without_safe_z(monkeypatch):
    monkeypatch.setenv(FLAG, "1")
    _shifted_ns(monkeypatch, none=True)
    res = obbt_mod.dbbt_on_relaxation(_dbbt_rel(), 2, 5.0)
    assert res.n_tightened == 0
    np.testing.assert_array_equal(res.tightened_ub, [10.0, 10.0])
    monkeypatch.setenv(FLAG, "0")
    assert obbt_mod.dbbt_on_relaxation(_dbbt_rel(), 2, 5.0).n_tightened >= 1


def test_strict_on_the_real_ns_bound_still_tightens_and_retains(monkeypatch):
    """Without the fake, strict is not vacuous on the panel and is never tighter than OFF."""
    n_tight = 0
    for make in PANEL.values():
        model = make()
        n = _n_orig(model)
        monkeypatch.setenv(FLAG, "0")
        off = obbt_mod.run_obbt_on_relaxation(_build_relaxation(model), n_orig=n)
        monkeypatch.setenv(FLAG, "1")
        on = obbt_mod.run_obbt_on_relaxation(_build_relaxation(model), n_orig=n)
        assert np.all(on.tightened_lb <= off.tightened_lb + 1e-12)
        assert np.all(on.tightened_ub >= off.tightened_ub - 1e-12)
        n_tight += on.n_tightened
    assert n_tight > 0

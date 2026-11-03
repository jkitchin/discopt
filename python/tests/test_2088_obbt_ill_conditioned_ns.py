"""#2088: on an ill-conditioned relaxation OBBT tightens only to the NS-safe bound.

OBBT may tighten over a relaxation past ``_OBBT_COND_LIMIT`` only because the
Neumaier-Shcherbina bound ``g`` is rigorous (``require_ns``). With
``DISCOPT_OBBT_NS_STRICT`` OFF, ``_ns_select`` still kept the vertex ``raw`` when it
sat inside the 1e-6 guard above ``g``. On the eps-perspective hull of GDPlib
clay0303 (``_cond`` ~1e16) a min probe returned raw 1.25e-7 against g = -6.9e-7,
the vertex was applied, the column's lb left its feasible value 0, the node holding
the optimum was fathomed, and the solve certified 28351.42 for 26669.11.

No small model makes the simplex return a vertex above its LP minimum, so the NS
bound is replaced by ``g_true - 5e-7`` (inside the legacy guard), as in
``test_1917_obbt_ns_strict.py``. With the flag OFF, a well-conditioned LP still
keeps ``raw`` (that file pins it); an ill-conditioned one must take ``g``. All
three tests below fail before the fix.

The end-to-end trigger is platform-dependent: which simplex vertex lands inside
the guard depends on floating-point and BLAS details. The clay0303 false
certificate reproduced on macOS (about 3 of 7 OFF runs), but a Linux cloud tracer
saw 0 events over 849 nodes and six seeds on the unfixed code. That is why these
tests pin ``_ns_select`` and force ``require_ns`` directly rather than solving
clay0303.
"""

from __future__ import annotations

import discopt._relax.obbt as obbt_mod
import discopt.modeling as dm
from test_1917_obbt_ns_strict import FLAG, SHIFT, _base, _n_orig, _shifted_ns
from test_obbt_warm_probes import PANEL, _build_relaxation


def test_ns_select_requires_g_on_ill_conditioned_lp():
    raw, g = 1.2499995349912751e-07, -6.944335331307157e-07  # the clay0303 probe
    assert obbt_mod._ns_select(raw, g, require_ns=True, strict=False) == g
    assert obbt_mod._ns_select(raw, None, require_ns=True, strict=False) is None
    # Well-conditioned legacy behaviour is unchanged.
    assert obbt_mod._ns_select(raw, g, require_ns=False, strict=False) == raw


def test_ill_conditioned_relaxation_obbt_takes_g_with_flag_off(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    # Every panel relaxation is now "ill-conditioned": OBBT runs on the NS path.
    monkeypatch.setattr(obbt_mod, "_OBBT_COND_LIMIT", 0.0)
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
                assert g is not None and res.tightened_lb[j] <= g
                checked += 1
            if res.tightened_ub[j] < ub0[j]:
                g = seen[(j, -1)]
                assert g is not None and res.tightened_ub[j] >= -g
                checked += 1
    assert checked > 0, "no bound tightened: the instrument measured nothing"
    print(f"EXECUTED {checked} ill-conditioned NS assertions")


def _ill_conditioned_model():
    # The 1e11 row puts the relaxation past _OBBT_COND_LIMIT without the patch.
    m = dm.Model("ill_cond_ns")
    x = m.continuous("x", lb=0.0, ub=10.0)
    y = m.continuous("y", lb=0.0, ub=10.0)
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.subject_to(x + y <= 4.0)
    m.subject_to(x - y >= -1.0)
    m.subject_to(1e11 * z <= 1e11)
    m.minimize(x + y + z)
    return m


def test_model_obbt_ill_conditioned_takes_g_with_flag_off(monkeypatch):
    monkeypatch.setenv(FLAG, "0")
    seen = _shifted_ns(monkeypatch)
    res = obbt_mod.run_obbt(_ill_conditioned_model())
    checked = 0
    for j in range(2):
        if res.tightened_ub[j] < 10.0:
            assert res.tightened_ub[j] >= -seen[(j, -1)]
            checked += 1
    assert checked > 0
    # x <= 4 is reported as x <= 4 + SHIFT (the rigorous bound), not the raw 4.
    assert res.tightened_ub[0] >= 4.0 + SHIFT - 1e-12

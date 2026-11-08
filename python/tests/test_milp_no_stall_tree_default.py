"""Python-MILP-tree retirement: DISCOPT_MILP_NO_STALL_TREE and
DISCOPT_MILP_RUST_OPEN_DEFAULT_BOX graduated together (default ON, 2026-11-08).

Default: run 1 of the Rust MILP engine is told it may extend its deadline when it
reaches the #291 stall slice with no incumbent (``extend_no_incumbent_s``), instead of
handing the MILP to the Python tree; and a default-box column side reaches the engine
as the 1e20 infinity, not as the finite 9.999e19. ``=0`` restores each old behaviour.
"""

from __future__ import annotations

import discopt._rust as _rust
import discopt.modeling as dm
import numpy as np
import pytest


def _spy(monkeypatch):
    calls: list[tuple[tuple, dict]] = []
    real = _rust.solve_milp_csc_py

    def spy(*a, **kw):
        calls.append((a, kw))
        return real(*a, **kw)

    monkeypatch.setattr(_rust, "solve_milp_csc_py", spy)
    return calls


def _model():
    m = dm.Model("nst")
    x = m.integer("x", lb=0, ub=10)
    y = m.continuous("y", lb=0)  # default upper side
    m.subject_to(3 * x + y >= 7.5)
    m.subject_to(x - y <= 4)
    m.minimize(2 * x + y)
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("flags_on", [True, False])
def test_engine_gets_the_stall_extension_and_open_default_box(monkeypatch, flags_on):
    monkeypatch.setenv("DISCOPT_LP_MILP_BACKEND", "rust")
    if not flags_on:
        # The no-stall continuation has no opt-out since the tree was deleted.
        monkeypatch.setenv("DISCOPT_MILP_RUST_OPEN_DEFAULT_BOX", "0")
    calls = _spy(monkeypatch)
    r = _model().solve(time_limit=30)
    if flags_on:
        assert r.status == "optimal" and r.gap_certified, (r.status, r.algorithm_route)
    else:
        # Measured 2026-11-08: with the default box left at 9.999e19 the engine's
        # vertices sit on it, no bound can be verified, and the solve ends
        # ``feasible`` uncertified -- the defect the graduation removes.
        assert r.status in ("optimal", "feasible"), r.status
    assert r.objective == pytest.approx(5.5, abs=1e-6)  # x = 2, y = 1.5
    assert calls, "the Rust MILP engine never ran"
    a, kw = calls[0]
    assert "extend_no_incumbent_s" in kw, kw.keys()
    x_u = np.asarray(a[8])
    y_ub = float(x_u[1])
    if flags_on:
        assert y_ub >= 1e20, y_ub
    else:
        assert 9e19 < y_ub < 1e20, y_ub

"""FBBT interval arithmetic must round outward (adversarial fuzz, seed 9219).

Round-to-nearest interval endpoints are not enclosures. On

    min  x2 - 1/(x0 + 10)
    s.t. (-3460*x2**3 + -1.15) + 3461.89 >= 0,   x0 in {-2..0},  x2 binary

both ``x2 = 0`` (slack 3460.74) and ``x2 = 1`` (slack 0.74) are feasible and the
optimum is ``x2 = 0`` at ``-0.125``. The Rust FBBT backward pass computed the
cube's preimage as ``<= -9.1e-14`` (exact: ``<= 0``); the odd root amplified that
to ``x2 >= 2.97e-6`` and integrality snapped it to ``x2 = 1``. The solve then
certified ``optimal`` at ``0.875`` with a dual bound of ``0.875`` -- above the true
optimum. The same row with the constants pre-folded (``+ 3460.74``) did not round
and solved correctly.

The fix makes FBBT's arithmetic outward-rounded (``crates/discopt-core``
``presolve/fbbt.rs``; opt-out ``DISCOPT_FBBT_OUTWARD_ROUND=0``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import discopt.modeling as dm
import pytest
from discopt._rust import model_to_repr

TRUE_OPT = -0.125


def _split_row_model(var_type: str = "binary") -> dm.Model:
    m = dm.Model("fbbt_outward")
    x0 = m.integer("x0", lb=-2, ub=0)
    x2 = m.binary("x2") if var_type == "binary" else m.continuous("x2", lb=0, ub=1)
    m.minimize(x2 - 1 / (x0 + 10))
    m.subject_to((-3460 * x2**3 + -1.15) + 3461.89 >= 0)
    return m


@pytest.mark.parametrize("var_type", ["binary", "continuous"])
def test_fbbt_does_not_cut_off_the_feasible_zero(var_type):
    m = _split_row_model(var_type)
    lbs, ubs = model_to_repr(m, None).fbbt(max_iter=20, tol=1e-8, time_limit_ms=None)
    # x2 is the second variable; x2 = 0 and x2 = 1 are both feasible.
    assert float(lbs[1]) <= 0.0, f"FBBT raised x2's lower bound to {float(lbs[1])!r}"
    assert float(ubs[1]) >= 1.0


def test_solve_certifies_the_true_optimum():
    r = _split_row_model().solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(TRUE_OPT, abs=1e-6)
    assert r.bound is not None and r.bound <= TRUE_OPT + 1e-6, (
        f"dual bound {r.bound} crosses the true optimum {TRUE_OPT}"
    )
    assert float(r.x["x2"]) == pytest.approx(0.0, abs=1e-6)


def test_fixture_is_not_vacuous_under_the_legacy_arithmetic():
    """The opt-out restores the defect, so the tests above exercise it (CLAUDE.md §6).

    The flag is read once per process (``OnceLock``), hence the subprocess.
    """
    code = textwrap.dedent(
        """
        import discopt.modeling as dm
        from discopt._rust import model_to_repr
        m = dm.Model("legacy")
        x0 = m.integer("x0", lb=-2, ub=0)
        x2 = m.binary("x2")
        m.minimize(x2 - 1 / (x0 + 10))
        m.subject_to((-3460 * x2**3 + -1.15) + 3461.89 >= 0)
        lbs, _ = model_to_repr(m, None).fbbt(max_iter=20, tol=1e-8, time_limit_ms=None)
        print(repr(float(lbs[1])))
        """
    )
    env = dict(os.environ, DISCOPT_FBBT_OUTWARD_ROUND="0")
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True
    )
    legacy_lb = float(out.stdout.strip().splitlines()[-1])
    assert legacy_lb > 0.0, (
        "legacy round-to-nearest FBBT no longer rounds on this fixture; the "
        "regression tests above would pass vacuously -- pick a fixture that rounds"
    )


def test_terminal_polish_does_not_degrade_an_exactly_feasible_incumbent():
    """``min acosh(x)`` on ``[1, 3]``: optimum 0 at the bound ``x = 1``.

    With FBBT's outward rounding the incumbent-cutoff box for ``x`` is
    ``[1, 1 + 4 ulp]`` rather than an exact ``[1, 1]``, so the terminal polish
    treats ``x`` as free, re-solves over the declared box and stops a barrier
    distance inside (``x = 1 + 1.5e-10``, objective 1.74e-5 -- acosh's slope is
    infinite at 1). That point used to be adopted as an "unchanged" purification
    and the 1e-6 absolute gap then failed: ``optimal`` -> ``feasible``.
    """
    m = dm.Model("acosh_polish")
    x = m.continuous("x", lb=1.0, ub=3.0)
    m.minimize(dm.acosh(x))
    r = m.solve(time_limit=30)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(0.0, abs=1e-9)
    assert float(r.x["x"]) == 1.0

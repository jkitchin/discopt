"""#1616 D-09 / #1990: backward FBBT through a product whose other factor touches zero.

``k * tau * (1 - X) == X`` with ``X in [0.9, 1]`` makes ``(1 - X) in [0, 0.1]``,
which touches 0. Since ``X >= 0.9`` excludes 0 from the product, ``k * tau >= 9``
is a valid deduction, and it lifts ``k``, ``T`` and ``tau`` through the Arrhenius
row.

FBBT used to skip the division whenever the cofactor's interval contained 0. A
one-sided version of the deduction (``DISCOPT_FBBT_ONE_SIDED_DIV``) was retired by
#1616 without reaching main, sound but not net-positive on its 29-instance panel.
#1990 brought it back as ``DISCOPT_FBBT_EXTENDED_MUL`` (one-sided and straddling
cofactors, ``presolve/fbbt.rs::extended_mul_preimage``) and graduated it default ON:
on the current base #1616's waterund22/25 losses do not reproduce, and the looser
bound under a tighter box that ex1252a showed came from the node LP's #671 row
filter, which #1990's RLT strip-and-retry removes (``docs/dev/performance-plan.md``
§97, ``docs/dev/flag-retirement-audit.md``).

The flag is read once per process in Rust, so the ``=0`` opt-out is checked in a
subprocess.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys

import discopt.modeling as dm
import pytest
from discopt.tightening import fbbt_box


def _cstr():
    m = dm.Model("cstr")
    T = m.continuous("T", lb=300, ub=500)
    t = m.continuous("tau", lb=0, ub=60)
    k = m.continuous("k", lb=0, ub=10)
    X = m.continuous("X", lb=0.9, ub=1)
    m.subject_to(k == 1e7 * dm.exp(-8000 / T))
    m.subject_to(k * t * (1 - X) == X)
    m.minimize(2 * t + 2 * (T - 300))
    return m


@pytest.fixture(scope="module")
def box():
    return fbbt_box(_cstr())


def test_one_sided_product_is_divided(box):
    """``k * tau >= 9`` with ``tau <= 60`` gives ``k >= 0.15``, hence ``T >= 444.07``
    through the Arrhenius row, and the propagated ``k <= 1.1254`` gives
    ``tau >= 7.9975`` (#1616's recorded witness)."""
    T_lb, tau_lb, k_lb = (float(v) for v in box.lb[:3])
    assert k_lb == pytest.approx(0.15, rel=1e-9)
    assert T_lb == pytest.approx(444.0691, abs=1e-3)
    assert tau_lb == pytest.approx(7.9975, abs=1e-3)


def test_known_feasible_point_survives(box):
    """A point satisfying both rows must stay inside the tightened box."""
    T = 480.0
    k = 1e7 * math.exp(-8000 / T)
    X = 0.95
    tau = X / (k * (1 - X))
    checks = 0
    for v, lo, hi in zip((T, tau, k, X), box.lb, box.ub, strict=True):
        assert float(lo) - 1e-9 <= v <= float(hi) + 1e-9
        checks += 1
    assert checks == 4


def test_opt_out_restores_the_division_only_rule():
    code = (
        "import json, sys\n"
        f"sys.path.insert(0, {os.path.dirname(__file__)!r})\n"
        "from test_1616_fbbt_one_sided_div import _cstr\n"
        "from discopt.tightening import fbbt_box\n"
        "b = fbbt_box(_cstr())\n"
        "print(json.dumps([float(v) for v in b.lb[:3]]))\n"
    )
    env = dict(os.environ, DISCOPT_FBBT_EXTENDED_MUL="0")
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, check=True
    )
    T_lb, tau_lb, k_lb = json.loads(out.stdout.strip().splitlines()[-1])
    assert T_lb == pytest.approx(300.0)
    assert tau_lb == pytest.approx(0.0)
    assert k_lb < 1e-3

"""End-to-end solve equivalence for the Decomposition Advisor (Phase 5, live solve).

Confirms that a reformulation built by ``advisor.decompose()`` reproduces the
monolithic optimum — the correctness guarantee behind the ``SoundnessCertificate``.
Requires the real solver stack (the compiled ``_rust`` extension + the POUNCE
node-LP engine), so the module is skipped where those are unavailable.
"""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import pytest

pytest.importorskip("pounce", reason="end-to-end solve needs the POUNCE engine")

import discopt.modeling as dm  # noqa: E402
from discopt.decomposition import MethodKind, analyze_decomposition  # noqa: E402

pytestmark = pytest.mark.requires_pounce


def _two_stage_facility():
    """Two-stage MILP: 3 binaries (budget-coupled) gating 3 recourse blocks."""
    m = dm.Model("facility")
    b = [m.binary(f"b{i}") for i in range(3)]
    x = [m.continuous(f"x{i}", lb=0, ub=10) for i in range(3)]
    for i in range(3):
        m.subject_to(x[i] <= 10 * b[i])
        m.subject_to(x[i] >= 4 * b[i])  # if built, must produce >= 4
    m.subject_to(b[0] + b[1] + b[2] >= 2)  # budget couples the binaries
    m.minimize((x[0] + x[1] + x[2]) + 2 * (b[0] + b[1] + b[2]))
    return m


def test_benders_reproduces_monolithic_optimum():
    m = _two_stage_facility()
    mono = m.solve()
    assert mono.status == "optimal"

    adv = analyze_decomposition(m)
    assert adv.recommendation().recommendation is MethodKind.BENDERS

    dcmp = adv.decompose()
    res = dcmp.solve()
    assert res.status == "optimal"
    # classical Benders is proven-equivalent for linear recourse
    assert abs(float(res.objective) - float(mono.objective)) < 1e-5


def test_forced_none_matches_monolith():
    m = _two_stage_facility()
    mono = m.solve()
    # the NONE reformulation must be identical to solving the model as written
    dcmp = analyze_decomposition(m).decompose(method="none")
    res = dcmp.solve()
    assert abs(float(res.objective) - float(mono.objective)) < 1e-5

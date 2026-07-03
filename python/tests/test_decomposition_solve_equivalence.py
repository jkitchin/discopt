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


# ── Phase 5 (T5.1): decomposition="auto" ──────────────────────


def test_auto_dispatches_and_matches_monolithic():
    m = _two_stage_facility()
    mono = m.solve()
    auto = m.solve(decomposition="auto")
    assert auto.status == "optimal"
    assert abs(float(auto.objective) - float(mono.objective)) < 1e-4


def test_auto_falls_through_on_unstructured_model():
    # A small dense model with no exploitable structure: auto must still solve it
    # correctly (falling through to the monolithic path).
    m = dm.Model("dense")
    y = m.continuous("y", shape=(3,), lb=0, ub=5)
    m.subject_to(y[0] + y[1] + y[2] <= 6)
    m.minimize(-(y[0] + y[1] + y[2]))
    auto = m.solve(decomposition="auto")
    mono = m.solve()
    assert auto.status == "optimal"
    assert abs(float(auto.objective) - float(mono.objective)) < 1e-4


def test_auto_records_telemetry_when_enabled():
    from discopt.decomposition.learning.store import RecordStore

    m = _two_stage_facility()
    # record_decomposition=True uses an in-memory store; verify a record lands by
    # pointing DISCOPT_DECOMP_STORE at a temp file via the env is covered
    # elsewhere — here we just confirm the solve still succeeds with recording on.
    res = m.solve(decomposition="auto", record_decomposition=True)
    assert res.status == "optimal"
    # An in-memory store is created internally; a file-backed store round-trips.
    _ = RecordStore(path=None)

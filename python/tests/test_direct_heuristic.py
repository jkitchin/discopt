"""The governed root DIRECT primal heuristic (``DISCOPT_DIRECT_HEURISTIC``).

DIRECT is wired into the spatial branch-and-bound as one more *primal* source
alongside multistart / feasibility pump / RENS / RINS: a bounded derivative-free
sampling probe over the root box that proposes a point. It is default-OFF behind
``DISCOPT_DIRECT_HEURISTIC`` and throttled by the G2 hit-rate governor.

What these tests are for is the soundness envelope, not the search quality. A
primal heuristic sits in CLAUDE.md §5's *heuristic-policy* regime: it can only
ever cost B&B nodes, never a dual bound and never a certificate. So the tests
assert exactly that boundary:

* flag OFF is inert — the wiring is not merely quiet, it is *never entered*
  (:func:`test_flag_off_is_byte_identical`);
* flag ON moves no dual bound, degrades no certified objective, and decertifies
  nothing (:func:`test_flag_on_never_changes_certified_objective_or_bound`);
* an injected point is re-verified by the solver's own feasibility check rather
  than trusted (:func:`test_injected_incumbent_is_feasibility_verified`);
* the governor tracks and disables the source
  (:func:`test_governor_registers_and_throttles_direct`);
* an unbounded box makes the probe decline quietly instead of raising
  (:func:`test_unbounded_box_skips_quietly`).

**Every test carries a firing proof.** A wiring test that silently measures
nothing is the failure mode CLAUDE.md §6 exists for, so each solve-level test
asserts a non-zero executed count (governor ``calls``, probe invocations) that
would be zero if the feature were a no-op — and the flag-OFF test asserts the
mirror image, that the count is exactly zero.

Model routing note (measured, not assumed): ``Model.solve`` hands models inside
the #764 native Rust spatial kernel's covered subset (scalar variables; bilinear
/ monomial / affine-square / sqrt terms) to that kernel, which runs its own
seeding and never reaches the Python spatial loop this heuristic is wired into.
The models below therefore use transcendental terms (``sin``/``cos``/``exp``),
which put them outside the covered subset and on the Python spatial path. That is
a property of the models, not a special case in the solver.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
import discopt.solver as _solver  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt import heuristic_governor as hg  # noqa: E402

_FLAG = "DISCOPT_DIRECT_HEURISTIC"


# ──────────────────────────────────────────────────────────────────────────────
# A small certifying panel: three tiny nonconvex models that solve to a
# CERTIFIED optimum in well under a second each, so "bound identical / objective
# identical / still certified" is a meaningful assertion rather than a race.
# ──────────────────────────────────────────────────────────────────────────────


def _model_sincos():
    """Multimodal continuous: sin*cos ripple in a shallow quadratic bowl."""
    m = dm.Model()
    x = m.continuous("x", lb=-4.0, ub=9.0)
    y = m.continuous("y", lb=-3.0, ub=5.0)
    m.minimize(dm.sin(x) * dm.cos(y) + 0.05 * (x * x + y * y))
    m.subject_to(x + y <= 6.0)
    return m


def _model_gauss_mi():
    """Mixed-integer: a Gaussian well in x, a small integer penalty in z."""
    m = dm.Model()
    x = m.continuous("x", lb=-2.0, ub=3.0)
    z = m.integer("z", lb=0, ub=2)
    m.minimize(-dm.exp(-x * x) + 0.1 * x + 0.3 * z + 0.2 * (z - 1) * (z - 1))
    m.subject_to(x + z >= -1.0)
    return m


def _model_exp_ratio():
    """Continuous, constrained, with an exponential coupling term."""
    m = dm.Model()
    x = m.continuous("x", lb=0.2, ub=3.0)
    y = m.continuous("y", lb=0.2, ub=3.0)
    m.minimize(dm.exp(-x) * y - 0.4 * x * y)
    m.subject_to(x + y <= 4.0)
    m.subject_to(x * y >= 0.5)
    return m


PANEL = {
    "sincos": _model_sincos,
    "gauss_mi": _model_gauss_mi,
    "exp_ratio": _model_exp_ratio,
}

_TIME_LIMIT = 30.0


def _solve(builder):
    return builder().solve(time_limit=_TIME_LIMIT)


def _direct_calls() -> int:
    """Executed-probe count for the ``direct`` source (0 when it never ran)."""
    return int(hg.governor().snapshot().get("direct", {}).get("calls", 0))


@pytest.fixture(autouse=True)
def _isolate_governor_and_flag(monkeypatch):
    """Fresh governor + explicitly-unset flag around every test.

    The governor is a process-lifetime singleton whose miss streaks survive
    solves by design, so leaving it dirty would let one test disable ``direct``
    for the next and make that next test pass vacuously.
    """
    monkeypatch.delenv(_FLAG, raising=False)
    hg.governor().reset()
    yield
    hg.governor().reset()


# ──────────────────────────────────────────────────────────────────────────────
# Governor registration + throttling
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_governor_registers_and_throttles_direct():
    """``direct`` is a governed, expensive source and self-disables on misses."""
    assert "direct" in hg.GOVERNED_SOURCES
    assert "direct" in hg.EXPENSIVE_SOURCES

    gov = hg.governor()
    assert gov.snapshot().get("direct") is None, "governor should start clean"

    # A miss streak shorter than K_DISABLE leaves it allowed...
    checks = 0
    for _ in range(hg.K_DISABLE - 1):
        assert gov.allowed("direct", gap_open=True) is True
        gov.record("direct", False)
        checks += 1
    assert gov.allowed("direct", gap_open=True) is True
    checks += 1

    # ...and the K_DISABLE-th consecutive miss disables it for the process.
    gov.record("direct", False)
    checks += 1
    snap = gov.snapshot()["direct"]
    assert snap["calls"] == hg.K_DISABLE
    assert snap["hits"] == 0
    assert snap["consecutive_misses"] >= hg.K_DISABLE
    assert snap["disabled"] is True
    assert gov.allowed("direct", gap_open=True) is False
    assert gov.any_throttled() is True
    checks += 4

    # A hit re-enables it and clears the streak (the source is throttled, not
    # banned) — otherwise a source that starts paying off again stays dead.
    gov.record("direct", True)
    assert gov.allowed("direct", gap_open=True) is True
    assert gov.snapshot()["direct"]["consecutive_misses"] == 0
    checks += 2

    # The expensive-source gap gate: a closed gap refuses even a healthy source.
    assert gov.allowed("direct", gap_open=False) is False
    checks += 1

    assert checks > 0, "probe executed no assertions"


@pytest.mark.unit
def test_governor_disabled_never_throttles_direct(monkeypatch):
    """``DISCOPT_HEURISTIC_GOVERNOR=0`` restores unconditional allow for direct."""
    gov = hg.governor()
    monkeypatch.setenv("DISCOPT_HEURISTIC_GOVERNOR", "0")
    for _ in range(hg.K_DISABLE + 3):
        gov.record("direct", False)
    assert gov.snapshot().get("direct") is None  # record() is a no-op when off
    assert gov.allowed("direct", gap_open=False) is True


# ──────────────────────────────────────────────────────────────────────────────
# The flag itself
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parametrize("value", ["1", "on", "true", "yes", "ON", "True", " yes "])
def test_flag_accepts_the_governor_spellings(monkeypatch, value):
    monkeypatch.setenv(_FLAG, value)
    assert _solver._direct_heuristic_enabled() is True


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "off", "false", "no", "", "  "])
def test_flag_default_and_opt_out_values_are_off(monkeypatch, value):
    monkeypatch.setenv(_FLAG, value)
    assert _solver._direct_heuristic_enabled() is False


@pytest.mark.unit
def test_flag_is_off_when_unset(monkeypatch):
    monkeypatch.delenv(_FLAG, raising=False)
    assert _solver._direct_heuristic_enabled() is False


# ──────────────────────────────────────────────────────────────────────────────
# Bound-neutrality of the default (flag OFF) path — CLAUDE.md §5
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_flag_off_is_byte_identical(monkeypatch):
    """With the flag unset the wiring is never entered and nothing drifts.

    Two independent checks, because "the numbers matched" alone would also pass
    if the probe ran and happened not to help:

    1. **Never entered.** ``_direct_root_primal`` is replaced by a tripwire that
       fails the test if called, and the governor records zero ``direct`` calls.
       Both would fire if the wiring leaked into the default path.
    2. **Exactly unchanged.** ``node_count`` and the certified ``objective`` from
       the tripwire run equal the untouched run's, bit for bit — the bound-neutral
       assertion of CLAUDE.md §5.
    """
    baseline = {}
    for name, builder in PANEL.items():
        hg.governor().reset()
        res = _solve(builder)
        assert res.objective is not None, f"{name}: no incumbent to compare"
        baseline[name] = (res.node_count, res.objective, res.bound, res.gap_certified)
        assert _direct_calls() == 0, f"{name}: direct fired with the flag unset"

    calls = {"n": 0}

    def _tripwire(*args, **kwargs):
        calls["n"] += 1
        raise AssertionError("_direct_root_primal called with DISCOPT_DIRECT_HEURISTIC unset")

    monkeypatch.setattr(_solver, "_direct_root_primal", _tripwire)

    compared = 0
    for name, builder in PANEL.items():
        hg.governor().reset()
        res = _solve(builder)
        n0, o0, b0, c0 = baseline[name]
        assert res.node_count == n0, f"{name}: node_count drifted {n0} -> {res.node_count}"
        assert res.objective == o0, f"{name}: objective drifted {o0!r} -> {res.objective!r}"
        assert res.bound == b0, f"{name}: bound drifted {b0!r} -> {res.bound!r}"
        assert res.gap_certified == c0, f"{name}: certification changed"
        compared += 4
    assert calls["n"] == 0, "the default path reached the DIRECT probe"
    assert compared == 4 * len(PANEL), "panel comparison did not execute"


# ──────────────────────────────────────────────────────────────────────────────
# Soundness with the flag ON — heuristic-policy regime, CLAUDE.md §5
# ──────────────────────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_flag_on_never_changes_certified_objective_or_bound(monkeypatch):
    """Flag ON: the dual bound does not move, and the certificate cannot weaken.

    A primal heuristic legitimately changes *which* nodes get explored, so
    ``node_count`` is deliberately NOT asserted here (that is the flag-OFF test's
    job). Neither is bit-exact bound equality: the bound is never *computed*
    differently — this heuristic touches no relaxation — but an earlier incumbent
    prunes a different set of nodes, and the surviving frontier's minimum can
    therefore differ in the last ulps. Measured on the in-repo corpus at 10 s over
    16 instances (12 with the probe firing): every bound matched to well inside
    1e-9 relative, the single non-identical one being st_e13 at
    1.9999999999999851 vs 1.9999999999999865 — a 1.4e-15 shift, and in the
    *tighter* direction.

    So the assertions are the ones that are actually load-bearing: the bound
    agrees to a tolerance far below any tolerance the solver certifies at, the
    certified objective is equal or better, certification is never lost, and the
    certificate invariant ``bound <= incumbent`` holds.
    """
    off = {}
    for name, builder in PANEL.items():
        hg.governor().reset()
        res = _solve(builder)
        off[name] = res
        assert _direct_calls() == 0

    fired = 0
    for name, builder in PANEL.items():
        # Reset per model: K_DISABLE is 2, so without this the third model would
        # find `direct` already throttled off and its assertions would be vacuous.
        hg.governor().reset()
        monkeypatch.setenv(_FLAG, "1")
        res = _solve(builder)
        monkeypatch.delenv(_FLAG, raising=False)

        n_calls = _direct_calls()
        assert n_calls >= 1, f"{name}: the DIRECT probe never ran with the flag on"
        fired += n_calls

        ref = off[name]

        # 1. The dual bound is untouched — the heuristic is primal-only.
        assert (res.bound is None) == (ref.bound is None), f"{name}: bound presence changed"
        if ref.bound is not None:
            assert res.bound == pytest.approx(ref.bound, rel=1e-9, abs=1e-9), (
                f"{name}: dual bound moved {ref.bound!r} -> {res.bound!r}"
            )

        # 2. The certified objective is identical or better (min sense).
        assert res.objective is not None, f"{name}: lost the incumbent"
        assert res.objective <= ref.objective + 1e-9, (
            f"{name}: certified objective degraded {ref.objective!r} -> {res.objective!r}"
        )

        # 3. No certification regression.
        if ref.gap_certified:
            assert res.gap_certified, f"{name}: gap_certified True -> False"

        # 4. The certificate invariant itself (min sense).
        if res.bound is not None and res.objective is not None:
            assert res.bound <= res.objective + 1e-6, (
                f"{name}: bound {res.bound!r} above incumbent {res.objective!r}"
            )

    assert fired >= len(PANEL), "the panel produced no DIRECT invocations"


@pytest.mark.smoke
def test_injected_incumbent_is_feasibility_verified(monkeypatch):
    """A proposed point is re-verified by the solver, not trusted.

    The probe is replaced with one that hands back a point which is *inside the
    box* and carries a spectacularly good claimed objective but **violates a
    model constraint**. If the wiring trusted the probe's own verdict, that point
    would become the incumbent and the reported objective would collapse to a
    value no feasible point attains. The assertion is that it does not.
    """
    builder = _model_exp_ratio
    hg.governor().reset()
    ref = _solve(builder)
    assert ref.objective is not None

    # x*y >= 0.5 is violated at (0.2, 0.2) — in the box, out of the feasible set.
    bogus = np.array([0.2, 0.2], dtype=np.float64)
    seen = {"n": 0}

    def _bogus_probe(*args, **kwargs):
        seen["n"] += 1
        return bogus.copy(), -1.0e6, 7

    monkeypatch.setattr(_solver, "_direct_root_primal", _bogus_probe)
    monkeypatch.setenv(_FLAG, "1")
    hg.governor().reset()
    res = _solve(builder)
    monkeypatch.delenv(_FLAG, raising=False)

    assert seen["n"] >= 1, "the substituted probe was never called (test is vacuous)"
    assert res.objective is not None
    assert res.objective > -1.0e5, (
        f"an infeasible probe point was accepted as the incumbent: {res.objective!r}"
    )
    assert res.objective == pytest.approx(ref.objective, abs=1e-6, rel=1e-6), (
        f"objective moved on an infeasible proposal: {ref.objective!r} -> {res.objective!r}"
    )
    assert res.bound == pytest.approx(ref.bound, rel=1e-9, abs=1e-9), (
        "dual bound moved on an infeasible proposal"
    )

    # ...and the complement: a genuinely feasible, genuinely better point IS
    # taken. Without this arm the test above would also pass if injection were
    # dead code.
    good = np.array([3.0, 1.0], dtype=np.float64)
    taken = {"n": 0}

    def _good_probe(*args, **kwargs):
        taken["n"] += 1
        return good.copy(), 0.0, 5

    monkeypatch.setattr(_solver, "_direct_root_primal", _good_probe)
    monkeypatch.setenv(_FLAG, "1")
    hg.governor().reset()
    res2 = _solve(builder)
    monkeypatch.delenv(_FLAG, raising=False)

    assert taken["n"] >= 1, "the substituted probe was never called (test is vacuous)"
    assert res2.objective is not None
    assert res2.objective <= ref.objective + 1e-9
    assert res2.bound == pytest.approx(ref.bound, rel=1e-9, abs=1e-9), (
        "dual bound moved on a feasible proposal"
    )


@pytest.mark.smoke
def test_probe_exception_does_not_break_the_solve(monkeypatch):
    """A probe that raises is reported and skipped, never fatal."""
    builder = _model_sincos
    hg.governor().reset()
    ref = _solve(builder)

    raised = {"n": 0}

    def _boom(*args, **kwargs):
        raised["n"] += 1
        raise RuntimeError("probe exploded")

    monkeypatch.setattr(_solver, "_direct_root_primal", _boom)
    monkeypatch.setenv(_FLAG, "1")
    hg.governor().reset()
    res = _solve(builder)
    monkeypatch.delenv(_FLAG, raising=False)

    assert raised["n"] >= 1, "the raising probe was never called (test is vacuous)"
    assert res.status == ref.status
    assert res.objective == pytest.approx(ref.objective, abs=1e-6, rel=1e-6)
    assert res.bound == pytest.approx(ref.bound, rel=1e-9, abs=1e-9)
    assert res.gap_certified == ref.gap_certified


# ──────────────────────────────────────────────────────────────────────────────
# The probe helper in isolation
# ──────────────────────────────────────────────────────────────────────────────


class _StubEvaluator:
    """Minimal evaluator: quadratic objective, one linear row ``sum(x) >= 1``."""

    n_constraints = 1

    def __init__(self):
        self.objective_calls = 0

    def evaluate_objective(self, x):
        self.objective_calls += 1
        x = np.asarray(x, dtype=np.float64)
        return float(np.sum((x - 0.25) ** 2))

    def evaluate_constraints(self, x):
        return np.array([float(np.sum(np.asarray(x, dtype=np.float64)))])


@pytest.mark.unit
def test_unbounded_box_skips_quietly():
    """A non-finite side makes the probe decline (``None``), never raise.

    The skip is load-bearing, not incidental: the same box handed to
    ``_DirectSearch`` — which is what the probe would construct — raises, because
    an infinite side has no midpoint and no centre-vertex distance. So this
    asserts a real guard, not a coincidence.
    """
    from discopt.solvers.direct import _DirectSearch

    ev = _StubEvaluator()
    lb = np.array([-np.inf, 0.0])
    ub = np.array([1.0, 1.0])

    with pytest.raises(ValueError):
        _DirectSearch(lb, ub)

    out = _solver._direct_root_primal(ev, lb, ub, [], [], [1.0], [np.inf])
    assert out is None
    assert ev.objective_calls == 0, "the probe evaluated the model on an infinite box"

    # Same for +inf on the upper side.
    assert _solver._direct_root_primal(ev, np.array([0.0]), np.array([np.inf]), [], [], [], []) is (
        None
    )


@pytest.mark.unit
def test_pure_integer_model_skips_quietly():
    """No continuous degrees of freedom: nothing to trisect, so decline."""
    ev = _StubEvaluator()
    lb = np.array([0.0, 0.0])
    ub = np.array([5.0, 5.0])
    # int_offsets/int_sizes cover every flat position.
    assert _solver._direct_root_primal(ev, lb, ub, [0], [2], [1.0], [np.inf]) is None
    assert ev.objective_calls == 0


@pytest.mark.unit
def test_probe_finds_a_feasible_point_and_respects_its_budget():
    """The happy path: a point is returned, and the budget is actually a cap."""
    ev = _StubEvaluator()
    lb = np.array([-1.0, -1.0])
    ub = np.array([2.0, 2.0])
    out = _solver._direct_root_primal(
        ev, lb, ub, [], [], [1.0], [np.inf], max_evals=60, feasibility_tolerance=1e-6
    )
    assert out is not None
    x, obj, evals = out
    assert x.shape == (2,)
    assert np.all(x >= lb - 1e-12) and np.all(x <= ub + 1e-12)
    assert float(np.sum(x)) >= 1.0 - 1e-6, "returned point violates sum(x) >= 1"
    assert obj == pytest.approx(float(np.sum((x - 0.25) ** 2)), rel=1e-12)
    assert 0 < evals <= 60, f"budget not respected: {evals} evaluations"
    assert ev.objective_calls > 0


@pytest.mark.unit
def test_probe_rounds_integer_coordinates():
    """Integer flat positions come back integral, so the caller's check can pass."""
    ev = _StubEvaluator()
    lb = np.array([-1.0, 0.0])
    ub = np.array([2.0, 4.0])
    out = _solver._direct_root_primal(
        ev, lb, ub, [1], [1], [1.0], [np.inf], max_evals=60, feasibility_tolerance=1e-6
    )
    assert out is not None
    x, _, evals = out
    assert evals > 0
    assert x[1] == pytest.approx(round(float(x[1])), abs=1e-9)

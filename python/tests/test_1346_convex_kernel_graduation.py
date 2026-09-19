"""#1346: ``DISCOPT_CONVEX_KERNEL`` graduates default-ON behind a root-bound guard.

Two things ship together and each needs its own pin:

1. **The default flip.** The flag became an opt-*out*. It was never a failed panel
   that kept it off -- #798 proved both §5 bars and the 66-instance Regime-2 panel
   came back cert-clean; #800 then deferred graduation to #807's *SCIP wall parity*,
   a bar strictly above what §5 asks for. #1346 re-ran the gate and acted on it.

   Graduation panel (``issue1346_convex_kernel_graduation_panel.py``, in-repo
   66-instance corpus, ``time_limit=60``, ``deterministic=True``, arms interleaved
   within each instance with the arm order alternated by index, idle machine)::

       clay0303hfsg   off  feasible/UNCERTIFIED  obj 29911.20 (12.2% above opt)  ->
                      on   optimal/CERTIFIED     obj 26669.1096                  GAIN
       syn05hfsg      off  optimal   277 nodes  22.9 s  ->  on  optimal  2 nodes  0.01 s

2. **The guard.** A cap alone does not make the kernel safe as a default: the
   recorded counter-case (parity analysis G-C) classifies convex and then spends the
   whole attempt *with no bound*, which under #911's deduction leaves the default
   path nothing. The guard probes a short slice of the budget and declines when no
   finite bound came back. See ``_run_guarded_tree`` for the falsification criterion
   -- the counter-case class is NOT in the in-repo corpus, so the guard is a bound on
   the damage rather than a proof of its absence, and these tests pin the mechanism,
   not the counter-case.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import discopt.modeling as dm  # noqa: E402
import pytest  # noqa: E402
from discopt.solvers import _convex_kernel as ck  # noqa: E402


@pytest.fixture
def flag(monkeypatch):
    def _set(name, value):
        if value is None:
            monkeypatch.delenv(name, raising=False)
        else:
            monkeypatch.setenv(name, value)

    return _set


# --------------------------------------------------------------------------- #
# 1. the default flip
# --------------------------------------------------------------------------- #


def test_kernel_defaults_to_on(flag):
    """Graduated by the #1346 panel: cert-clean AND net-positive."""
    flag("DISCOPT_CONVEX_KERNEL", None)
    assert ck.convex_kernel_enabled() is True


@pytest.mark.parametrize(
    "value,expected",
    [("0", False), ("", False), ("false", False), ("False", False), ("1", True), ("on", True)],
)
def test_kernel_opt_out_is_preserved(flag, value, expected):
    """§5 keeps the ``=0`` opt-out and the legacy path intact on graduation."""
    flag("DISCOPT_CONVEX_KERNEL", value)
    assert ck.convex_kernel_enabled() is expected


def test_guard_defaults_to_on_and_can_be_opted_out(flag):
    flag("DISCOPT_CONVEX_KERNEL_GUARD", None)
    assert ck.convex_kernel_guard_enabled() is True
    flag("DISCOPT_CONVEX_KERNEL_GUARD", "0")
    assert ck.convex_kernel_guard_enabled() is False


# --------------------------------------------------------------------------- #
# 2. the guard's decision rule
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "bound,finite",
    [
        (None, False),
        (float("nan"), False),
        (float("inf"), False),
        (1e20, False),  # the LP layer's INF sentinel is 1e20, not f64::INFINITY
        (-1e20, False),
        (1e21, False),
        (1e19, True),  # below the sentinel: a real, if huge, bound
        (0.0, True),
        (-3.5, True),
    ],
)
def test_bound_finiteness_is_sentinel_aware(bound, finite):
    """CLAUDE.md: ``INF`` in the Rust LP layer is ``1e20``. Testing against
    ``math.isinf`` would read the sentinel as an ordinary finite bound and let the
    counter-case straight through the guard."""
    assert ck._bound_is_finite(bound) is finite


def _fake_tree(results, probe_cost: float = 0.0):
    """A stand-in for ``solve_convex_tree`` that returns `results` in order and
    records the budget it was handed on each call.

    ``probe_cost`` makes the first call actually consume wall, so the deduction of
    the probe from stage 2's budget is exercised rather than assumed -- with an
    instant mock the probe costs ~0 and stage 2 legitimately gets back almost the
    whole budget, which is correct behaviour and proves nothing about deduction.
    """
    import time as _time

    calls: list[float] = []

    def _tree(spec, *, time_limit_s=None, **cfg):
        calls.append(time_limit_s)
        if len(calls) == 1 and probe_cost:
            _time.sleep(probe_cost)
        return results[min(len(calls) - 1, len(results) - 1)]

    return _tree, calls


def test_guard_declines_without_a_finite_bound_and_keeps_the_budget(monkeypatch, flag):
    """The counter-case signature: the probe comes back with no bound, so the guard
    declines *after the probe* instead of after the whole budget.

    This is the property the graduation rests on -- without it, default-ON turns a
    60 s solve of such a model into nothing, because #911 deducts the attempt."""
    flag("DISCOPT_CONVEX_KERNEL_GUARD", "1")
    tree, calls = _fake_tree([{"status": "limit", "bound": None, "node_count": 7}])
    monkeypatch.setattr(ck, "solve_convex_tree", tree)

    out = ck._run_guarded_tree({}, budget=60.0, gap_tolerance=1e-4)

    assert out is None, "a no-bound probe must decline"
    assert calls == [3.0], "the guard must spend only the probe, never the full budget"
    reason, probe_bound, probe_nodes = ck.last_guard_decision()
    assert reason == "declined_no_finite_bound"
    assert (probe_bound, probe_nodes) == (None, 7)


def test_guard_continues_with_the_remaining_budget_when_a_bound_exists(monkeypatch, flag):
    flag("DISCOPT_CONVEX_KERNEL_GUARD", "1")
    probe = {"status": "limit", "bound": -12.5, "node_count": 4}
    final = {"status": "optimal", "bound": -1.0, "node_count": 99}
    tree, calls = _fake_tree([probe, final], probe_cost=0.25)
    monkeypatch.setattr(ck, "solve_convex_tree", tree)

    out = ck._run_guarded_tree({}, budget=60.0, gap_tolerance=1e-4)

    assert out is final
    assert len(calls) == 2 and calls[0] == 3.0
    # Stage 2 is charged the probe's ACTUAL wall, not its allowance: a probe that
    # returns early has not spent the budget and must not be billed for it. What
    # must hold is that the two stages together never exceed the caller's budget --
    # the whole point of #911's deduction.
    assert calls[1] <= 60.0 - 0.25 + 1e-3
    assert calls[0] + calls[1] <= 60.0 + 3.0, "the attempt must stay within budget"
    assert ck.last_guard_decision()[0] == "continued"


@pytest.mark.parametrize(
    "status,reason", [("optimal", "probe_certified"), ("infeasible", "probe_infeasible")]
)
def test_a_probe_that_settles_costs_nothing_extra(monkeypatch, flag, status, reason):
    """The fast win: the kernel finishes inside the probe, so there is no restart and
    the guard adds exactly zero overhead (measured: ``syn05hfsg`` 0.01 s)."""
    flag("DISCOPT_CONVEX_KERNEL_GUARD", "1")
    settled = {"status": status, "bound": 1.0, "node_count": 2}
    tree, calls = _fake_tree([settled, {"status": "limit", "bound": None, "node_count": 0}])
    monkeypatch.setattr(ck, "solve_convex_tree", tree)

    out = ck._run_guarded_tree({}, budget=60.0, gap_tolerance=1e-4)

    assert out is settled
    assert len(calls) == 1, "a settled probe must not restart the tree"
    assert ck.last_guard_decision()[0] == reason


def test_guard_opt_out_restores_the_single_shot_attempt(monkeypatch, flag):
    flag("DISCOPT_CONVEX_KERNEL_GUARD", "0")
    r = {"status": "limit", "bound": None, "node_count": 0}
    tree, calls = _fake_tree([r])
    monkeypatch.setattr(ck, "solve_convex_tree", tree)

    out = ck._run_guarded_tree({}, budget=60.0, gap_tolerance=1e-4)

    assert out is r, "with the guard off the attempt is handed straight back"
    assert calls == [60.0], "the whole budget goes to one call, as before #1346"
    assert ck.last_guard_decision()[0] == "disabled"


def test_a_budget_no_larger_than_a_probe_is_not_split(monkeypatch, flag):
    """Splitting a 1 s budget would pay the restart twice for no extra information."""
    flag("DISCOPT_CONVEX_KERNEL_GUARD", "1")
    r = {"status": "limit", "bound": 3.0, "node_count": 1}
    tree, calls = _fake_tree([r])
    monkeypatch.setattr(ck, "solve_convex_tree", tree)

    ck._run_guarded_tree({}, budget=1.0, gap_tolerance=1e-4)

    assert calls == [1.0]
    assert ck.last_guard_decision()[0] == "single_shot"


# --------------------------------------------------------------------------- #
# 3. end to end -- the guard never changes an answer, only who computes it
# --------------------------------------------------------------------------- #


def _convex_minlp() -> dm.Model:
    """Composite-of-affine convex, linear objective: in the kernel's scope."""
    m = dm.Model("cvx")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.integer("y", lb=0, ub=4)
    z = m.continuous("z", lb=0.0, ub=50.0)
    m.minimize(z + 2 * y)
    m.subject_to((x - 3.0) ** 2 <= z)
    m.subject_to(x + y >= 3)
    return m


@pytest.mark.parametrize("guard", ["0", "1"])
@pytest.mark.parametrize("kernel", ["0", "1"])
def test_the_answer_does_not_depend_on_either_switch(flag, guard, kernel):
    """Four combinations, one answer. The kernel is a *route*, not a relaxation: it
    is adopted only when it certifies and its incumbent verifies against the pristine
    model (#779), so neither switch may move the objective or invert the bound."""
    flag("DISCOPT_CONVEX_KERNEL", kernel)
    flag("DISCOPT_CONVEX_KERNEL_GUARD", guard)
    r = _convex_minlp().solve(time_limit=60)
    assert r.objective is not None
    assert r.objective == pytest.approx(2.0, abs=1e-4)
    if r.bound is not None:
        assert r.bound <= r.objective + 1e-6, "UNSOUND: bound above incumbent (min)"


def test_a_model_out_of_scope_never_reaches_the_guard(flag):
    """``build_convex_spec`` declines first, so the guard is unreachable for a
    non-convex model -- which is why the graduation panel's ineligible instances are
    byte-identical between the arms."""
    flag("DISCOPT_CONVEX_KERNEL", "1")
    flag("DISCOPT_CONVEX_KERNEL_GUARD", "1")
    m = dm.Model("bilinear")
    x = m.continuous("x", lb=0.0, ub=2.0)
    y = m.continuous("y", lb=0.0, ub=2.0)
    k = m.integer("k", lb=0, ub=2)
    m.minimize(x * y + k)  # bilinear: not composite-of-affine convex
    m.subject_to(x + y + k >= 1)
    r = m.solve(time_limit=60)
    assert r.objective is not None
    assert ck.last_guard_decision()[0] in ("not_run", "not_eligible")

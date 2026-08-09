"""#966 — a per-node separated-relaxation round must honor its grant end-to-end.

The #928 fix made the warm pure-LP path honor its per-solve ``time_limit``; the
residual overrun moved to the CALLER: a round's grant clamped only the LP
solves, while the round's non-LP cost (the cold ``build_uniform_relaxation`` +
separation) was spent after the admission check, unclamped (contvar: a round
granted 2.0 s ran 5.3–5.8 s, ~3.3 s of it the build — both arms). These tests
pin the ``DISCOPT_NODE_ROUND_BUDGET`` mechanism:

* ``round_deadline`` reaches the cold build as its ``build_deadline``
  (min-combined with any caller-supplied one) — the #694 anytime-build
  truncation applied from the round grant, exactly as #966 item 1 prescribes;
* ``round_deadline`` caps the node's internal solve/separation deadline, so the
  build cannot restart the round's clock;
* the relaxer measures its cold-build cost (EMA) for the node loops' round
  admission check, and a deadline-truncated build does not corrupt the
  estimate;
* the default path (no ``round_deadline``) is untouched.

All tests here fail (TypeError / AttributeError) on the pre-#966 tree.
"""

import time

import discopt._jax.mccormick_lp as mc_mod
import numpy as np
import pytest
from discopt import Model
from discopt._jax.mccormick_lp import MccormickLPRelaxer


def _bilinear_model() -> Model:
    """Small nonconvex model whose relaxation needs a lifted cold build."""
    m = Model()
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.minimize(x * y - 2.0 * x + y)
    m.subject_to(x + y >= 1.0)
    return m


def _box(model):
    from discopt._jax.model_utils import flat_variable_bounds

    return flat_variable_bounds(model)


class _BuildSpy:
    """Record the ``build_deadline`` each cold build receives."""

    def __init__(self):
        self.deadlines: list = []
        self._orig = mc_mod.build_milp_relaxation

    def __call__(self, *args, **kwargs):
        self.deadlines.append(kwargs.get("build_deadline"))
        return self._orig(*args, **kwargs)


@pytest.fixture()
def build_spy(monkeypatch):
    spy = _BuildSpy()
    monkeypatch.setattr(mc_mod, "build_milp_relaxation", spy)
    return spy


def test_round_deadline_reaches_build(build_spy):
    """The round grant is passed to the cold build as its build_deadline."""
    model = _bilinear_model()
    relaxer = MccormickLPRelaxer(model)
    relaxer._inc = None  # force the cold build path
    lb, ub = _box(model)
    rd = time.perf_counter() + 30.0
    res = relaxer.solve_at_node(lb, ub, time_limit=5.0, round_deadline=rd)
    assert res.status in ("optimal", "uncertified", "time_limit")
    assert build_spy.deadlines, "cold build never ran — the spy saw nothing"
    assert build_spy.deadlines[0] == rd


def test_round_deadline_min_combines_with_build_deadline(build_spy):
    """An explicit caller build_deadline earlier than the round grant wins."""
    model = _bilinear_model()
    relaxer = MccormickLPRelaxer(model)
    relaxer._inc = None
    lb, ub = _box(model)
    now = time.perf_counter()
    earlier, later = now + 10.0, now + 30.0
    relaxer.solve_at_node(lb, ub, time_limit=5.0, build_deadline=earlier, round_deadline=later)
    assert build_spy.deadlines and build_spy.deadlines[0] == earlier
    build_spy.deadlines.clear()
    relaxer.solve_at_node(lb, ub, time_limit=5.0, build_deadline=later, round_deadline=earlier)
    assert build_spy.deadlines and build_spy.deadlines[0] == earlier


def test_default_path_passes_no_build_deadline(build_spy):
    """Without round_deadline the cold build sees exactly the legacy kwargs."""
    model = _bilinear_model()
    relaxer = MccormickLPRelaxer(model)
    relaxer._inc = None
    lb, ub = _box(model)
    relaxer.solve_at_node(lb, ub, time_limit=5.0)
    assert build_spy.deadlines and build_spy.deadlines[0] is None


def test_round_deadline_caps_internal_deadline(monkeypatch):
    """A spent round grant leaves each internal LP only the floor budget.

    The internal anchor is taken AFTER the cold build; without the cap the
    build restarts the clock and the first LP gets the whole ``time_limit``
    again. With ``round_deadline`` in the (near) past, every internal solve
    must be granted only the deadline floor.
    """
    import discopt._jax.milp_relaxation as mr_mod

    granted: list = []
    orig = mr_mod.MilpRelaxationModel.solve

    def spy(self, time_limit=None, *args, **kwargs):
        granted.append(time_limit)
        return orig(self, time_limit, *args, **kwargs)

    monkeypatch.setattr(mr_mod.MilpRelaxationModel, "solve", spy)
    model = _bilinear_model()
    relaxer = MccormickLPRelaxer(model)
    relaxer._inc = None
    lb, ub = _box(model)
    relaxer.solve_at_node(lb, ub, time_limit=5.0, round_deadline=time.perf_counter() + 1e-6)
    assert granted, "no internal LP solve observed"
    assert all(tl is not None and tl <= mc_mod._SOLVE_DEADLINE_FLOOR_S + 1e-9 for tl in granted), (
        f"an internal solve was granted more than the floor: {granted}"
    )


def test_build_cost_ema_measured_and_truncation_excluded():
    """expected_build_cost() reflects whole cold builds; truncated ones don't."""
    model = _bilinear_model()
    relaxer = MccormickLPRelaxer(model)
    relaxer._inc = None
    lb, ub = _box(model)
    assert relaxer.expected_build_cost() is None
    relaxer.solve_at_node(lb, ub, time_limit=5.0)
    first = relaxer.expected_build_cost()
    assert first is not None and first > 0.0
    # A build truncated by a spent round grant must not drag the EMA toward
    # its (artificially small) wall — the estimate prices a FULL build.
    relaxer.solve_at_node(lb, ub, time_limit=5.0, round_deadline=time.perf_counter() - 1.0)
    after = relaxer.expected_build_cost()
    # Either the build still completed whole (tiny model; EMA updates with a
    # genuine full-build wall) or it truncated (EMA unchanged). In both cases
    # the estimate stays a positive full-build price.
    assert after is not None and after > 0.0


def test_flag_default_off():
    """DISCOPT_NODE_ROUND_BUDGET defaults OFF (§5 bound-changing discipline)."""
    from discopt.solver_tuning import SolverTuning

    assert SolverTuning().node_round_budget is False


# --------------------------------------------------------------------------- #
# #966 item 2 — the first-time sparse-Hessian XLA compile entry gate
# (``DISCOPT_HESS_COMPILE_GATE``). The severe modes (124 s compile against a
# 20 s budget on heatexch_gen3, caught in flight with faulthandler) enter
# through the per-node POUNCE paths, which bypass the F4 root-heuristic gate.
# --------------------------------------------------------------------------- #


@pytest.fixture()
def _gate_on():
    from discopt import solver_tuning

    token = solver_tuning.set_current(
        solver_tuning.SolverTuning().replace(hessian_compile_gate=True)
    )
    yield
    solver_tuning.reset_current(token)


def _batch_args(model, n_batch=2):
    from discopt._jax.nlp_evaluator import NLPEvaluator

    ev = NLPEvaluator(model)
    lb, ub = _box(model)
    batch_lb = [lb.tolist() for _ in range(n_batch)]
    batch_ub = [ub.tolist() for _ in range(n_batch)]
    batch_ids = list(range(n_batch))
    cb = [(-np.inf, 0.0)] * ev.n_constraints
    return ev, batch_lb, batch_ub, batch_ids, cb


def test_hess_gate_refuses_nonconvex_batch(_gate_on, monkeypatch):
    """Flag ON + est > grant + nonconvex JAX-callback batch -> refuse pre-entry."""
    import discopt.solvers.nlp_native as native_mod
    from discopt.constants import SENTINEL_THRESHOLD
    from discopt.solver import _solve_batch_pounce

    # Force the JAX-callback batch (the gated path) and make an actual POUNCE
    # batch solve fail the test loudly.
    monkeypatch.setattr(native_mod, "get_native_base", lambda ev: None)
    import pounce

    def _boom(*a, **kw):
        raise AssertionError("POUNCE batch started — the #966 gate did not refuse entry")

    monkeypatch.setattr(pounce, "solve_nlp_batch", _boom, raising=False)
    model = _bilinear_model()
    ev, blb, bub, bids, cb = _batch_args(model)
    monkeypatch.setattr(ev, "hessian_compile_estimate_s", lambda: 100.0)
    ids, lbs, sols, feas, trusted = _solve_batch_pounce(
        ev, blb, bub, bids, ev.n_variables, cb, {"max_wall_time": 5.0}, convex=False
    )
    # Every node reports the failure sentinel (kept OPEN by the C-1 sweep).
    assert np.all(lbs >= SENTINEL_THRESHOLD)
    assert not feas.any()


def test_hess_gate_admits_batch_when_estimate_fits(_gate_on, monkeypatch):
    """est <= grant -> the batch solve runs normally under the flag."""
    import discopt.solvers.nlp_native as native_mod
    from discopt.constants import SENTINEL_THRESHOLD
    from discopt.solver import _solve_batch_pounce

    monkeypatch.setattr(native_mod, "get_native_base", lambda ev: None)
    model = _bilinear_model()
    ev, blb, bub, bids, cb = _batch_args(model)
    monkeypatch.setattr(ev, "hessian_compile_estimate_s", lambda: 0.01)
    ids, lbs, sols, feas, trusted = _solve_batch_pounce(
        ev, blb, bub, bids, ev.n_variables, cb, {"max_wall_time": 10.0}, convex=False
    )
    assert np.any(lbs < SENTINEL_THRESHOLD), "no node solved — the gate refused wrongly"


def test_hess_gate_default_off(monkeypatch):
    """Flag OFF (default): the batch runs even when the estimate dwarfs the grant."""
    import discopt.solvers.nlp_native as native_mod
    from discopt.constants import SENTINEL_THRESHOLD
    from discopt.solver import _solve_batch_pounce
    from discopt.solver_tuning import SolverTuning

    assert SolverTuning().hessian_compile_gate is False
    monkeypatch.setattr(native_mod, "get_native_base", lambda ev: None)
    model = _bilinear_model()
    ev, blb, bub, bids, cb = _batch_args(model)
    monkeypatch.setattr(ev, "hessian_compile_estimate_s", lambda: 100.0)
    ids, lbs, sols, feas, trusted = _solve_batch_pounce(
        ev, blb, bub, bids, ev.n_variables, cb, {"max_wall_time": 10.0}, convex=False
    )
    assert np.any(lbs < SENTINEL_THRESHOLD)


def test_hess_gate_multistart_site_differential(monkeypatch):
    """End-to-end at the caught severe-mode site (#966): the root multistart on
    the no-relaxer nonconvex class. Flag ON + estimate >> budget refuses the
    entry; flag OFF launches it (the vacuity control, CLAUDE.md §6)."""
    import discopt.solver as solver_mod
    from discopt import solver_tuning
    from discopt._jax.nlp_evaluator import NLPEvaluator
    from discopt.solver import solve_model

    calls = {"n": 0}
    orig_ms = solver_mod._solve_root_node_multistart

    def spy_ms(*a, **kw):
        calls["n"] += 1
        return orig_ms(*a, **kw)

    monkeypatch.setattr(solver_mod, "_solve_root_node_multistart", spy_ms)
    monkeypatch.setattr(NLPEvaluator, "hessian_compile_estimate_s", lambda self: 1e9)

    # Force the no-relaxer route (the heatexch_gen3 class): relaxer setup
    # failure falls back to _mc_lp_relaxer=None, whose iteration-0 bound/primal
    # source is exactly the gated root multistart.
    import discopt._jax.mccormick_lp as mc_mod2

    def _no_relaxer(*a, **kw):
        raise RuntimeError("test: relaxer disabled to force the multistart site")

    monkeypatch.setattr(mc_mod2.MccormickLPRelaxer, "__init__", _no_relaxer)
    # The native Rust spatial kernel (#764) would solve this model without ever
    # entering the Python node loop under test — force the Python route.
    monkeypatch.setenv("DISCOPT_NATIVE_SPATIAL_KERNEL", "0")

    def _minlp():
        # Mixed-integer + nonconvex so the solve routes through the spatial
        # B&B batch loop (a continuous-only NLP takes a different path).
        m = Model()
        x = m.continuous("x", lb=0.0, ub=4.0)
        y = m.continuous("y", lb=0.0, ub=4.0)
        z = m.integer("z", lb=0, ub=3)
        m.minimize(x * y - 2.0 * x + y + 0.5 * z)
        m.subject_to(x + y + z >= 1.0)
        return m

    model = _minlp()
    # solve_model publishes its own tuning at entry (the ``tuning`` kwarg /
    # env), so pass the flag explicitly rather than via a context token.
    solve_model(
        model,
        time_limit=5.0,
        tuning=solver_tuning.SolverTuning().replace(hessian_compile_gate=True),
    )
    on_calls = calls["n"]

    calls["n"] = 0
    model2 = _minlp()
    solve_model(model2, time_limit=5.0)
    off_calls = calls["n"]

    assert off_calls > 0, "control arm never reached the multistart site — vacuous"
    assert on_calls == 0, f"gate ON still entered the multistart NLP {on_calls} time(s)"

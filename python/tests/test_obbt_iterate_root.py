"""#282 — iterate root OBBT to convergence over the McCormick LP (QCQP root bound).

The root OBBT over the McCormick LP (``obbt_tighten_root``) already iterates and
rebuilds the envelope on the tightened box each sweep, but the default ``rounds=3``
cap stops far short of the fixpoint on a *wide-box dense QCQP*: tightening ``x_i``
shrinks the McCormick envelopes, which unlocks tightening ``x_j``, so the root
bound only converges after many sweeps. The ``DISCOPT_OBBT_ITERATE`` lever
(default OFF) raises that cap for the quadratic/wide-box structural class and adds
a ``min_improvement`` convergence early-stop.

These tests lock the two properties that make the lever a *win* and *safe*:

  1. **Iterated >> one-pass (and default) on QCQP** — on a vendored dense integer
     QCQP whose McCormick envelope needs >3 sweeps to converge, the iterated root
     bound is dramatically tighter than a single OBBT pass and meaningfully
     tighter than the ``rounds=3`` default.
  2. **Sound** — differential (the iterated bound is a *valid* lower bound: it
     never exceeds the true box optimum and is never looser than the one-pass
     bound) plus feasible-point sampling (no feasible point of the original model,
     in particular the global optimum, is cut by the tightened box).
  3. **OFF path is inert** — ``min_improvement=None`` reproduces the legacy loop
     bit-for-bit, and the solve-path flag helper defaults OFF.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt._jax.mccormick_lp import MccormickLPRelaxer, build_milp_relaxation
from discopt._jax.obbt import obbt_tighten_root
from discopt.modeling.core import Model
from discopt.solvers import SolveStatus
from discopt.solvers.lp_backend import get_exact_dual_lp_solver, get_exact_lp_solver

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


# --- Vendored dense integer QCQP -------------------------------------------
# Minimize -sum_{i<j} x_i x_j  s.t.  sum_i x_i^2 <= 400,  x in [0, UB]^N integer.
# The single sum-of-squares coupling over a very wide box makes the McCormick
# envelope catastrophically loose, so OBBT needs ~5 sweeps to reach the fixpoint
# (the rounds=3 default stops short). Brute-force global optimum: x = 8 (all),
# sum sq = 6*64 = 384 <= 400, objective = -C(6,2)*64 = -960.
_N = 6
_UB = 2000
_SS_CAP = 400
_TRUE_OPT = -960.0


def _qcqp() -> Model:
    m = Model("iterate_root_qcqp")
    x = m.integer("x", _N, lb=0, ub=_UB)
    obj = 0
    for i in range(_N):
        for j in range(i + 1, _N):
            obj = obj - x[i] * x[j]
    m.minimize(obj)
    ss = 0
    for i in range(_N):
        ss = ss + x[i] * x[i]
    m.subject_to(ss <= _SS_CAP)
    return m


def _flat_bounds(model: Model):
    lbs, ubs = [], []
    for v in model._variables:
        lbs.append(v.lb.flatten())
        ubs.append(v.ub.flatten())
    return np.concatenate(lbs).astype(float), np.concatenate(ubs).astype(float)


def _mccormick_root_bound(model: Model, lb: np.ndarray, ub: np.ndarray):
    """McCormick LP dual bound over the box ``[lb, ub]`` (discopt machinery)."""
    relaxer = MccormickLPRelaxer(model, build_incremental=False)
    milp, _ = build_milp_relaxation(
        relaxer._model, relaxer._terms, relaxer._disc, bound_override=(lb, ub)
    )
    _lp = get_exact_dual_lp_solver() or get_exact_lp_solver()
    res = _lp(
        c=np.asarray(milp._c, dtype=float),
        A_ub=milp._A_ub,
        b_ub=milp._b_ub,
        bounds=list(milp._bounds),
        time_limit=5.0,
    )
    if res.status != SolveStatus.OPTIMAL:
        return None
    return float(res.objective) + float(milp._obj_offset)


_ONE_PASS = dict(rounds=1, min_improvement=None)
_DEFAULT = dict(rounds=3, min_improvement=None)
_ITERATE = dict(rounds=50, min_improvement=1e-3)


def _obbt_box(model, lb, ub, **kw):
    import time as _t

    r = obbt_tighten_root(model, lb.copy(), ub.copy(), deadline=_t.perf_counter() + 90.0, **kw)
    return r


@pytest.mark.correctness
def test_iterated_root_bound_much_tighter_than_one_pass_and_default():
    """Iterating to convergence dwarfs one-pass and beats the rounds=3 default."""
    m = _qcqp()
    lb0, ub0 = _flat_bounds(m)

    r_one = _obbt_box(m, lb0, ub0, **_ONE_PASS)
    b_one = _mccormick_root_bound(m, r_one.lb, r_one.ub)
    r_def = _obbt_box(m, lb0, ub0, **_DEFAULT)
    b_def = _mccormick_root_bound(m, r_def.lb, r_def.ub)
    r_it = _obbt_box(m, lb0, ub0, **_ITERATE)
    b_it = _mccormick_root_bound(m, r_it.lb, r_it.ub)

    assert b_one is not None and b_def is not None and b_it is not None
    # Iterating needs strictly more than 3 sweeps to reach the fixpoint here.
    assert r_it.n_rounds > 3
    # Iterated bound is a large multiple tighter than a single OBBT pass.
    assert b_it > b_one * 0.2  # e.g. -2.5e3 vs -9.4e5  => >100x tighter
    assert (b_one / b_it) > 20.0
    # And meaningfully tighter than the rounds=3 default (this is what the flag buys).
    assert b_it > b_def + 1e-6
    assert (b_def / b_it) > 1.3


@pytest.mark.correctness
def test_iterated_root_bound_is_sound():
    """Differential + feasible-point soundness of the iterated tightening."""
    m = _qcqp()
    lb0, ub0 = _flat_bounds(m)

    r_one = _obbt_box(m, lb0, ub0, **_ONE_PASS)
    b_one = _mccormick_root_bound(m, r_one.lb, r_one.ub)
    r_it = _obbt_box(m, lb0, ub0, **_ITERATE)
    b_it = _mccormick_root_bound(m, r_it.lb, r_it.ub)

    # Differential: the iterated bound is a valid dual (lower) bound — never looser
    # than one-pass, and never above the true box optimum (a bound that crossed the
    # optimum would be a false certificate).
    assert b_it >= b_one - 1e-6
    assert b_it <= _TRUE_OPT + 1e-4

    # The tightened box must be a subset of the input box (OBBT only shrinks).
    assert np.all(r_it.lb >= lb0 - 1e-9)
    assert np.all(r_it.ub <= ub0 + 1e-9)

    # Feasible-point sampling: no feasible integer point of the original model may
    # be cut by the tightened box. Enumerate the (tiny) feasible region sum sq<=400
    # with each coordinate in [0, 20] (20 = floor(sqrt(400))), plus the global
    # optimum, and assert every one lies inside the iterated box.
    lo, hi = r_it.lb, r_it.ub
    opt = np.full(_N, 8.0)
    assert np.all(opt >= lo - 1e-9) and np.all(opt <= hi + 1e-9)

    rng = np.random.default_rng(0)
    checked = 0
    for _ in range(4000):
        pt = rng.integers(0, 21, size=_N).astype(float)
        if float(np.sum(pt * pt)) > _SS_CAP:
            continue
        checked += 1
        assert np.all(pt >= lo - 1e-9), (pt, lo)
        assert np.all(pt <= hi + 1e-9), (pt, hi)
    assert checked > 50  # sampled a non-trivial number of genuine feasible points


@pytest.mark.correctness
def test_min_improvement_none_is_inert():
    """``min_improvement=None`` reproduces the legacy loop bit-for-bit (OFF path)."""
    m = _qcqp()
    lb0, ub0 = _flat_bounds(m)
    for rounds in (1, 3, 7):
        a = obbt_tighten_root(m, lb0.copy(), ub0.copy(), rounds=rounds)
        b = obbt_tighten_root(m, lb0.copy(), ub0.copy(), rounds=rounds, min_improvement=None)
        assert np.array_equal(a.lb, b.lb)
        assert np.array_equal(a.ub, b.ub)
        assert a.n_rounds == b.n_rounds
        assert a.n_tightened == b.n_tightened


def test_solver_flag_defaults_off():
    """The solve-path lever helper is OFF unless explicitly enabled."""
    from discopt.solver import _obbt_iterate_root_enabled

    saved = os.environ.pop("DISCOPT_OBBT_ITERATE", None)
    try:
        assert _obbt_iterate_root_enabled() is False
        for on in ("1", "true", "yes", "on", "ON"):
            os.environ["DISCOPT_OBBT_ITERATE"] = on
            assert _obbt_iterate_root_enabled() is True
        for off in ("0", "false", "", "no"):
            os.environ["DISCOPT_OBBT_ITERATE"] = off
            assert _obbt_iterate_root_enabled() is False
    finally:
        os.environ.pop("DISCOPT_OBBT_ITERATE", None)
        if saved is not None:
            os.environ["DISCOPT_OBBT_ITERATE"] = saved

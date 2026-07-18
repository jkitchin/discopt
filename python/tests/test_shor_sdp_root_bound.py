"""Regression tests for the strong-Shor SDP root bound (issue #661).

The bound-changing verification regime (CLAUDE.md §5) for a relaxation-strengthening
change: a *differential bound* test (new bound >= the old McCormick bound AND <= the
true box optimum on a fixed box) plus *feasible-point sampling* (no valid integer
point is cut). Both are exercised on small synthetic Koopmans-Beckmann QAPs whose
optimum is found by brute force — the class where McCormick collapses to ~0 and the
strong Shor SDP is (measured) essentially exact.

Soundness centerpiece: the surfaced value is the **safe dual bound** recomputed from
the conic solver's multipliers (`shor_sdp_safe_dual_bound`), which is a rigorous
global lower bound for *any* multipliers by weak duality plus an eigenvalue shift on
the dual slack matrix — locked here by evaluating it on adversarially perturbed
duals and checking it never crosses the true optimum.

The flag `SolverTuning.shor_sdp_root_bound` is verified default-off so the wired
path is a no-op unless opted in (`DISCOPT_SHOR_SDP_ROOT_BOUND`).

SCS (`discopt[sdp]`) is an optional dependency: solver-dependent tests skip without
it, while build/eligibility/safe-bound tests run regardless.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import (
    build_milp_relaxation,
    sanitize_relaxation_for_conditioning,
)
from discopt._jax.model_utils import binary_flat_cols, flat_variable_bounds
from discopt._jax.shor_sdp import (
    build_shor_sdp,
    shor_sdp_lower_bound,
    shor_sdp_safe_dual_bound,
)
from discopt._jax.term_classifier import classify_nonlinear_terms

_HAS_SCS = True
try:  # pragma: no cover - environment probe
    import scs  # noqa: F401
except ImportError:  # pragma: no cover
    _HAS_SCS = False

needs_scs = pytest.mark.skipif(not _HAS_SCS, reason="scs not installed (discopt[sdp])")


def _synthetic_qap(n: int, seed: int):
    """Small Koopmans-Beckmann QAP discopt model + its brute-force optimum
    (same construction as test_rlt_root_bound.py: indefinite objective, so the
    term-wise McCormick LP bound collapses to ~0)."""
    rng = np.random.default_rng(seed)
    F = rng.integers(0, 10, (n, n))
    F = F + F.T
    np.fill_diagonal(F, 0)
    D = rng.integers(0, 10, (n, n))
    D = D + D.T
    np.fill_diagonal(D, 0)

    m = dm.Model()
    x = m.binary("x", shape=(n * n,))

    def v(i, k):
        return x[i * n + k]

    for i in range(n):
        m.subject_to(dm.sum([v(i, k) for k in range(n)]) == 1)
    for k in range(n):
        m.subject_to(dm.sum([v(i, k) for i in range(n)]) == 1)
    terms = []
    Q = np.zeros((n * n, n * n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                for ll in range(n):
                    if i == j or k == ll:
                        continue
                    w = float(F[i, j] * D[k, ll])
                    if w:
                        terms.append(w * v(i, k) * v(j, ll))
                    Q[i * n + k, j * n + ll] += F[i, j] * D[k, ll]
    m.minimize(dm.sum(terms))

    best = np.inf
    for perm in itertools.permutations(range(n)):
        xv = np.zeros(n * n)
        for i, k in enumerate(perm):
            xv[i * n + k] = 1.0
        best = min(best, float(xv @ Q @ xv))
    return m, best


def _built(model):
    lb, ub = flat_variable_bounds(model)
    terms = classify_nonlinear_terms(model)
    relax, info = build_milp_relaxation(
        model, terms, DiscretizationState(), bound_override=(lb, ub)
    )
    relax = sanitize_relaxation_for_conditioning(relax)
    return relax, info


def _mccormick_lp_bound(relax):
    """Continuous McCormick LP root bound via the exact oracle (the loose baseline)."""
    from discopt._jax.obbt import get_exact_lp_solver

    _lp = get_exact_lp_solver()
    res = _lp(
        c=np.asarray(relax._c),
        A_ub=relax._A_ub,
        b_ub=np.asarray(relax._b_ub),
        bounds=list(relax._bounds),
        time_limit=30.0,
    )
    assert res.objective is not None
    return float(res.objective) + float(relax._obj_offset)


@needs_scs
@pytest.mark.parametrize("n,seed", [(4, 0), (5, 1)])
def test_shor_sdp_differential_bound_lifts_loose_mccormick(n, seed):
    """§5 differential bound: the strong-Shor safe bound >= the (near-zero)
    McCormick LP bound and stays <= the true optimum on the fixed root box."""
    model, opt = _synthetic_qap(n, seed)
    relax, info = _built(model)
    mcc = _mccormick_lp_bound(relax)
    bound, dim = shor_sdp_lower_bound(
        model, relax, info, binary_vars=binary_flat_cols(model), time_limit=60.0
    )
    assert dim == n * n + 1
    assert bound is not None
    # McCormick on an indefinite binary QP is trivially loose (~0).
    assert mcc <= 1e-3
    # New bound is a genuine tightening ...
    assert bound >= mcc - 1e-6
    assert bound > mcc + 1.0
    # ... and never crosses the true optimum (rigorous lower bound).
    assert bound <= opt + 1e-4
    # Measured (entry experiment): the strong Shor SDP is essentially *exact* on
    # small synthetic QAPs — require it to recover >= 99 % of the optimum.
    assert bound >= 0.99 * opt


@pytest.mark.parametrize("n,seed", [(4, 0), (4, 2)])
def test_shor_sdp_never_cuts_a_feasible_point(n, seed):
    """Feasible-point sampling: every genuine assignment lifts to a rank-1 PSD
    moment matrix that satisfies every equality and inequality row — no valid
    integer point is removed — and the objective there equals its true cost."""
    model, opt = _synthetic_qap(n, seed)
    relax, info = _built(model)
    prob = build_shor_sdp(model, relax, info, binary_vars=binary_flat_cols(model))
    assert prob is not None

    for perm in itertools.permutations(range(n)):
        x = np.zeros(n * n)
        for i, k in enumerate(perm):
            x[i * n + k] = 1.0
        u = prob.pack_point(x)
        assert np.max(np.abs(prob.A_eq @ u - prob.b_eq)) <= 1e-9
        assert np.max(prob.A_in @ u - prob.h) <= 1e-9
        # The lifted point is rank-1 [1;x][1;x]^T, hence PSD by construction; its
        # objective is the true cost of that assignment.
        cost = float(prob.c_svec @ u + prob.offset)
        assert cost >= opt - 1e-6


@needs_scs
def test_shor_sdp_bound_le_every_feasible_objective():
    """The surfaced bound underestimates the objective at *every* feasible
    assignment (weak duality end-to-end)."""
    model, opt = _synthetic_qap(4, 3)
    relax, info = _built(model)
    prob = build_shor_sdp(model, relax, info, binary_vars=binary_flat_cols(model))
    assert prob is not None
    bound, _ = shor_sdp_lower_bound(
        model, relax, info, binary_vars=binary_flat_cols(model), time_limit=60.0
    )
    assert bound is not None
    for perm in itertools.permutations(range(4)):
        x = np.zeros(16)
        for i, k in enumerate(perm):
            x[i * 4 + k] = 1.0
        u = prob.pack_point(x)
        obj = float(prob.c_svec @ u + prob.offset)
        assert bound <= obj + 1e-4


def test_shor_sdp_safe_bound_sound_for_arbitrary_duals():
    """Soundness centerpiece: `shor_sdp_safe_dual_bound` is a rigorous global
    lower bound for *any* multiplier vector — zero, random, or sign-violating
    (the clamp repairs inequality multipliers) — never crossing the true optimum.
    This is the property that makes surfacing a first-order (approximate) SDP
    solve sound: convergence affects tightness only."""
    model, opt = _synthetic_qap(4, 0)
    relax, info = _built(model)
    prob = build_shor_sdp(model, relax, info, binary_vars=binary_flat_cols(model))
    assert prob is not None
    n_dual = prob.b_eq.shape[0] + prob.h.shape[0]

    rng = np.random.default_rng(0)
    candidates = [
        np.zeros(n_dual),
        rng.normal(size=n_dual),
        1e3 * rng.normal(size=n_dual),
        -np.ones(n_dual),  # every inequality multiplier sign-violating
    ]
    for y in candidates:
        g = shor_sdp_safe_dual_bound(prob, y)
        assert g is not None
        assert g <= opt + 1e-6, f"safe bound {g} crosses optimum {opt}"


@needs_scs
def test_shor_sdp_surfaced_bound_is_the_safe_dual_value_not_the_scs_objective():
    """Mechanism lock: the surfaced bound IS the safe dual value recomputed from
    the returned multipliers (perturbing the duals moves it; it is not the raw
    first-order objective, which carries no certificate)."""
    import scipy.sparse as sp
    import scs as _scs

    model, opt = _synthetic_qap(4, 1)
    relax, info = _built(model)
    prob = build_shor_sdp(model, relax, info, binary_vars=binary_flat_cols(model))
    assert prob is not None

    m = prob.m
    A = sp.vstack([prob.A_eq, prob.A_in, -sp.identity(m, format="csr")], format="csc")
    b = np.concatenate([prob.b_eq, prob.h, np.zeros(m)])
    cone = {"z": int(prob.b_eq.shape[0]), "l": int(prob.h.shape[0]), "s": [prob.dim]}
    sol = _scs.SCS(
        {"A": A, "b": b, "c": prob.c_svec},
        cone,
        verbose=False,
        eps_abs=1e-5,
        eps_rel=1e-5,
        max_iters=100_000,
    ).solve()
    assert str(sol["info"]["status"]).lower().startswith("solved")
    safe = shor_sdp_safe_dual_bound(prob, np.asarray(sol["y"], dtype=np.float64))
    assert safe is not None

    bound, _ = shor_sdp_lower_bound(
        model, relax, info, binary_vars=binary_flat_cols(model), time_limit=60.0
    )
    assert bound is not None
    # Same mechanism (identical formulation + solver settings) -> same value up to
    # first-order solver reproducibility; and both are rigorous under-estimates.
    assert bound == pytest.approx(safe, rel=1e-6, abs=1e-6)
    assert bound <= opt + 1e-4
    assert safe <= opt + 1e-4


def test_shor_sdp_no_op_without_equality_constraints():
    """The lifted-equality RLT rows — the load-bearing part of the *strong* Shor
    bound (the plain Shor SDP is unbounded on this class) — need equalities; a
    purely inequality-constrained binary QP is declined (a sound no-op)."""
    m = dm.Model()
    x = m.binary("x", shape=(3,))
    m.subject_to(dm.sum([x[i] for i in range(3)]) <= 2)  # inequality only
    m.minimize(x[0] * x[1] + x[1] * x[2] - x[0] * x[2])
    relax, info = _built(m)
    assert build_shor_sdp(m, relax, info, binary_vars=binary_flat_cols(m)) is None
    bound, dim = shor_sdp_lower_bound(m, relax, info, binary_vars=binary_flat_cols(m))
    assert bound is None and dim == 0


def test_shor_sdp_no_op_on_continuous_qp():
    """Continuous variables break the binary moment diagonal ``X_ii = x_i``, the
    ``X_pp = x_p`` lifted-row substitution, and the trace cap — the build must
    decline any model with a non-binary variable."""
    m = dm.Model()
    x = m.continuous("x", shape=(3,), lb=0.0, ub=1.0)
    m.subject_to(dm.sum([x[i] for i in range(3)]) == 1)
    m.minimize(x[0] * x[1] + x[1] * x[2] - x[0] * x[2])
    relax, info = _built(m)
    assert build_shor_sdp(m, relax, info, binary_vars=binary_flat_cols(m)) is None


def test_shor_sdp_size_guard():
    """The moment-dimension guard declines an oversize model (a sound no-op)."""
    model, _ = _synthetic_qap(4, 0)
    relax, info = _built(model)
    # 16 vars -> dim 17; a guard below that must decline.
    assert (
        build_shor_sdp(model, relax, info, binary_vars=binary_flat_cols(model), max_dim=16) is None
    )
    bound, dim = shor_sdp_lower_bound(
        model, relax, info, binary_vars=binary_flat_cols(model), max_dim=16
    )
    assert bound is None and dim == 0


def test_shor_sdp_no_op_when_scs_missing(monkeypatch):
    """A missing conic solver is a sound no-op (`discopt[sdp]` is optional): the
    build succeeds but no bound is surfaced, and the eligible dimension is still
    reported."""
    import sys

    model, _ = _synthetic_qap(4, 0)
    relax, info = _built(model)
    monkeypatch.setitem(sys.modules, "scs", None)  # forces `import scs` to fail
    bound, dim = shor_sdp_lower_bound(model, relax, info, binary_vars=binary_flat_cols(model))
    assert bound is None and dim == 17


def test_shor_sdp_flag_default_off():
    """The wired lever is opt-in (§3): default-off, so the root path is unchanged
    unless DISCOPT_SHOR_SDP_ROOT_BOUND is set."""
    from discopt.solver_tuning import SolverTuning

    assert SolverTuning().shor_sdp_root_bound is False


@needs_scs
def test_shor_sdp_root_wiring_is_sound_when_enabled():
    """With the flag on, `_root_relaxation_lower_bound` returns a bound that never
    crosses the true optimum (the certificate invariant)."""
    from discopt.solver import _root_relaxation_lower_bound
    from discopt.solver_tuning import SolverTuning, reset_current, set_current

    model, opt = _synthetic_qap(4, 1)
    lb, ub = flat_variable_bounds(model)
    tok = set_current(SolverTuning(shor_sdp_root_bound=True))
    try:
        bound = _root_relaxation_lower_bound(model, lb, ub, time_limit=30.0)
    finally:
        reset_current(tok)
    assert bound is not None
    assert bound <= opt + 1e-4
    # The SDP candidate joins by `max`, so with McCormick ~0 the surfaced root
    # bound must reflect the SDP's tightening on this class.
    assert bound >= 0.9 * opt

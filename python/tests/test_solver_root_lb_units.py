"""Unit tests for root-bound seeding, strong branching, and GAMS warm seed (#87).

Direct calls with soundness checks: the root MILP relaxation value must
under-estimate the true optimum; strong branching must return one of its
candidates (branching order can never affect soundness); a GAMS-parsed
model's `.l` levels seed the solve without changing the certificate.
"""

from __future__ import annotations

import os
import textwrap

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.modeling.core import Model
from discopt.solver import _root_relaxation_lower_bound, _strong_branch_lp

pytestmark = pytest.mark.unit


def test_root_relaxation_lower_bound_is_sound():
    m = Model("rootlb")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.subject_to(x * y >= 1.0)
    m.minimize(x + y)
    lb = np.array([0.0, 0.0])
    ub = np.array([4.0, 4.0])
    bound = _root_relaxation_lower_bound(m, lb, ub, time_limit=20.0)
    assert bound is not None
    # True optimum is 2.0 at x=y=1: the relaxation value must never exceed it.
    assert bound <= 2.0 + 1e-8
    # And on this box the McCormick envelope gives a nontrivial bound.
    assert bound > -1e6


def test_root_relaxation_lower_bound_with_psd_cuts():
    m = Model("rootpsd")
    x = m.continuous("x", lb=-2.0, ub=2.0)
    y = m.continuous("y", lb=-2.0, ub=2.0)
    m.subject_to(x + y >= 0.5)
    m.minimize(x * x + y * y - x * y)
    bound = _root_relaxation_lower_bound(
        m, np.array([-2.0, -2.0]), np.array([2.0, 2.0]), time_limit=20.0, psd_cuts=True
    )
    if bound is not None:  # psd strengthening may abstain; that is sound
        # True optimum: symmetric x=y=0.25 -> 0.0625.
        assert bound <= 0.0625 + 1e-8


def test_strong_branch_lp_returns_candidate():
    m = Model("sb")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    m.subject_to(x * y >= 1.0)
    m.minimize(x + y)
    ev = NLPEvaluator(m)
    sol = np.array([0.5, 2.0])
    cands = np.array([0, 1])
    picked = _strong_branch_lp(
        ev,
        sol,
        np.array([0.0, 0.0]),
        np.array([4.0, 4.0]),
        cands,
        parent_lb=0.0,
        time_limit=2.0,
    )
    # Branching-order metadata only: any candidate (or an abstention) is
    # valid, but a returned index must come from the candidate list.
    assert picked is None or picked in (0, 1)


def test_gams_parsed_model_with_levels_solves():
    from discopt.modeling.gams_parser import parse_gams

    gms = textwrap.dedent("""\
        Free Variables x1, x2, obj ;
        x1.lo = -2 ; x1.up = 2 ;
        x2.lo = -2 ; x2.up = 2 ;
        x1.l = -1.0 ;
        x2.l = 1.0 ;
        Equations rosenbrock ;
        rosenbrock.. obj =e= sqr(1 - x1) + 100 * sqr(x2 - sqr(x1)) ;
        Model rosen / all / ;
        Solve rosen using NLP minimizing obj ;
    """)
    m = parse_gams(gms)
    # Certifying the quartic globally takes minutes; a short budget still
    # exercises the .l warm-seed path and must return the (locally found)
    # optimum as the incumbent without any false certificate.
    res = m.solve(time_limit=5.0)
    assert res.status in ("optimal", "feasible", "time_limit")
    # Rosenbrock optimum 0 at (1, 1); the .l seed must not change that.
    assert res.objective is not None
    assert res.objective == pytest.approx(0.0, abs=1e-4)


def test_time_limited_nonconvex_solve_never_lies():
    # A short budget on an iterating instance: whatever status comes back,
    # the certificate rules hold (bound <= objective for min sense; a
    # claimed optimal matches the true optimum 2 + 2*sqrt(2)).
    m = Model("shortbudget")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.continuous("y", lb=0.0, ub=4.0)
    u = m.continuous("u", lb=0.0, ub=4.0)
    v = m.continuous("v", lb=0.0, ub=4.0)
    b = m.binary("b", shape=(4,))
    m.subject_to(x * y >= 1.0)
    m.subject_to(u * v >= 2.0)
    m.subject_to(x + u <= 3.0 + b[0] + 2.0 * b[1])
    m.subject_to(y + v <= 3.0 + b[2] + 2.0 * b[3])
    m.minimize(x + y + u + v + b[0] + b[1] + b[2] + b[3])
    res = m.solve(time_limit=0.6)
    true_opt = 2.0 + 2.0 * np.sqrt(2.0)
    if res.objective is not None:
        assert res.objective >= true_opt - 1e-6  # incumbents are feasible
    if res.bound is not None and res.objective is not None:
        assert res.bound <= res.objective + 1e-6
    if res.status == "optimal":
        assert res.objective == pytest.approx(true_opt, abs=1e-3)

"""End-to-end tests for the ``psd_cuts`` solve flag (Wave 2, W2c).

PSD (moment) cuts strengthen the McCormick relaxation toward the SDP bound on
nonconvex QCQP. Two properties are pinned here:

* **soundness** — ``Model.solve(psd_cuts=True)`` returns the same global optimum
  as the baseline (cuts only tighten the relaxation; they never remove a point);
* **strength** — at the relaxation level the PSD loop closes the plain-McCormick
  root gap to the true optimum on an indefinite QCQP.

Note: on small box-QPs discopt's existing per-node separation already reaches the
optimum, so PSD cuts *match* (do not beat) the baseline node count there; their
distinct value is closing the gap with a cheap root LP-cut loop.
"""

from __future__ import annotations

import itertools
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling.core as dm
import numpy as np

# Symmetric, indefinite Q on the unit box; optimum is at a vertex.
_Q = np.array([[1.0, -3.0, 0.0], [-3.0, 1.0, -2.0], [0.0, -2.0, 1.0]])


def _build():
    m = dm.Model("indef3")
    x = m.continuous("x", shape=(3,), lb=0, ub=1)
    expr = None
    for i in range(3):
        for j in range(3):
            if _Q[i, j] != 0.0:
                term = _Q[i, j] * x[i] * x[j]
                expr = term if expr is None else expr + term
    m.minimize(expr)
    return m


def _vertex_optimum() -> float:
    best = np.inf
    for v in itertools.product((0.0, 1.0), repeat=3):
        v = np.array(v)
        best = min(best, float(v @ _Q @ v))
    return best


def test_psd_cuts_flag_preserves_global_optimum():
    opt = _vertex_optimum()
    res = _build().solve(psd_cuts=True, time_limit=30)
    assert res.status == "optimal"
    assert abs(float(res.objective) - opt) < 1e-4


def test_psd_cuts_match_baseline_optimum():
    base = _build().solve(psd_cuts=False, time_limit=30)
    psd = _build().solve(psd_cuts=True, time_limit=30)
    assert abs(float(base.objective) - float(psd.objective)) < 1e-4


def test_psd_closes_plain_mccormick_root_gap():
    """At the relaxation level, pairwise PSD cuts close the plain-McCormick gap.

    Uses ``min x0^2 + x1^2 - 3 x0 x1`` on ``[0,1]^2`` (optimum -1, plain McCormick
    bound -1.5) — a single 2-variable moment cut closes it exactly. (Pairwise PSD
    cuts capture 2-variable minors; genuinely 3-way moment coupling would need
    dense k>=3 cuts, a documented follow-on.)
    """
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.psd_cuts import psd_strengthen_relaxation_bound
    from discopt._jax.term_classifier import classify_nonlinear_terms

    m = dm.Model("indef2")
    x = m.continuous("x", shape=(2,), lb=0, ub=1)
    m.minimize(x[0] * x[0] + x[1] * x[1] - 3 * x[0] * x[1])
    lb = np.zeros(2)
    ub = np.ones(2)
    terms = classify_nonlinear_terms(m)
    relax, info = build_milp_relaxation(m, terms, DiscretizationState(), bound_override=(lb, ub))
    z_before, z_after, n_cuts = psd_strengthen_relaxation_bound(relax, info, max_rounds=12)

    opt = -1.0
    assert n_cuts >= 1
    assert z_before < opt - 1e-3  # plain McCormick is loose (-1.5)
    assert z_after > z_before + 1e-4  # PSD tightens it
    assert z_after <= opt + 1e-6  # and stays a valid bound
    # Gap (largely) closed: PSD reaches >= 90% of the way to the optimum.
    assert z_after >= z_before + 0.9 * (opt - z_before)

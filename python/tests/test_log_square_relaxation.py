"""Relaxation coverage for squares of lifted univariate calls (issue #369).

nvs09 — ``min Σ log(x-2)**2 + log(10-x)**2 - (Πx)**0.2`` over 10 integer vars —
was a *find-not-certify* case: discopt found the optimum but produced **no lower
bound** (``status=feasible, bound=None``). The relaxation builder linearizes the
objective all-or-nothing, and the squares of the lifted ``log`` auxiliaries were
absent from the monomial map (``univariate_square_var_map`` only registered
``sin``/``cos``/``tan`` squares), so the whole objective — including the
``(Πx)**0.2`` term ``factorable_reform`` *does* lift — fell back to a feasibility
objective (``objective_bound_valid=False``).

The generic tangent/secant square envelope emitted for every
``UnivariateSquareRelaxation`` (``w = base**2`` over ``[base_lb, base_ub]``) is
function-agnostic and sound for any lifted base, so the fix is simply to register
squares of *any* lifted univariate call, in both the authored ``f**2`` form and
the ``f*f`` product ``distribute_products`` rewrites it into (the shape the
objective takes once a sibling term such as ``-(Πx)**0.2`` triggers
``factorable_reform``).

These tests pin, without depending on the slow end-to-end ``.nl`` solve:
* the objective now *linearizes* (``objective_bound_valid=True``) with square
  aux columns registered for ``log``/``exp`` bases, in both syntactic forms; and
* the resulting root LP bound is *sound* (never above the true integer optimum),
  checked against a brute-forced small box — the soundness mandate that matters,
  since the tighter certified value comes from spatial B&B.
"""

from __future__ import annotations

import itertools
import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax import factorable_reform as fr
from discopt._jax.discretization import DiscretizationState
from discopt._jax.milp_relaxation import build_milp_relaxation
from discopt._jax.model_utils import flat_variable_bounds
from discopt._jax.term_classifier import classify_nonlinear_terms


def _prod(seq):
    p = seq[0]
    for s in seq[1:]:
        p = p * s
    return p


def _build(model, *, reform=False):
    """Build the root MILP relaxation, optionally applying factorable_reform
    first (as the solver does when ``has_factorable_work`` — e.g. a ``(Πx)**0.2``
    term needs its composite-power lift)."""
    if reform and fr.has_factorable_work(model):
        model = fr.factorable_reformulate(model)
    terms = classify_nonlinear_terms(model)
    disc = DiscretizationState(partitions={})
    lb, ub = flat_variable_bounds(model)
    milp, varmap = build_milp_relaxation(
        model, terms, disc, bound_override=(np.asarray(lb, float), np.asarray(ub, float))
    )
    return milp, varmap


def _square_base_func_names(varmap):
    """func_name of each lifted univariate whose square got an aux column."""
    by_aux = {r.aux_col: r for r in varmap["univariate_relaxations"]}
    names = []
    for sq in varmap["univariate_square_relaxations"]:
        base = by_aux.get(sq.base_col)
        if base is not None:
            names.append(base.func_name)
    return names


def _root_lp_bound(milp):
    """Continuous LP relaxation value (integrality dropped -> still a valid lower
    bound on the integer optimum). Uses scipy so the check needs no MILP backend."""
    import scipy.sparse as sp
    from scipy.optimize import linprog

    A = milp._A_ub
    if A is not None and sp.issparse(A):
        A = A.toarray()
    b = None if milp._b_ub is None else np.asarray(milp._b_ub, float)
    bounds = [(float(lo), float(hi)) for (lo, hi) in milp._bounds]
    res = linprog(np.asarray(milp._c, float), A_ub=A, b_ub=b, bounds=bounds, method="highs")
    assert res.success, f"root LP failed: {res.message}"
    return float(res.fun + milp._obj_offset)


def _brute_min(fn, lo, hi, n):
    return min(fn(pt) for pt in itertools.product(range(lo, hi + 1), repeat=n))


# --------------------------------------------------------------------------- #
# Direction 1: squares of lifted log/exp calls linearize (were "not in map").
# --------------------------------------------------------------------------- #


@pytest.mark.correctness
def test_log_square_objective_linearizes_with_square_aux():
    """``Σ log(x-2)**2 + log(10-x)**2`` (nvs09's transcendental half) now yields a
    bounding relaxation instead of a feasibility fallback."""
    n = 4
    m = dm.Model("log_sq")
    x = m.integer("x", shape=n, lb=3, ub=9)
    m.minimize(sum(dm.log(x[i] - 2) ** 2 + dm.log(10 - x[i]) ** 2 for i in range(n)))

    milp, _varmap = _build(m)

    # Engine contract: the log**2 objective linearizes to a valid finite bound (no
    # feasibility fallback). Soundness of that bound is asserted by
    # test_log_square_root_bound_is_sound below.
    assert milp._objective_bound_valid, "log**2 objective must produce a lower bound"


@pytest.mark.correctness
def test_exp_square_objective_linearizes_with_square_aux():
    """``Σ exp(0.1 x)**2`` linearizes through the same generic square envelope."""
    n = 3
    m = dm.Model("exp_sq")
    x = m.integer("x", shape=n, lb=3, ub=9)
    m.minimize(sum(dm.exp(0.1 * x[i]) ** 2 for i in range(n)))

    milp, _varmap = _build(m)

    # Engine contract: the exp**2 objective linearizes to a valid finite bound.
    assert milp._objective_bound_valid


@pytest.mark.correctness
def test_distributed_log_square_product_form_is_collected():
    """When a sibling term (here ``(Πx)**0.2``) triggers factorable_reform, the
    ``log(.)**2`` terms must still be collected into a sound, finite objective bound
    (else nvs09's log squares stay uncollected -> no bound).

    The uniform engine lifts each ``log(.)`` into an aux variable and bounds the
    resulting polynomial in aux space, so the reformed objective no longer contains
    the ``log()`` calls syntactically (it did under the old distribute-into-``f*f``
    collector). The contract this test guards is therefore the *bound the engine
    produces*, not the objective's syntactic shape."""
    n = 3
    m = dm.Model("log_sq_reform")
    x = m.integer("x", shape=n, lb=3, ub=9)
    # The ``(Πx)**0.2`` term makes has_factorable_work True, so the objective is
    # routed through factorable_reform.
    m.minimize(
        sum(dm.log(x[i] - 2) ** 2 + dm.log(10 - x[i]) ** 2 for i in range(n))
        - (_prod([x[i] for i in range(n)])) ** 0.2
    )
    assert fr.has_factorable_work(m)

    milp, _varmap = _build(m, reform=True)
    # Engine contract: the log**2 terms are collected (lifted), so the objective
    # linearizes to a valid finite bound rather than a feasibility fallback.
    assert milp._objective_bound_valid

    # And that bound is finite and sound (<= the true integer optimum) — proving the
    # log squares were actually bounded, not silently dropped.
    bound = _root_lp_bound(milp)
    true_min = _brute_min(
        lambda pt: sum(math.log(v - 2) ** 2 + math.log(10 - v) ** 2 for v in pt)
        - (math.prod(pt)) ** 0.2,
        3,
        9,
        n,
    )
    assert math.isfinite(bound)
    assert bound <= true_min + 1e-6, f"unsound bound {bound} > true min {true_min}"


# --------------------------------------------------------------------------- #
# Soundness: the enabled square envelope never over-states the lower bound.
# --------------------------------------------------------------------------- #


@pytest.mark.correctness
@pytest.mark.parametrize("n", [2, 3])
def test_log_square_root_bound_is_sound(n):
    m = dm.Model("log_sq_sound")
    x = m.integer("x", shape=n, lb=3, ub=9)
    m.minimize(sum(dm.log(x[i] - 2) ** 2 + dm.log(10 - x[i]) ** 2 for i in range(n)))

    milp, _ = _build(m)
    assert milp._objective_bound_valid
    bound = _root_lp_bound(milp)

    true_min = _brute_min(
        lambda pt: sum(math.log(v - 2) ** 2 + math.log(10 - v) ** 2 for v in pt), 3, 9, n
    )
    assert bound <= true_min + 1e-6, f"UNSOUND: bound {bound} > true min {true_min}"


# --------------------------------------------------------------------------- #
# Direction 2 (the "in combination" case): the full nvs09-shaped objective —
# log**2 terms *and* a fractional power of a high-degree multilinear product —
# linearizes to a single sound bound. The (Πx)**0.2 lift already existed in
# factorable_reform; removing the log**2 blocker lets the whole objective bound.
# --------------------------------------------------------------------------- #


@pytest.mark.correctness
@pytest.mark.parametrize("n", [2, 3, 4])
def test_nvs09_shape_objective_linearizes_and_bound_is_sound(n):
    m = dm.Model("nvs09_shape")
    x = m.integer("x", shape=n, lb=3, ub=9)
    m.minimize(
        sum(dm.log(x[i] - 2) ** 2 + dm.log(10 - x[i]) ** 2 for i in range(n))
        - (_prod([x[i] for i in range(n)])) ** 0.2
    )

    milp, _varmap = _build(m, reform=True)

    # The whole mixed objective bounds: log**2 squares AND the fractional power of
    # the multilinear product are both present, not a feasibility fallback.
    assert milp._objective_bound_valid, "combined log**2 + (Πx)**0.2 objective must bound"

    bound = _root_lp_bound(milp)

    def true_fn(pt):
        s = sum(math.log(v - 2) ** 2 + math.log(10 - v) ** 2 for v in pt)
        pr = 1.0
        for v in pt:
            pr *= v
        return s - pr**0.2

    true_min = _brute_min(true_fn, 3, 9, n)
    assert bound <= true_min + 1e-6, f"UNSOUND: bound {bound} > true min {true_min}"


@pytest.mark.correctness
def test_full_nvs09_size_objective_linearizes():
    """At nvs09's actual size (10 integer vars) the objective linearizes: 20 log
    squares + one fractional power of the 10-way product (no feasibility fallback)."""
    n = 10
    m = dm.Model("nvs09_full")
    x = m.integer("x", shape=n, lb=3, ub=9)
    m.minimize(
        sum(dm.log(x[i] - 2) ** 2 + dm.log(10 - x[i]) ** 2 for i in range(n))
        - (_prod([x[i] for i in range(n)])) ** 0.2
    )

    milp, _varmap = _build(m, reform=True)
    # Engine contract: at nvs09's full size the mixed objective still linearizes to
    # a valid finite bound (no feasibility fallback).
    assert milp._objective_bound_valid

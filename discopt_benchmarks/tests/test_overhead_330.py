"""Regression tests for the per-solve overhead fixes (issue #330).

The headline fix routes ``from_nl`` / repr-only LP/QP/MIQP models through the
fast numeric Rust-repr extractor instead of the per-primitive eager-JAX autodiff
fallback. ``alan`` (a MIQP whose algebraic DAG walk fails) was the flagship: it
recompiled ~100 XLA programs during QP data extraction. These tests pin both the
*correctness* of the fast path (it must match the autodiff result bit-for-bit on
the affine/quadratic coefficients) and the *overhead* win (the extra compiles
are gone).
"""

from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm  # noqa: E402
from discopt._jax import problem_classifier as pc  # noqa: E402
from discopt_benchmarks.perf.measure import count_xla_compiles  # noqa: E402
from discopt_benchmarks.perf.panel import DATA_DIR  # noqa: E402

_ALAN = os.path.join(DATA_DIR, "alan.nl")


def _require(path: str) -> None:
    if not os.path.exists(path):
        pytest.skip(f"{os.path.basename(path)} not vendored")


@pytest.mark.regression
def test_repr_qp_extraction_matches_autodiff():
    """The fast repr extractor and the autodiff fallback must agree on Q, c, A_eq
    for a ``from_nl`` MIQP (``_builder is None``) — the fast path is a pure
    speedup, not a behaviour change."""
    _require(_ALAN)
    model = dm.from_nl(_ALAN)
    assert getattr(model, "_builder", None) is None  # from_nl: no fast-API builder
    # The algebraic walk genuinely fails here, which is *why* the slow autodiff
    # fallback used to run; guard that assumption so the test stays meaningful.
    with pytest.raises((pc._NotQuadraticError, pc._NotLinearError)):
        pc.extract_qp_data_algebraic(model)

    fast = pc._extract_qp_data_from_repr(model)
    slow = pc._extract_qp_data_autodiff(model)
    assert np.allclose(np.asarray(fast.Q), np.asarray(slow.Q), atol=1e-9)
    assert np.allclose(np.asarray(fast.c), np.asarray(slow.c), atol=1e-9)
    assert np.asarray(fast.A_eq).shape == np.asarray(slow.A_eq).shape
    assert np.allclose(np.asarray(fast.A_eq), np.asarray(slow.A_eq), atol=1e-9)


@pytest.mark.regression
def test_extract_qp_data_prefers_repr_for_from_nl():
    """``extract_qp_data`` (the dispatcher) must return the fast repr result for a
    ``from_nl`` MIQP, i.e. not fall through to the autodiff path. Verified by
    value-equality with the repr extractor."""
    _require(_ALAN)
    model = dm.from_nl(_ALAN)
    got = pc.extract_qp_data(model)
    ref = pc._extract_qp_data_from_repr(model)
    assert np.allclose(np.asarray(got.Q), np.asarray(ref.Q), atol=1e-12)
    assert np.allclose(np.asarray(got.c), np.asarray(ref.c), atol=1e-12)


@pytest.mark.regression
def test_alan_solve_does_not_recompile_per_node():
    """Solving ``alan`` must not trigger the autodiff-extraction compile storm.

    Pre-fix this recompiled ~100 XLA programs (QP/LP data extraction dispatched
    hundreds of eager primitives, each lowered separately). The bound is loose
    (well under that, comfortably above the handful of legitimate evaluator
    compiles) so the test keys on the regime change, not an exact count.
    """
    _require(_ALAN)
    model = dm.from_nl(_ALAN)
    with count_xla_compiles() as compiles:
        r = model.solve(time_limit=60, gap_tolerance=1e-4)
    assert abs(float(r.objective) - 2.925) < 1e-3  # MINLPLib optimum
    assert compiles.count < 30, f"expected the autodiff compile storm gone, saw {compiles.count}"


# ── SCIP-style heuristic effort budget (the Python-orchestration half) ──────
_EX1226 = os.path.join(DATA_DIR, "ex1226.nl")


def _solve_counting_nlp(path: str, budget_on: bool):
    """Solve ``path`` and return (objective, n_nlp_solves) with the heuristic
    budget toggled via the env switch the solver reads."""
    import discopt.solvers.nlp_pounce as npn

    calls = {"n": 0}
    orig = npn.solve_nlp

    def _counted(*a, **k):
        calls["n"] += 1
        return orig(*a, **k)

    prev = os.environ.get("DISCOPT_HEUR_BUDGET")
    os.environ["DISCOPT_HEUR_BUDGET"] = "1" if budget_on else "0"
    npn.solve_nlp = _counted
    try:
        r = dm.from_nl(path).solve(time_limit=60, gap_tolerance=1e-4)
    finally:
        npn.solve_nlp = orig
        if prev is None:
            os.environ.pop("DISCOPT_HEUR_BUDGET", None)
        else:
            os.environ["DISCOPT_HEUR_BUDGET"] = prev
    return (None if r.objective is None else float(r.objective)), calls["n"]


@pytest.mark.regression
def test_heuristic_budget_cuts_solves_without_changing_optimum():
    """The success-weighted node-proportional budget must defer the heavy
    standalone-strength improvers on an easy nonconvex MINLP — fewer sub-NLP
    solves — while returning the *same* certified optimum (it changes speed, not
    the answer). ex1226 (3 nodes, optimum -17) fires the enumeration phase + LNS
    unconditionally without the budget."""
    _require(_EX1226)
    obj_off, n_off = _solve_counting_nlp(_EX1226, budget_on=False)
    obj_on, n_on = _solve_counting_nlp(_EX1226, budget_on=True)
    # Same optimum either way — soundness/correctness is never traded.
    assert obj_off is not None and abs(obj_off - (-17.0)) < 1e-2
    assert obj_on is not None and abs(obj_on - (-17.0)) < 1e-2
    # The budget strictly reduces the heuristic sub-NLP volume on this instance.
    assert n_on < n_off, f"budget did not cut sub-NLP solves: on={n_on} off={n_off}"

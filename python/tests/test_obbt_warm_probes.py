"""cert:T2.2 — differential neutrality of the persistent per-sweep OBBT LP.

``run_obbt_on_relaxation`` assembles the standard-form CSC once per sweep and
warm-starts each probe from the previous probe's optimal basis (T2.2 (a)+(b)).
Because consecutive probes differ only in the objective over a (weakly)
shrinking box, the warm basis is usually still primal-feasible and the Rust
``solve_lp_cols_warm`` path finishes in a primal phase-2 — and falls to the
trusted cold two-phase solve whenever the basis is unusable. The optimum is the
exact vertex either way, so this is a *bound-neutral* speedup: the applied
tightenings must be **bit-identical** to the pre-T2.2 cold-seam path.

These tests pin that invariant. They compare the persistent+warm path (the
default when the Rust simplex binding is importable) against the seam-based cold
path (forced by disabling ``_simplex_available``), which reproduces the old
behavior. A regression that made warming change any tightening — the only way
this optimization could be unsound — fails here.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt._jax.obbt as obbt_mod
import numpy as np
import pytest
from discopt._jax.obbt import _PersistentProbeLP, run_obbt_on_relaxation
from discopt.modeling.core import Model

pytestmark = pytest.mark.filterwarnings("ignore::RuntimeWarning")


def _build_relaxation(model: Model):
    from discopt._jax.discretization import initialize_partitions
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms

    terms = classify_nonlinear_terms(model)
    state = initialize_partitions([], lb=[], ub=[], n_init=2)
    milp, _varmap = build_milp_relaxation(model, terms, state, incumbent=None)
    return milp


def _bilinear() -> Model:
    m = Model()
    x = m.continuous("x", lb=-2.0, ub=3.0)
    y = m.continuous("y", lb=-1.0, ub=4.0)
    z = m.continuous("z", lb=-10.0, ub=10.0)
    m.subject_to(x * y <= 5.0)
    m.subject_to(x + y + z <= 6.0)
    m.subject_to(z >= x * y - 2.0)
    m.minimize(z)
    return m


def _ratio() -> Model:
    m = Model()
    x = m.continuous("x", lb=1.0, ub=10.0)
    y = m.continuous("y", lb=1.0, ub=10.0)
    m.subject_to(x / y <= 5.0)
    m.subject_to(x + y <= 15.0)
    m.minimize(x - y)
    return m


def _mixed() -> Model:
    m = Model()
    x = m.continuous("x", lb=-3.0, ub=3.0)
    y = m.integer("y", lb=-3, ub=3)
    z = m.continuous("z", lb=-20.0, ub=20.0)
    m.subject_to(x * y <= 4.0)
    m.subject_to(z >= x * x - y)
    m.subject_to(x + y + z <= 10.0)
    m.minimize(z)
    return m


def _wide() -> Model:
    m = Model()
    xs = [m.continuous(f"x{i}", lb=-2.0, ub=2.0) for i in range(4)]
    m.subject_to(xs[0] * xs[1] + xs[2] * xs[3] <= 3.0)
    m.subject_to(sum(xs) <= 4.0)
    m.subject_to(xs[0] * xs[2] >= -3.0)
    m.minimize(xs[0] + xs[3])
    return m


PANEL = {"bilinear": _bilinear, "ratio": _ratio, "mixed": _mixed, "wide": _wide}


def _maxdiff(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # Matching non-finite entries (an untightened +/-inf bound in both paths) are
    # equal; only finite entries carry a numeric difference.
    both_inf = ~np.isfinite(a) & ~np.isfinite(b) & (np.sign(a) == np.sign(b))
    d = np.abs(a - b)
    d[both_inf] = 0.0
    return float(np.max(d)) if d.size else 0.0


def _cold(model: Model, n_orig: int, cutoff):
    """Force the seam-based cold path (pre-T2.2 behavior)."""
    orig = obbt_mod._simplex_available
    obbt_mod._simplex_available = lambda: False
    try:
        rel = _build_relaxation(model)
        return run_obbt_on_relaxation(
            rel, n_orig=n_orig, time_limit_per_lp=5.0, incumbent_cutoff=cutoff
        )
    finally:
        obbt_mod._simplex_available = orig


def _warm(model: Model, n_orig: int, cutoff):
    """The default persistent + warm-start path."""
    assert obbt_mod._simplex_available(), "Rust simplex binding required for this test"
    rel = _build_relaxation(model)
    return run_obbt_on_relaxation(
        rel, n_orig=n_orig, time_limit_per_lp=5.0, incumbent_cutoff=cutoff
    )


@pytest.mark.parametrize("name", list(PANEL))
@pytest.mark.parametrize("cutoff", [None, 100.0, 5.0])
def test_warm_probes_bound_neutral(name, cutoff):
    """Warm bound == cold bound and applied tightenings are identical (<=1e-9)."""
    model = PANEL[name]()
    n_orig = sum(v.size for v in model._variables)
    warm = _warm(model, n_orig, cutoff)
    cold = _cold(model, n_orig, cutoff)

    assert warm.n_lp_solves == cold.n_lp_solves
    assert warm.n_tightened == cold.n_tightened
    assert _maxdiff(warm.tightened_lb, cold.tightened_lb) <= 1e-9
    assert _maxdiff(warm.tightened_ub, cold.tightened_ub) <= 1e-9


def test_warm_probes_never_loosen_bounds():
    """The persistent path never returns a bound outside the original box."""
    model = _bilinear()
    n_orig = sum(v.size for v in model._variables)
    rel = _build_relaxation(model)
    orig_bounds = np.asarray(rel._bounds[:n_orig], dtype=np.float64)
    res = run_obbt_on_relaxation(rel, n_orig=n_orig, time_limit_per_lp=5.0)
    # tightening only: lb non-decreasing, ub non-increasing.
    assert np.all(res.tightened_lb >= orig_bounds[:, 0] - 1e-9)
    assert np.all(res.tightened_ub <= orig_bounds[:, 1] + 1e-9)


def test_persistent_lp_matches_fresh_solve():
    """A persistent-LP probe reproduces a fresh cold solve of the same LP."""
    from discopt.solvers import SolveStatus

    model = _bilinear()
    rel = _build_relaxation(model)
    A_ub, b_ub = rel._A_ub, rel._b_ub
    bounds = list(rel._bounds)
    n_total = len(bounds)
    lb = np.array([b[0] for b in bounds], dtype=np.float64)
    ub = np.array([b[1] for b in bounds], dtype=np.float64)

    probe = _PersistentProbeLP(A_ub, b_ub, n_total)
    warm_basis = None
    for j in range(n_total):
        c = np.zeros(n_total)
        c[j] = 1.0
        st, obj, _duals, basis, _wall = probe.solve(c, lb, ub, warm_basis)
        if st != SolveStatus.OPTIMAL:
            warm_basis = None
            continue
        warm_basis = basis
        # Independent cold solve of the identical LP via the seam.
        from discopt.solvers.lp_simplex import solve_lp as cold_solve

        cold = cold_solve(
            c=c,
            A_ub=A_ub,
            b_ub=b_ub,
            bounds=[(float(lb[i]), float(ub[i])) for i in range(n_total)],
        )
        assert cold.status == SolveStatus.OPTIMAL
        assert obj == pytest.approx(float(cold.objective), abs=1e-9, rel=1e-9)

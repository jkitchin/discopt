"""#2011: the native spatial kernel takes a model with a free column that enters only
linearly (``DISCOPT_KERNEL_FREE_LINEAR``).

The producer declined any spec with an infinite column bound, because an unbounded
FACTOR makes a McCormick envelope degenerate (#764). A free column that is no lifted
term's factor has no envelope, yet one such column sent util and sporttournament14
to the Python tree. These tests pin:

* the spec: built under the flag with the ``1e20`` sentinel (never ``inf``) on the
  free column, which is no OBBT candidate; declined with the flag off, and declined
  under the flag when the free column is a factor of a lifted term;
* the solve: the kernel takes the model and certifies the Python tree's objective;
* soundness: over seeded random boxes the kernel's certified bound never exceeds the
  closed-form optimum.
"""

from __future__ import annotations

import itertools

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.spatial_producer import build_spatial_kernel_spec


def _model(xlo=0.0, xhi=3.0, ylo=0.0, yhi=3.0, free_factor=False):
    """min x*y + w  s.t.  z1 - z2 == x - y,  w >= z1 - z2,  z1 + z2 >= 1.

    ``z1``/``z2`` are free and enter only linearly; no row bounds either one (each
    row has two unbounded terms), so they stay free through presolve, as util's
    and sporttournament14's do. ``w`` in [-10, 10] makes them matter:
    ``w >= x - y``, so on boxes inside [-3, 3]^2 the optimum is
    ``min_box x*y + x - y``, a bilinear form whose minimum is at a box vertex."""
    m = dm.Model("free_linear")
    x = m.continuous("x", lb=xlo, ub=xhi)
    y = m.continuous("y", lb=ylo, ub=yhi)
    w = m.continuous("w", lb=-10.0, ub=10.0)
    # Free as a ``.nl`` file declares it: infinite bounds. (A column declared with
    # no bounds gets the finite default box +-9.999e19 and never reached the guard.)
    z1 = m.continuous("z1", lb=-np.inf, ub=np.inf)
    z2 = m.continuous("z2", lb=-np.inf, ub=np.inf)
    m.subject_to(z1 - z2 == x - y)
    m.subject_to(w >= z1 - z2)
    m.subject_to(z1 + z2 >= 1)
    if free_factor:
        m.subject_to(x * z1 <= 5)
    m.minimize(x * y + w)
    return m


def _oracle(xlo, xhi, ylo, yhi):
    return min(a * b + a - b for a, b in itertools.product((xlo, xhi), (ylo, yhi)))


@pytest.mark.smoke
def test_spec_admits_a_free_linear_column_with_the_sentinel(monkeypatch):
    monkeypatch.setenv("DISCOPT_KERNEL_FREE_LINEAR", "0")
    assert build_spatial_kernel_spec(_model()) is None, "flag off must keep declining"
    monkeypatch.setenv("DISCOPT_KERNEL_FREE_LINEAR", "1")
    spec = build_spatial_kernel_spec(_model())
    assert spec is not None
    lo, hi = spec["global_lo"], spec["global_hi"]
    assert np.isfinite(lo).all() and np.isfinite(hi).all(), "inf reached the kernel"
    assert (lo[3], hi[3]) == (-1e20, 1e20) and (lo[4], hi[4]) == (-1e20, 1e20)
    assert not {3, 4} & set(spec["obbt_candidates"].tolist())


@pytest.mark.smoke
def test_default_on_with_an_opt_out(monkeypatch):
    """Graduated on its section 5 panel (performance-plan section 99)."""
    monkeypatch.delenv("DISCOPT_KERNEL_FREE_LINEAR", raising=False)
    assert build_spatial_kernel_spec(_model()) is not None


@pytest.mark.smoke
def test_a_free_factor_still_declines(monkeypatch):
    monkeypatch.setenv("DISCOPT_KERNEL_FREE_LINEAR", "1")
    assert build_spatial_kernel_spec(_model(free_factor=True)) is None


def test_kernel_certifies_the_python_trees_objective(monkeypatch):
    monkeypatch.setenv("DISCOPT_KERNEL_FREE_LINEAR", "1")
    r = _model().solve(time_limit=30)
    assert (r.algorithm_route or "").startswith("native-spatial"), r.algorithm_route
    assert r.status == "optimal" and r.gap_certified
    monkeypatch.setenv("DISCOPT_KERNEL_FREE_LINEAR", "0")
    p = _model().solve(time_limit=30)
    assert not (p.algorithm_route or "").startswith("native-spatial"), p.algorithm_route
    assert p.status == "optimal"
    assert r.objective == pytest.approx(p.objective, abs=1e-5)
    assert r.objective == pytest.approx(_oracle(0, 3, 0, 3), abs=1e-5)


def test_random_boxes_never_bound_above_the_optimum(monkeypatch):
    monkeypatch.setenv("DISCOPT_KERNEL_FREE_LINEAR", "1")
    rng = np.random.default_rng(2011)
    executed = 0
    for _ in range(12):
        xlo, xhi = np.sort(rng.uniform(-3.0, 3.0, 2))
        ylo, yhi = np.sort(rng.uniform(-3.0, 3.0, 2))
        opt = _oracle(xlo, xhi, ylo, yhi)
        r = _model(xlo, xhi, ylo, yhi).solve(time_limit=20)
        assert (r.algorithm_route or "").startswith("native-spatial"), r.algorithm_route
        assert r.bound is not None
        tol = 1e-6 * (1.0 + abs(opt))
        assert r.bound <= opt + tol, (xlo, xhi, ylo, yhi, r.bound, opt)
        if r.objective is not None:
            assert r.objective >= opt - tol
        if r.gap_certified:
            assert r.objective == pytest.approx(opt, abs=1e-4)
        executed += 1
    assert executed == 12

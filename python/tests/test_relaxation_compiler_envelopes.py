"""Envelope-containment property tests for the relaxation compiler (#87).

For each recognized atom shape (bilinear, squares, transcendentals, the
trilinear chain, x*log(x), Monod and Arrhenius kinetics, signed powers) the
compiled relaxation must satisfy ``cv(x) <= f(x) <= cc(x)`` at every sampled
point of the box — the defining property of a valid convex/concave envelope
pair. A violation is a soundness bug, not a tightness matter.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt._jax.relaxation_compiler import compile_objective_relaxation
from discopt.modeling.core import Model

pytestmark = pytest.mark.relaxation


def _check_envelope(build, lb, ub, n=7, partitions=0, tol=1e-7):
    """Assert cv <= f <= cc on a sampled grid of the box."""
    m = Model("env")
    xs = [
        m.continuous(f"v{k}", lb=float(lo), ub=float(hi)) for k, (lo, hi) in enumerate(zip(lb, ub))
    ]
    m.minimize(build(*xs))
    relax_fn = compile_objective_relaxation(m, partitions=partitions)
    ev = NLPEvaluator(m)
    lb_arr = np.asarray(lb, dtype=np.float64)
    ub_arr = np.asarray(ub, dtype=np.float64)
    grids = [np.linspace(lo, hi, n) for lo, hi in zip(lb_arr, ub_arr)]
    mesh = np.meshgrid(*grids)
    pts = np.stack([g.ravel() for g in mesh], axis=1)
    for pt in pts:
        cv, cc = relax_fn(pt, pt, lb_arr, ub_arr)
        f = float(ev.evaluate_objective(pt))
        assert float(cv) <= f + tol, f"cv {float(cv)} > f {f} at {pt}"
        assert float(cc) >= f - tol, f"cc {float(cc)} < f {f} at {pt}"


def test_bilinear_envelope_containment():
    _check_envelope(lambda x, y: x * y, [-1.0, 0.5], [2.0, 3.0])


def test_square_and_linear_mix():
    _check_envelope(lambda x, y: 2.0 * x**2 - 3.0 * y + 1.0, [-2.0, 0.0], [1.0, 2.0])


def test_transcendental_envelopes():
    _check_envelope(lambda x: dm.exp(x), [-1.0], [1.5])
    _check_envelope(lambda x: dm.log(x), [0.5], [3.0])
    _check_envelope(lambda x: dm.sqrt(x), [0.1], [4.0])
    _check_envelope(lambda x: dm.sin(x), [0.0], [3.0])


def test_trilinear_chain_envelope():
    _check_envelope(lambda x, y, z: x * y * z, [0.5, 0.5, 0.5], [2.0, 2.0, 2.0], n=5)
    # A sign-mixed box exercises the general trilinear hull.
    _check_envelope(lambda x, y, z: x * y * z, [-1.0, 0.5, -0.5], [1.0, 2.0, 1.5], n=5)


def test_xlogx_envelope():
    _check_envelope(lambda x: x * dm.log(x), [0.2], [3.0])


def test_monod_kinetics_envelope():
    # Monod rate x / (k + x): recognized as a dedicated atom.
    _check_envelope(lambda x: x / (0.7 + x), [0.0], [4.0])


def test_arrhenius_envelope():
    # Arrhenius exp(-a / x) on a strictly positive box.
    _check_envelope(lambda x: dm.exp(-1.3 / x), [0.4], [3.0])


def test_signed_power_envelope():
    # x * |x| (signpower p=2 smooth form).
    _check_envelope(lambda x: x * dm.abs(x), [-2.0], [2.0])


def test_partitioned_relaxation_still_contains():
    _check_envelope(lambda x, y: x * y, [0.0, 0.0], [4.0, 4.0], partitions=4)


def test_partitioned_relaxation_is_no_looser_at_midpoint():
    # Piecewise McCormick with partitions must be at least as tight as the
    # unpartitioned envelope at the box midpoint.
    m0 = Model("p0")
    x0 = m0.continuous("x", lb=0.0, ub=4.0)
    y0 = m0.continuous("y", lb=0.0, ub=4.0)
    m0.minimize(x0 * y0)
    plain = compile_objective_relaxation(m0, partitions=0)

    m4 = Model("p4")
    x4 = m4.continuous("x", lb=0.0, ub=4.0)
    y4 = m4.continuous("y", lb=0.0, ub=4.0)
    m4.minimize(x4 * y4)
    part = compile_objective_relaxation(m4, partitions=4)

    lb = np.array([0.0, 0.0])
    ub = np.array([4.0, 4.0])
    mid = 0.5 * (lb + ub)
    cv0, cc0 = plain(mid, mid, lb, ub)
    cv4, cc4 = part(mid, mid, lb, ub)
    assert float(cv4) >= float(cv0) - 1e-9
    assert float(cc4) <= float(cc0) + 1e-9

"""Regression tests for #923: the dense Lagrangian Hessian must not go wrong at scale.

``NLPEvaluator.evaluate_lagrangian_hessian`` built the dense matrix with
``jacfwd(jacfwd(L))``. ``L`` is a scalar, so the inner forward pass carries one
tangent per input through every intermediate and the length-``m`` constraint
vector becomes an ``(n, n, m)`` buffer. The peak allocation is exactly

    2 * n^2 * m * 8 bytes

— measured, not estimated: XLA's ``RESOURCE_EXHAUSTED`` request matched that
closed form to the byte on the emfl-shaped replica below, at n=1068/m=1050
(19,162,483,200) and n=1368/m=1350 (40,422,758,400).

That is 66 GB on MINLPLib's ``emfl050_3_3`` (n=1611, m=1593) and 413 GB on
``emfl100_3_3`` (n=2961, m=2943), which is precisely why #923 reported the first
as correct and the second as wrong. On Linux the over-large buffer surfaces as a
clean ``RESOURCE_EXHAUSTED``; in the #923 report it surfaced as an allocation
that read back as zeros, so ``evaluate_lagrangian_hessian`` returned an all-zero
matrix — right shape, no exception, no warning — while the sparse
``evaluate_hessian_values`` path and finite differences both gave the true value.

The fix routes the dense build to ``jacfwd(grad(L))`` (peak ``~2 * n * m * 8``)
once the forward-over-forward peak exceeds a byte budget. Both nestings compute
the same matrix, so the change is bound-neutral; that is asserted directly here.

There was no cross-check between the dense Hessian, the sparse Hessian and
finite differences anywhere in the suite, which is why the defect survived. The
first test closes that gap.
"""

from __future__ import annotations

import discopt._jax.nlp_evaluator as nev
import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import (
    NLPEvaluator,
    _dense_hessian_fwd_over_fwd_peak_bytes,
)


def _emfl_shaped_model(n_cone: int, n_new: int = 9, seed: int = 0) -> dm.Model:
    """The structural shape of MINLPLib's ``emfl*_3_3`` family.

    ``n_cone`` distance variables, ``n_cone`` coordinate-difference pairs and
    ``n_new`` facility positions in 2-D, giving ``n = 3 * n_cone + 2 * n_new``
    and ``m = 3 * n_cone``:

        cone rows    -d_r^2 + u_r^2 + v_r^2 <= 0        (n_cone of them)
        linear rows  u_r - px_j + ax_r == 0, likewise v (2 * n_cone of them)

    At ``n_cone = 981`` that is exactly emfl100_3_3's n=2961 / m=2943, and row 0
    is ``-x0**2 + x981**2 + x982**2 <= 0`` — the row #923 reports.
    """
    rng = np.random.default_rng(seed)
    m = dm.Model(f"emfl_shape_{n_cone}")
    d = [m.continuous(f"d{r}", lb=0.0, ub=100.0) for r in range(n_cone)]
    uv = [
        (
            m.continuous(f"u{r}", lb=-100.0, ub=100.0),
            m.continuous(f"v{r}", lb=-100.0, ub=100.0),
        )
        for r in range(n_cone)
    ]
    p = [
        (m.continuous(f"px{j}", lb=-100.0, ub=100.0), m.continuous(f"py{j}", lb=-100.0, ub=100.0))
        for j in range(n_new)
    ]
    for r in range(n_cone):
        u, v = uv[r]
        m.subject_to(-(d[r] ** 2) + u**2 + v**2 <= 0)
    for r in range(n_cone):
        u, v = uv[r]
        px, py = p[r % n_new]
        m.subject_to(u - px + float(rng.uniform(-10, 10)) == 0)
        m.subject_to(v - py + float(rng.uniform(-10, 10)) == 0)
    # Balanced-tree sum: a left-nested chain of ~1000 additions used to trip the
    # dag_compiler recursion limit noted at the end of #923 (a separate defect,
    # fixed under #925). Kept as-is so this test keeps measuring the Hessian,
    # not the compiler; the depth itself is covered by
    # test_925_dag_compiler_deep_chain.py.
    terms: list = list(d)
    while len(terms) > 1:
        terms = [
            terms[i] + terms[i + 1] if i + 1 < len(terms) else terms[i]
            for i in range(0, len(terms), 2)
        ]
    m.minimize(terms[0])
    return m


def _sparse_at(ev: NLPEvaluator, x, obj_factor, lam, i, j) -> float:
    rows, cols = ev.hessian_structure()
    vals = np.asarray(ev.evaluate_hessian_values(x, obj_factor, lam))
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    mask = ((rows == i) & (cols == j)) | ((rows == j) & (cols == i))
    return float(vals[mask].sum())


def _fd_second_derivative(ev: NLPEvaluator, x, lam, i, h=1e-4) -> float:
    """Central second difference of ``lam . g(x)`` in coordinate ``i``.

    Uses only constraint *values* — no AD Hessian on either side — so it is a
    neutral arbiter between the dense and sparse paths.
    """

    def L(xx):
        return float(np.dot(lam, ev.evaluate_constraints(xx)))

    xp = np.asarray(x, dtype=np.float64).copy()
    xm = xp.copy()
    xp[i] += h
    xm[i] -= h
    return (L(xp) - 2.0 * L(x) + L(xm)) / h**2


@pytest.mark.unit
def test_dense_sparse_and_finite_differences_agree_on_emfl_shape():
    """The cross-check that did not exist: dense vs sparse vs finite differences.

    Row 0 is ``-d_0^2 + u_0^2 + v_0^2``, so the exact curvature is
    ``diag(-2, +2, +2)`` on ``(0, 0)``, ``(n_cone, n_cone)`` and
    ``(n_cone + 1, n_cone + 1)`` with ``lam = e_0`` -- the ``u``/``v`` pair is
    interleaved, exactly as emfl100_3_3's ``-x0**2 + x981**2 + x982**2``.
    """
    n_cone = 60
    ev = NLPEvaluator(_emfl_shaped_model(n_cone))
    x = np.full(ev.n_variables, 1.3)
    lam = np.zeros(ev.n_constraints)
    lam[0] = 1.0

    H = np.asarray(ev.evaluate_lagrangian_hessian(x, 0.0, lam))
    assert H.shape == (ev.n_variables, ev.n_variables)

    checked = 0
    for idx, expected in ((0, -2.0), (n_cone, 2.0), (n_cone + 1, 2.0)):
        assert H[idx, idx] == pytest.approx(expected, abs=1e-9)
        assert _sparse_at(ev, x, 0.0, lam, idx, idx) == pytest.approx(expected, abs=1e-9)
        assert _fd_second_derivative(ev, x, lam, idx) == pytest.approx(expected, rel=1e-3)
        checked += 1
    assert checked == 3, "cross-check asserted nothing"

    # An all-zero Hessian is the #923 signature; assert against it explicitly.
    assert np.abs(H).max() > 0.0
    assert np.count_nonzero(H) == 3

    # Full-multiplier arm: the dense and sparse supports must coincide, not be
    # disjoint (#923 reported 1571 dense nonzeros vs 2943 sparse, overlap 0).
    lam_all = np.ones(ev.n_constraints)
    H_all = np.asarray(ev.evaluate_lagrangian_hessian(x, 0.0, lam_all))
    rows, cols = ev.hessian_structure()
    vals = np.asarray(ev.evaluate_hessian_values(x, 0.0, lam_all))
    assert np.allclose(vals, H_all[np.asarray(rows), np.asarray(cols)], atol=1e-12)
    assert np.count_nonzero(H_all) == 3 * n_cone


@pytest.mark.unit
def test_peak_byte_model_and_gate_routing():
    """The gate is on the measured ``2 * n^2 * m * 8`` peak, and it routes."""
    assert _dense_hessian_fwd_over_fwd_peak_bytes(1068, 1050) == 19_162_483_200
    assert _dense_hessian_fwd_over_fwd_peak_bytes(1368, 1350) == 40_422_758_400
    # emfl050_3_3 (correct in #923) vs emfl100_3_3 (wrong in #923).
    assert _dense_hessian_fwd_over_fwd_peak_bytes(1611, 1593) / 2**30 == pytest.approx(61.6, abs=1)
    assert _dense_hessian_fwd_over_fwd_peak_bytes(2961, 2943) / 2**30 == pytest.approx(384.5, abs=1)
    # m floored at 1 so an unconstrained model is still charged for (n, n).
    assert _dense_hessian_fwd_over_fwd_peak_bytes(10, 0) == 2 * 100 * 8

    small = NLPEvaluator(_emfl_shaped_model(20))
    assert small._dense_hessian_mode == "fwd_over_fwd"
    assert (
        _dense_hessian_fwd_over_fwd_peak_bytes(small.n_variables, small.n_constraints)
        <= nev._DENSE_HESSIAN_FWD_OVER_FWD_PEAK_BYTES
    )

    big = NLPEvaluator(_emfl_shaped_model(200))
    assert (
        _dense_hessian_fwd_over_fwd_peak_bytes(big.n_variables, big.n_constraints)
        > nev._DENSE_HESSIAN_FWD_OVER_FWD_PEAK_BYTES
    )
    assert big._dense_hessian_mode == "fwd_over_rev"


@pytest.mark.unit
def test_both_ad_nestings_return_the_same_matrix(monkeypatch):
    """Bound-neutrality: the routing changes memory, never a value.

    AD mode never changes a derivative, but it does change the *summation order*
    inside it, so the two builds are equal to floating-point rounding rather than
    bit-identical. A differential panel over the in-repo ``.nl`` corpus (both
    nestings forced, two multiplier settings each, 130 comparisons over 60
    instances; ``hda.nl`` excluded because its forward-over-forward peak is
    5.6 GiB) put the worst RELATIVE entrywise drift at 4.1e-15, on
    ``syn05hfsg.nl``. The tolerance below is that scale.

    Two instances (``4stufen``, ``beuster``) additionally differ in where the
    matrix is non-finite when evaluated at an out-of-domain point, and the
    asymmetry is one-directional in all four arms: 21009/21605/23393/24021
    entries are non-finite under forward-over-forward and finite under
    forward-over-reverse, and zero entries go the other way. The routed path is
    never less defined than the one it replaces.
    """
    model = _emfl_shaped_model(20)
    x = 1.3 + 0.05 * np.arange(3 * 20 + 18, dtype=np.float64)

    out = {}
    for mode, budget in (("fwd_over_fwd", 1 << 62), ("fwd_over_rev", 0)):
        monkeypatch.setattr(nev, "_DENSE_HESSIAN_FWD_OVER_FWD_PEAK_BYTES", budget)
        ev = NLPEvaluator(model)
        assert ev._dense_hessian_mode == mode
        lam = np.linspace(-1.0, 1.0, ev.n_constraints)
        out[mode] = np.asarray(ev.evaluate_lagrangian_hessian(x, 1.0, lam))

    a, b = out["fwd_over_fwd"], out["fwd_over_rev"]
    assert a.shape == b.shape
    scale = max(float(np.abs(a).max()), 1.0)
    assert float(np.abs(a - b).max()) <= 1e-13 * scale
    assert np.count_nonzero(a) == np.count_nonzero(b)
    assert np.abs(a).max() > 0.0


@pytest.mark.slow
def test_dense_hessian_correct_past_the_forward_over_forward_memory_wall():
    """The real #923 regression, at a shape no machine can afford unfixed.

    ``n_cone = 400`` gives n=1218 / m=1200, whose forward-over-forward peak is
    26.5 GiB (XLA requested exactly 28,483,660,800 bytes here before the fix).
    The routed forward-over-reverse build needs ~22 MiB of intermediates and
    returns the true curvature.
    """
    n_cone = 400
    ev = NLPEvaluator(_emfl_shaped_model(n_cone))
    assert ev.n_variables == 3 * n_cone + 18
    assert ev.n_constraints == 3 * n_cone
    assert ev.is_gauss_newton is False
    assert ev._dense_hessian_mode == "fwd_over_rev"
    assert (
        _dense_hessian_fwd_over_fwd_peak_bytes(ev.n_variables, ev.n_constraints) == 28_483_660_800
    )

    x = np.full(ev.n_variables, 1.3)
    lam = np.zeros(ev.n_constraints)
    lam[0] = 1.0
    H = np.asarray(ev.evaluate_lagrangian_hessian(x, 0.0, lam))

    checked = 0
    for idx, expected in ((0, -2.0), (n_cone, 2.0), (n_cone + 1, 2.0)):
        assert H[idx, idx] == pytest.approx(expected, abs=1e-9)
        assert _sparse_at(ev, x, 0.0, lam, idx, idx) == pytest.approx(expected, abs=1e-9)
        assert _fd_second_derivative(ev, x, lam, idx) == pytest.approx(expected, rel=1e-3)
        checked += 1
    assert checked == 3, "regression asserted nothing"
    assert np.count_nonzero(H) == 3

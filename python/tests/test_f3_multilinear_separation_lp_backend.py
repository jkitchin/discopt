"""F3 (extends perf-d1/#484): the multilinear vertex-hull separation LP routes to
the in-house warm simplex, not a cold POUNCE IPM per call, under the same
``DISCOPT_SEPARATION_LP_SIMPLEX`` flag as the edge-concave separator.

Regime: bound-neutral-with-caveat (a degenerate hull LP's dual may differ between
the IPM and the simplex, so the derived cut can differ). The cut's *validity* is
engine-independent — :func:`_solve_envelope` recomputes the intercept to the exact
validity boundary over the box vertices, so ``a·x + b`` bounds the monomial
everywhere for ANY slope. These tests lock the routing (flag honored + off-switch),
the per-LP POUNCE fallback (so a cold-simplex stall never regresses), and cut
soundness on both engine paths (feasible-point sampling removes no true point).

Fails on the pre-F3 code, which hard-coded ``lp_pounce.solve_lp`` in
``multilinear_separation._solve_envelope``.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.filterwarnings("ignore")


def test_multilinear_separation_lp_simplex_flag_honored(monkeypatch):
    """The F3 env selector mirrors the edge-concave flag (default ON, '0' off)."""
    from discopt._jax.multilinear_separation import _separation_lp_simplex_enabled

    monkeypatch.delenv("DISCOPT_SEPARATION_LP_SIMPLEX", raising=False)
    assert _separation_lp_simplex_enabled() is True  # default ON

    monkeypatch.setenv("DISCOPT_SEPARATION_LP_SIMPLEX", "1")
    assert _separation_lp_simplex_enabled() is True

    for off in ("0", "false", "no", "off", "OFF"):
        monkeypatch.setenv("DISCOPT_SEPARATION_LP_SIMPLEX", off)
        assert _separation_lp_simplex_enabled() is False


def test_multilinear_separator_uses_simplex_when_on(monkeypatch):
    """Flag ON (default): the hull LP is solved by the in-house simplex, and the
    POUNCE LP solver is NOT called on a well-conditioned (simplex-converging) LP."""
    import discopt.solvers.lp_pounce as lp_pounce
    import discopt.solvers.lp_simplex as lp_simplex
    from discopt._jax import multilinear_separation as ms

    if not lp_simplex.SIMPLEX_AVAILABLE:
        pytest.skip("in-house simplex binding not built")

    monkeypatch.setenv("DISCOPT_SEPARATION_LP_SIMPLEX", "1")

    pounce_calls = {"n": 0}
    _orig_pounce = lp_pounce.solve_lp

    def _count_pounce(*a, **k):
        pounce_calls["n"] += 1
        return _orig_pounce(*a, **k)

    monkeypatch.setattr(lp_pounce, "solve_lp", _count_pounce)

    simplex_calls = {"n": 0}
    _orig_simplex = lp_simplex.solve_lp

    def _count_simplex(*a, **k):
        simplex_calls["n"] += 1
        return _orig_simplex(*a, **k)

    monkeypatch.setattr(lp_simplex, "solve_lp", _count_simplex)

    # A trilinear product w = x0*x1*x2 over a box, with w* far below the hull ->
    # both under/over hull LPs are solved.
    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    xs = np.array([0.5, 0.5, 0.5])
    cuts = ms.separate_multilinear_envelope(lb, ub, xs, w_star=-5.0)
    assert cuts, "expected a violated multilinear hull cut"
    assert simplex_calls["n"] >= 1, "flag ON must route the hull LP to the simplex"
    assert pounce_calls["n"] == 0, "well-conditioned hull LP must not fall back to POUNCE"


def test_multilinear_separator_off_switch_uses_pounce(monkeypatch):
    """Off-switch ('0') restores the POUNCE-IPM path (the simplex is not called)."""
    import discopt.solvers.lp_pounce as lp_pounce
    import discopt.solvers.lp_simplex as lp_simplex
    from discopt._jax import multilinear_separation as ms

    monkeypatch.setenv("DISCOPT_SEPARATION_LP_SIMPLEX", "0")

    pounce_calls = {"n": 0}
    _orig_pounce = lp_pounce.solve_lp

    def _count_pounce(*a, **k):
        pounce_calls["n"] += 1
        return _orig_pounce(*a, **k)

    monkeypatch.setattr(lp_pounce, "solve_lp", _count_pounce)

    simplex_calls = {"n": 0}
    _orig_simplex = lp_simplex.solve_lp

    def _count_simplex(*a, **k):
        simplex_calls["n"] += 1
        return _orig_simplex(*a, **k)

    monkeypatch.setattr(lp_simplex, "solve_lp", _count_simplex)

    lb = np.array([0.0, 0.0, 0.0])
    ub = np.array([1.0, 1.0, 1.0])
    xs = np.array([0.5, 0.5, 0.5])
    cuts = ms.separate_multilinear_envelope(lb, ub, xs, w_star=-5.0)
    assert cuts, "expected a violated multilinear hull cut on the POUNCE path"
    assert pounce_calls["n"] >= 1, "off-switch must use POUNCE"
    assert simplex_calls["n"] == 0, "off-switch must not call the simplex"


def _sample_cut_valid(cut, lb, ub, seed=0, npts=2000):
    """A hull cut removes no true (x, prod x) point over the box."""
    rng = np.random.default_rng(seed)
    n = lb.shape[0]
    pts = lb + (ub - lb) * rng.random((npts, n))
    w = np.prod(pts, axis=1)
    est = pts @ cut.a + cut.b
    if cut.sense == "under":  # w >= a.x + b  ->  est <= w everywhere
        return bool(np.all(est <= w + 1e-6))
    return bool(np.all(est >= w - 1e-6))  # over: est >= w


@pytest.mark.parametrize("flag", ["1", "0"])
def test_multilinear_cut_sound_on_both_engines(monkeypatch, flag):
    """The separated cut is a valid under/over estimator over the whole box on
    BOTH the simplex (flag ON) and POUNCE (flag OFF) paths — feasible-point
    sampling removes no true point. Validity is engine-independent because the
    intercept is recomputed to the exact boundary over the box vertices."""
    import discopt.solvers.lp_simplex as lp_simplex
    from discopt._jax import multilinear_separation as ms

    if flag == "1" and not lp_simplex.SIMPLEX_AVAILABLE:
        pytest.skip("in-house simplex binding not built")
    monkeypatch.setenv("DISCOPT_SEPARATION_LP_SIMPLEX", flag)

    cases = [
        (np.array([-2.0, 0.5]), np.array([3.0, 4.0]), np.array([0.1, 2.0]), -50.0),
        (np.array([-2.0, 0.5]), np.array([3.0, 4.0]), np.array([0.1, 2.0]), 50.0),
        (np.array([0.0, 0.0, -1.0]), np.array([2.0, 2.0, 1.0]), np.array([1.0, 1.0, 0.0]), -20.0),
    ]
    got_any = False
    for lb, ub, xs, w_star in cases:
        cuts = ms.separate_multilinear_envelope(lb, ub, xs, float(w_star))
        for cut in cuts:
            got_any = True
            assert _sample_cut_valid(cut, lb, ub), (
                f"[flag={flag}] {cut.sense} cut removes a feasible point"
            )
    assert got_any, "expected at least one violated cut across the cases"


def test_multilinear_capped_simplex_falls_back_to_pounce(monkeypatch):
    """When the capped simplex returns non-optimal (simulated stall), the hull LP
    falls back to POUNCE and still yields a valid cut — so a cold-simplex stall
    never drops a cut the POUNCE path would have found."""
    import discopt.solvers.lp_pounce as lp_pounce
    import discopt.solvers.lp_simplex as lp_simplex
    from discopt._jax import multilinear_separation as ms
    from discopt.solvers import LPResult, SolveStatus

    if not lp_simplex.SIMPLEX_AVAILABLE:
        pytest.skip("in-house simplex binding not built")
    monkeypatch.setenv("DISCOPT_SEPARATION_LP_SIMPLEX", "1")

    # Force every simplex hull-LP solve to look like a stall (iteration limit).
    monkeypatch.setattr(
        lp_simplex, "solve_lp", lambda *a, **k: LPResult(status=SolveStatus.ITERATION_LIMIT)
    )

    pounce_calls = {"n": 0}
    _orig_pounce = lp_pounce.solve_lp

    def _count_pounce(*a, **k):
        pounce_calls["n"] += 1
        return _orig_pounce(*a, **k)

    monkeypatch.setattr(lp_pounce, "solve_lp", _count_pounce)

    lb = np.array([-2.0, 0.5])
    ub = np.array([3.0, 4.0])
    xs = np.array([0.1, 2.0])
    cuts = ms.separate_multilinear_envelope(lb, ub, xs, w_star=-50.0)
    assert cuts, "fallback must still separate a cut when the simplex stalls"
    assert pounce_calls["n"] >= 1, "a simplex stall must fall back to POUNCE"
    for cut in cuts:
        assert _sample_cut_valid(cut, lb, ub), "fallback cut removes a feasible point"


def test_lp_simplex_max_iter_passthrough_caps_pivots():
    """``lp_simplex.solve_lp(max_iter=1)`` returns a non-optimal status rather than
    a (possibly wrong) objective — the cap is soundness-neutral (never a bound)."""
    import discopt.solvers.lp_simplex as lp_simplex
    from discopt.solvers import SolveStatus

    if not lp_simplex.SIMPLEX_AVAILABLE:
        pytest.skip("in-house simplex binding not built")

    # A small LP that needs >1 pivot; capping at 1 must NOT report OPTIMAL.
    c = np.array([-1.0, -1.0])
    A_ub = np.array([[1.0, 2.0], [2.0, 1.0]])
    b_ub = np.array([4.0, 4.0])
    bounds = [(0.0, 10.0), (0.0, 10.0)]
    r_full = lp_simplex.solve_lp(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds)
    assert r_full.status == SolveStatus.OPTIMAL
    r_capped = lp_simplex.solve_lp(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, max_iter=1)
    # Either it happened to converge in 1 pivot (still optimal + same obj) or it
    # capped out (non-optimal, no bound) — never a wrong optimum.
    if r_capped.status == SolveStatus.OPTIMAL:
        assert abs(r_capped.objective - r_full.objective) < 1e-6
    else:
        assert r_capped.status == SolveStatus.ITERATION_LIMIT
        assert r_capped.objective is None

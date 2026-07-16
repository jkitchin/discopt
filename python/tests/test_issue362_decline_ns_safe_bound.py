"""Regression tests for #362 part (b): NS safe bound at the certification edge.

Root cause (``docs/dev/nvs05-decline-taint-2026-07-16.md``): at nvs05's
certification edge one node LP (99 cols / 378 rows, coefficient magnitude up to
2.3e9 from the lifted square/product envelopes at aux scale ~1.85e8 = 13600²)
defeats BOTH in-house warm simplex attempts (bare and equilibrated: numerical /
iteration-limit), and the generic cold path that then solves it ``optimal``
(LP optimum 5.470728) carries no certificate of its own. ``_certify`` therefore
sees ``safe_bound=None`` on a fully-finite LP, and its conditioning guard
(``_max_finite_magnitude`` 2.3e9 > 1e7) rightly refuses to trust the raw vertex
— so the node produces NO bound, its NLP-failure sentinel survives, and it is
non-rigorously sentinel-fathomed carrying pop-time floor 5.469616: 2.4e-4 below
the incumbent 5.4709341, just outside the 1e-4 certification tolerance. One
declined node keeps the whole exhausted tree at ``feasible``.

Meanwhile both failed warm attempts had ALREADY yielded rigorous
Neumaier–Shcherbina safe bounds from their own dual candidates (5.288 bare,
5.4658 equilibrated) — valid for ANY multiplier vector by weak duality — and
threw them away.

Fix (same flag as #517, ``DISCOPT_NODE_NUMERICAL_DUAL_BOUND``, default OFF —
bound-changing regime): surface the stashed NS bound as ``safe_bound`` on an
``optimal`` generic-path solve, so ``_certify`` certifies the node (which then
*branches* on its rigorous bound instead of tainting the tree). A finite NS
bound is itself a proof the LP is bounded, so this can never fabricate a bound
on a genuinely unbounded relaxation (himmel16 class).

Measured (in-container, 2026-07-16): nvs05 default config, tl=180 — flag OFF
``feasible`` / bound 5.469616074027518 / 173 nodes (byte-identical to
pre-change); flag ON ``optimal`` / bound 5.4705684830157715 / 179 nodes — the
first full rigorous certificate on nvs05 (issue #362).

The declining node LP is vendored verbatim (``data/nvs05_node171_decline_lp.npz``,
extracted from the live solve) so the unit tests below exercise the exact
warm-fails/equilibrated-fails/generic-optimal chain without a 100 s solve.
"""

import math
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest
import scipy.sparse as sp

_DATA = os.path.join(os.path.dirname(__file__), "data")
_LP_NPZ = os.path.join(_DATA, "nvs05_node171_decline_lp.npz")
_NVS05_NL = os.path.join(_DATA, "minlplib", "nvs05.nl")

_FLAG = "DISCOPT_NODE_NUMERICAL_DUAL_BOUND"
_NVS05_OPT = 5.4709341  # MINLPLib reference optimum (minimize)
# The vendored node LP's true optimum (generic cold path, internally
# equilibrated + dual-feasibility-verified): any valid NS bound must stay below.
_NODE_LP_OPT = 5.470728084010654


def _load_decline_lp():
    d = np.load(_LP_NPZ)
    A = sp.csr_matrix((d["A_data"], d["A_indices"], d["A_indptr"]), shape=tuple(d["A_shape"]))
    bounds = [(float(lo), float(hi)) for lo, hi in d["bounds"]]
    return d["c"], A, d["b_ub"], bounds, float(d["obj_offset"][0])


def _solve_decline_lp():
    from discopt._jax.milp_relaxation import MilpRelaxationModel

    c, A, b, bounds, off = _load_decline_lp()
    milp = MilpRelaxationModel(c, A, b, bounds, obj_offset=off)
    return milp.solve(backend="simplex")


def test_decline_lp_flag_on_surfaces_ns_safe_bound(monkeypatch):
    """Flag ON: the generic-path optimal result carries the stashed NS safe
    bound (fails before the #362 fix: ``safe_bound=None`` → node declined)."""
    monkeypatch.setenv(_FLAG, "1")
    res = _solve_decline_lp()
    assert res.status == "optimal", f"vendored LP must solve optimal, got {res.status}"
    assert res.safe_bound is not None, (
        "the NS safe bound recovered from the failed warm/equilibrated duals "
        "must be surfaced on the generic-path optimal result (#362)"
    )
    assert math.isfinite(res.safe_bound)
    # Soundness: an NS bound is valid for ANY dual, so it never exceeds the
    # LP's true optimum (weak duality).
    assert res.safe_bound <= _NODE_LP_OPT + 1e-9, (
        f"UNSOUND: NS safe bound {res.safe_bound!r} exceeds the LP optimum {_NODE_LP_OPT}"
    )
    # Usefulness: at the certification edge the bound must be materially above
    # the rest of the frontier (the measured equilibrated-dual value is 5.4658).
    assert res.safe_bound >= 5.0, f"NS bound unexpectedly loose: {res.safe_bound!r}"


def test_decline_lp_flag_off_baseline_unchanged(monkeypatch):
    """Default (flag OFF): the generic path still reports no certificate — the
    fix is opt-in and the default search is byte-identical."""
    monkeypatch.delenv(_FLAG, raising=False)
    res = _solve_decline_lp()
    assert res.status == "optimal"
    assert res.safe_bound is None, (
        f"flag OFF must leave the generic path certificate-free, got {res.safe_bound!r}"
    )


@pytest.mark.slow
def test_nvs05_flag_on_certifies_global_optimum(monkeypatch):
    """End-to-end (issue #362 part b): with the flag ON, nvs05 under the default
    config produces a sound dual bound past the 5.0 bar — and, once the tree
    exhausts, a full rigorous certificate (measured in-container: ``optimal`` /
    bound 5.47057 / ~110 s wall)."""
    monkeypatch.setenv(_FLAG, "1")
    import discopt.modeling as dm

    r = dm.from_nl(_NVS05_NL).solve(time_limit=180)
    assert r.objective is not None and abs(r.objective - _NVS05_OPT) < 1e-2
    assert r.bound is not None and math.isfinite(r.bound)
    # Soundness first: the reported dual bound never crosses the oracle optimum
    # and never crosses the incumbent (certificate invariant, minimize).
    assert r.bound <= _NVS05_OPT + 1e-3, f"UNSOUND: bound {r.bound!r} crosses the optimum"
    assert r.bound <= r.objective + 1e-9
    # Issue #362 (b) acceptance: the dual bound certifies past 5.0.
    assert r.bound >= 5.0, f"nvs05 dual bound did not certify: {r.bound!r}"
    # No false certificate: "optimal" is only legitimate with a closed gap.
    if r.status == "optimal":
        gap = (r.objective - r.bound) / max(1.0, abs(r.objective))
        assert gap <= 1e-3, f"false certificate: status=optimal with gap {gap:.3g}"

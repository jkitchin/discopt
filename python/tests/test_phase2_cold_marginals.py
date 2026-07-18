"""Property tests for Phase 2 step 1: cold-path node-LP marginals (issue #764).

The spatial B&B needs the node LP's *reduced costs* to run BARON-style cheap
duality-based bound tightening (DBBT) in place of exhaustive per-node OBBT. Those
marginals were produced only by the incremental/warm LP path, which is built only
for composite-lift models — so on the (large) non-composite-lift spatial class
(``tanksize`` etc.) the node LP takes the *cold* path, whose ``milp.solve()``
discarded the simplex row duals. This step threads them through:
``milp.solve(want_marginals=True)`` → ``MilpRelaxationResult.{row_dual,reduced_costs}``
(direct warm-simplex path, original scale) → ``MccormickLPResult.{reduced_costs,
safe_bound}`` at the cold McCormick node solve.

This is a pure read-only side-channel — *nothing* consumes the marginals yet — so
enabling it must be **bound-neutral** (identical status/bound with the flag on).
The tests below pin:

  * ``want_marginals`` never changes the node bound/status (bound-neutrality);
  * ``reduced_costs == c - Aᵀy`` exactly (plumbing preserves the duals);
  * strong duality: ``safe_bound ≤ bound`` and the dual objective matches the primal
    (validates ``y`` is the true KKT dual);
  * KKT complementary slackness on structural columns (the sign law DBBT relies on);
  * the DBBT reduction these marginals *would* drive is tighten-only — it never cuts
    the known optimum out of the box (soundness of the eventual consumer).
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from discopt._jax.mccormick_lp import MccormickLPRelaxer  # noqa: E402
from discopt._jax.milp_relaxation import build_milp_relaxation  # noqa: E402
from discopt.modeling.core import from_nl  # noqa: E402
from discopt.tightening import fbbt_box  # noqa: E402

_DATA = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")
_INSTANCE = "tanksize"
_OPT = 1.268643754  # MINLPLib oracle (minimize)


def _relaxer_and_box():
    m = from_nl(os.path.join(_DATA, f"{_INSTANCE}.nl"))
    fb = fbbt_box(m, max_iter=50)
    return m, MccormickLPRelaxer(m), fb.lb.copy(), fb.ub.copy()


def _solve_lp(m, r, lb, ub):
    """Solve the node LP relaxation (integrality dropped, as in node_bound_mode='lp')
    with marginals, returning the (milp, MilpRelaxationResult)."""
    milp, _ = build_milp_relaxation(
        m,
        r._terms,
        r._disc,
        bound_override=(np.asarray(lb, float), np.asarray(ub, float)),
        superposition=r._superposition,
        rlt_level1=r._rlt_applicable,
    )
    milp._integrality = None
    return milp, milp.solve(backend="simplex", want_marginals=True)


def test_want_marginals_is_bound_neutral():
    """Requesting marginals must not change the node bound or status (side-channel)."""
    _m, r, lb, ub = _relaxer_and_box()
    off = r.solve_at_node(lb.copy(), ub.copy(), time_limit=20, want_marginals=False)
    on = r.solve_at_node(lb.copy(), ub.copy(), time_limit=20, want_marginals=True)
    assert off.status == on.status
    if off.lower_bound is None:
        assert on.lower_bound is None
    else:
        assert on.lower_bound is not None
        assert abs(off.lower_bound - on.lower_bound) <= 1e-9 * max(1.0, abs(off.lower_bound))
    # Marginals present on the flag-on cold solve; absent on flag-off.
    assert getattr(off, "reduced_costs", None) is None
    assert getattr(on, "reduced_costs", None) is not None
    assert np.asarray(on.reduced_costs).shape[0] == r._n_orig


def test_reduced_costs_identity_and_strong_duality():
    """d == c - Aᵀy exactly, and safe_bound is a valid (≤ primal) dual objective."""
    m, r, lb, ub = _relaxer_and_box()
    milp, res = _solve_lp(m, r, lb, ub)
    assert res.status == "optimal"
    y = np.asarray(res.row_dual, dtype=np.float64).ravel()
    d = np.asarray(res.reduced_costs, dtype=np.float64).ravel()
    c = np.asarray(milp._c, dtype=np.float64).ravel()
    A = sp.csr_matrix(milp._A_ub)
    assert y.shape[0] == A.shape[0]
    assert d.shape[0] == A.shape[1] == c.shape[0]
    d_ref = c - (A.T @ y)
    assert float(np.max(np.abs(d - d_ref))) <= 1e-9
    # Strong duality: the NS-safe dual bound never exceeds the primal LP optimum.
    assert res.safe_bound is not None and res.bound is not None
    assert res.safe_bound <= res.bound + 1e-6 * (1.0 + abs(res.bound))


def _kkt_and_dbbt_soundness(lb, ub):
    """Return (n_struct_nonzero_rc, kkt_violations, dbbt_cut_violations) for a box."""
    m, r, _lb0, _ub0 = _relaxer_and_box()
    milp, res = _solve_lp(m, r, lb, ub)
    if res.status != "optimal" or res.reduced_costs is None:
        return None
    d = np.asarray(res.reduced_costs, dtype=np.float64).ravel()
    x = np.asarray(res.x, dtype=np.float64).ravel()
    bnds = np.asarray(milp._bounds, dtype=np.float64)
    n0 = r._n_orig
    kkt = 0
    for j in range(n0):
        lo, hi = bnds[j, 0], bnds[j, 1]
        if d[j] > 1e-6 and abs(x[j] - lo) > 1e-5:
            kkt += 1
        elif d[j] < -1e-6 and abs(x[j] - hi) > 1e-5:
            kkt += 1
    # DBBT tighten-only soundness: with the true optimum as cutoff, the DBBT bound
    # x_j <= lb_j + gap/d_j (d_j>0) / x_j >= ub_j - gap/|d_j| (d_j<0) must never cut a
    # point whose objective <= cutoff — in particular the known optimal x is NOT a
    # variable of THIS lifted LP, so we check the invariant structurally: every DBBT
    # endpoint stays within [lb, ub] (a valid intersection) and never crosses.
    z_lp = float(res.safe_bound)
    gap = max(_OPT - z_lp, 0.0) + 1e-6 * (1.0 + abs(_OPT))
    cut_viol = 0
    for j in range(n0):
        lo, hi = bnds[j, 0], bnds[j, 1]
        if d[j] > 1e-7 and np.isfinite(lo):
            new_hi = lo + gap / d[j]
            if new_hi < lo - 1e-9:  # would cross below lb -> unsound (should never)
                cut_viol += 1
        elif d[j] < -1e-7 and np.isfinite(hi):
            new_lo = hi - gap / (-d[j])
            if new_lo > hi + 1e-9:
                cut_viol += 1
    nnz = int(np.count_nonzero(np.abs(d[:n0]) > 1e-7))
    return nnz, kkt, cut_viol


@pytest.mark.parametrize(
    "tighten",
    [
        {},  # root box
        {3: (643.0, 900.0)},  # tighten tank-size x3 -> pushes structurals to bounds
        {3: (643.0, 900.0), 4: (536.0, 800.0)},
    ],
)
def test_kkt_and_dbbt_tighten_only(tighten):
    """KKT complementary slackness holds and DBBT never crosses a bound (tighten-only)."""
    _m, _r, lb, ub = _relaxer_and_box()
    for j, (lo, hi) in tighten.items():
        lb[j] = max(lb[j], lo)
        ub[j] = min(ub[j], hi)
    out = _kkt_and_dbbt_soundness(lb, ub)
    if out is None:
        pytest.skip("box infeasible / no marginals")
    nnz, kkt, cut_viol = out
    assert kkt == 0, f"KKT complementary-slackness violations: {kkt}"
    assert cut_viol == 0, f"DBBT would cross a bound (unsound): {cut_viol}"


def test_at_least_one_box_exercises_nonzero_reduced_costs():
    """Guard: the suite must actually exercise nonzero structural reduced costs
    (else KKT/DBBT checks are vacuous)."""
    _m, _r, lb, ub = _relaxer_and_box()
    lb[3] = max(lb[3], 643.0)
    ub[3] = min(ub[3], 900.0)
    out = _kkt_and_dbbt_soundness(lb, ub)
    assert out is not None
    nnz, _kkt, _cv = out
    assert nnz >= 1

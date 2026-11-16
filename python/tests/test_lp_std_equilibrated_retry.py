"""The Rust std-form LP engine retries a numerical breakdown on an exact equilibration.

``RustEngine`` (``lp_std_engine``) used to map a ``numerical`` simplex exit to
``undecided``, so ``solve_lp_std`` returned ``error`` and its caller lost the bound. The
fixture is the McCormick root LP of ex14_1_7 (MINLPLib, optimum 0) that the verified
free-column route (#1869/#1941) hands to ``solve_lp_std``: |b| up to 2e11, coefficients
9e-11..2e8. HiGHS solves it to optimal 0; the cold Rust solve breaks down. Under the
graduated ``DISCOPT_HIGHS_CONFIG`` the root LP of that instance is exactly this one, so
the solve lost its root certificate (optimal in 3 nodes -> uncertified after 363 at 45 s).

The retry is exact (power-of-two factors: ``x = C x'``, ``y = R y'``) and nothing it
returns is trusted: ``solve_lp_std`` re-verifies the point, the NS bound and any ray
against the original form. The forced-breakdown tests below pin the mapping on LPs
whose answer is known, including an infeasible one (Farkas ray mapped by ``R``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

H = pytest.importorskip("discopt.solvers.lp_milp_highs")
E = pytest.importorskip("discopt.solvers.lp_std_engine")

_FIX = Path(__file__).parent / "data" / "ex14_1_7_root_mccormick_lp.npz"


def _fixture_stdform():
    """The fixture as ``_verified_free_column_bound`` builds it (slacks in [0, inf))."""
    from discopt._relax.milp_relaxation import _RELAX_NUMERIC_CAP

    d = np.load(_FIX)
    A = sp.csr_matrix((d["data"], d["indices"], d["indptr"]), shape=tuple(d["shape"]))
    m, _ = A.shape
    lo, hi = d["lo"].copy(), d["hi"].copy()
    lo[lo <= -_RELAX_NUMERIC_CAP] = -np.inf
    hi[hi >= _RELAX_NUMERIC_CAP] = np.inf
    return H.StdForm.from_arrays(
        np.concatenate([d["c"], np.zeros(m)]),
        sp.hstack([A, sp.identity(m, format="csr")]).tocsc(),
        d["b"],
        np.concatenate([lo, np.zeros(m)]),
        np.concatenate([hi, np.full(m, np.inf)]),
        obj_const=float(d["off"][0]),
    )


def test_badly_scaled_relaxation_lp_is_certified(monkeypatch):
    monkeypatch.delenv("DISCOPT_LP_STD_EQUIL_RETRY", raising=False)
    out = H.solve_lp_std(_fixture_stdform(), engine="rust")
    assert out.status == "optimal" and out.gap_certified, out.message
    # HiGHS: optimal 0. The certified bound is a valid lower bound near it.
    assert out.bound is not None and -1e-5 <= out.bound <= 1e-9
    assert out.highs_status.endswith("+equilibrated")


def test_opt_out_restores_the_previous_error(monkeypatch):
    # The pre-fix behaviour, so the test above is known to fail without the retry.
    monkeypatch.setenv("DISCOPT_LP_STD_EQUIL_RETRY", "0")
    out = H.solve_lp_std(_fixture_stdform(), engine="rust")
    assert out.status == "error"
    assert "could not decide between infeasible and unbounded" in out.message


def _scaled_lp(rng, m, n, feasible=True):
    """A well-posed LP ``A x = b`` with badly scaled rows and columns."""
    A = sp.random(m, n, density=0.4, random_state=rng, format="csr")
    A = A + sp.hstack([sp.identity(m), sp.csr_matrix((m, n - m))])
    rs = 10.0 ** rng.uniform(-6, 6, m)
    cs = 10.0 ** rng.uniform(-4, 4, n)
    A = (sp.diags(rs) @ A @ sp.diags(cs)).tocsc()
    x0 = rng.uniform(0.2, 0.8, n) / cs
    b = A @ x0
    if not feasible:
        b = b.copy()
        b[0] = -abs(b[0]) - 1.0 * rs[0]  # row 0 has nonnegative entries on x >= 0
        A = A.tolil()
        A[0, :] = abs(A[0, :].toarray())
        A = A.tocsc()
    c = rng.normal(size=n) * cs
    return H.StdForm.from_arrays(c, A, b, np.zeros(n), 1.0 / cs, obj_const=0.0)


@pytest.mark.parametrize("feasible", [True, False])
def test_forced_breakdown_maps_back_exactly(monkeypatch, feasible):
    """Force the first (unscaled) solve to report a breakdown: the retry's answer must be
    the direct solve's verdict, re-certified on the original form."""
    rng = np.random.default_rng(7 if feasible else 11)
    sf = _scaled_lp(rng, 12, 30, feasible=feasible)
    monkeypatch.delenv("DISCOPT_LP_STD_EQUIL_RETRY", raising=False)
    direct = H.solve_lp_std(sf, engine="rust")
    assert direct.status == ("optimal" if feasible else "infeasible"), direct.message

    real = E.RustEngine._solve
    calls = []

    def first_breaks(self, c, A, b, xl, xu, time_limit):
        calls.append(1)
        if len(calls) == 1:
            return E.LpRaw("undecided", "rust:numerical", iters=1.0)
        return real(self, c, A, b, xl, xu, time_limit)

    monkeypatch.setattr(E.RustEngine, "_solve", first_breaks)
    out = H.solve_lp_std(sf, engine="rust")
    assert len(calls) >= 2, "the retry never ran"
    assert out.highs_status.endswith("+equilibrated")
    assert out.status == direct.status and out.gap_certified
    if feasible:
        tol = 1e-7 * max(1.0, abs(direct.objective))
        assert abs(out.objective - direct.objective) <= tol
        assert out.bound <= out.objective

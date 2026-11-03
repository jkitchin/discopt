"""#2107: the in-house engine confirms a HiGHS lazy-master claim with no SCIP.

Since #2072 a certifying HiGHS master verdict is re-derived by an independent
engine; with ``pyscipopt`` absent (the default install) that is only the in-house
Rust MILP driver. On the ``clay0204hfsg`` OA master
(``data/issue2107/clay0204hfsg_master.npz``: 164 columns, 793 rows, the final
master ``lp_nlp_bb`` held at the 60 s solve) the driver drained its tree but
reported ``bound = -inf``: the OA cut rows carry gradient noise (9 entries,
2e-14 to 6.5e-11) on columns declared ``[0, inf)``, and the #1296 tiny-entry
guard withdrew the certificate. The claim (6545.0, the MINLPLib optimum) was
lowered and the solve ran to its time limit.

The driver now measures those entries over the box the rows imply and relaxes
the negligible ones into their row's slack (``milp_driver::relax_noise_entries``),
a superset of the feasible set, so the bound it proves stays valid.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import SolveStatus, milp_highs

highspy = pytest.importorskip("highspy")

#: Raw arrays, not MPS: HiGHS ``readModel`` drops entries below its
#: ``small_matrix_value`` (1e-9), i.e. exactly the noise this test is about.
NPZ = pathlib.Path(__file__).parent / "data" / "issue2107" / "clay0204hfsg_master.npz"
#: The HiGHS lazy master's claim on this master inside ``lp_nlp_bb`` (#2107).
CLAIM = 6545.000000000002
#: SCIP 6.3's proven optimum of the master, = the MINLPLib optimum of clay0204hfsg.
TRUTH = 6545.0


@pytest.fixture(scope="module")
def master():
    z = np.load(NPZ)
    n = z["c"].shape[0]
    inf = highspy.kHighsInf
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    lb = np.where(np.isinf(z["lb"]), -inf, z["lb"])
    h.addVars(n, lb, np.where(np.isinf(z["ub"]), inf, z["ub"]))
    h.changeColsCost(n, np.arange(n), z["c"])
    kinds = [highspy.HighsVarType.kInteger, highspy.HighsVarType.kContinuous]
    h.changeColsIntegrality(
        n, np.arange(n), np.array([kinds[0 if i else 1] for i in z["integrality"]])
    )
    # The arrays are the engine's view, every row scaled by a power of two into
    # [0.5, 1) by ``_lp_as_arrays``; the master HiGHS held had larger rows. Undo an
    # exact power of two per row so HiGHS keeps the small entries (it drops
    # |a| <= small_matrix_value); ``_lp_as_arrays`` then rescales them back bit for bit.
    A = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=(z["row_lo"].size, n))
    absmin = np.array(
        [np.abs(A.data[A.indptr[i] : A.indptr[i + 1]]).min() for i in range(A.shape[0])]
    )
    up = np.ldexp(1.0, np.maximum(0, np.frexp(1e-6 / absmin)[1]))
    A = sp.diags(up).dot(A).tocsr()
    lo = np.where(np.isinf(z["row_lo"]), -inf, z["row_lo"] * up)
    hi = np.where(np.isinf(z["row_hi"]), inf, z["row_hi"] * up)
    st = h.addRows(
        lo.size, lo, hi, A.nnz, A.indptr[:-1].astype(np.int32), A.indices.astype(np.int32), A.data
    )
    assert st == highspy.HighsStatus.kOk  # kWarning would mean entries were dropped
    # The engine sees exactly the stored master, noise included (CLAUDE.md section 6).
    arrays = milp_highs._lp_as_arrays(h.getLp(), highspy)
    got = sp.vstack([arrays["A_ub"], arrays["A_eq"]]).tocsr()
    want = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=got.shape)
    assert (got != want).nnz == 0
    assert int(np.sum(np.abs(got.data) < 1e-10)) >= 9
    return h


@pytest.mark.slow
@pytest.mark.correctness
def test_inhouse_engine_alone_confirms_the_clay0204hfsg_claim(master, monkeypatch):
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    from discopt._rust import profile_counters_py, profile_reset_py

    profile_reset_py()
    status, bound, diag = milp_highs._confirm_lazy_master_independently(
        master, highspy, SolveStatus.OPTIMAL, CLAIM, 120.0, CLAIM, None, 1e-4, engines=("simplex",)
    )
    counters = profile_counters_py()
    assert diag["ran"] and diag["runs"][0]["engine"] == "simplex", diag
    # Before #2107: the tree drained, the bound came back -inf, the claim was lowered.
    assert diag["confirmed_by"] == "simplex", diag
    assert status == SolveStatus.OPTIMAL, diag
    # Soundness: never above the proven optimum; and close enough to certify.
    assert bound is not None and bound <= TRUTH + 1e-6 * abs(TRUTH), (bound, diag)
    assert bound >= TRUTH - 1e-4 * abs(TRUTH), (bound, diag)
    # CLAUDE.md section 6: the relaxation is what made the difference.
    assert counters["MilpNoiseEntriesRelaxed"] > 0, counters
    assert counters["MilpTinyEntryDecert"] == 0, counters

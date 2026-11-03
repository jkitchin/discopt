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

On the ``rsyn0820m`` master (``data/issue2107/rsyn0820m_master.npz``: 215 columns,
468 rows) the in-house root stalled far below the claim: most continuous columns are
bounded only by big-M variable-bound rows, which neither single-row MIR nor GMI uses.
The aggregated c-MIR separator (``lp/cmir.rs``, ``DISCOPT_MILP_CMIR``) closes it.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import SolveStatus, milp_highs

highspy = pytest.importorskip("highspy")

#: Raw arrays, not MPS: HiGHS ``readModel`` drops entries below its
#: ``small_matrix_value`` (1e-9), i.e. exactly the noise these tests are about.
DATA = pathlib.Path(__file__).parent / "data" / "issue2107"
#: The HiGHS lazy master's claim on each master inside ``lp_nlp_bb`` (#2107).
CLAY_CLAIM = 6545.000000000002
#: SCIP 6.3's proven optimum of the master, = the MINLPLib optimum of clay0204hfsg.
CLAY_TRUTH = 6545.0
RSYN_CLAIM = -1150.300535412604
#: SCIP 6.3's proven optimum of the rsyn0820m master (internal minimization sense).
RSYN_TRUTH = -1150.3005354126


def _load(name: str, min_tiny: int):
    z = np.load(DATA / f"{name}_master.npz")
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
    got = sp.vstack([M for M in (arrays["A_ub"], arrays["A_eq"]) if M is not None]).tocsr()
    want = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=got.shape)
    assert (got != want).nnz == 0
    assert int(np.sum(np.abs(got.data) < 1e-10)) >= min_tiny
    seed = z["x_seed"] if "x_seed" in z.files else None
    return h, seed


def _confirm(h, claim, seed):
    from discopt._rust import profile_counters_py, profile_reset_py

    profile_reset_py()
    status, bound, diag = milp_highs._confirm_lazy_master_independently(
        h, highspy, SolveStatus.OPTIMAL, claim, 120.0, claim, seed, 1e-4, engines=("simplex",)
    )
    return status, bound, diag, profile_counters_py()


@pytest.mark.slow
@pytest.mark.correctness
def test_inhouse_engine_alone_confirms_the_clay0204hfsg_claim(monkeypatch):
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    h, _ = _load("clay0204hfsg", min_tiny=9)
    status, bound, diag, counters = _confirm(h, CLAY_CLAIM, None)
    assert diag["ran"] and diag["runs"][0]["engine"] == "simplex", diag
    # Before #2107: the tree drained, the bound came back -inf, the claim was lowered.
    assert diag["confirmed_by"] == "simplex", diag
    assert status == SolveStatus.OPTIMAL, diag
    # Soundness: never above the proven optimum; and close enough to certify.
    assert bound is not None and bound <= CLAY_TRUTH + 1e-6 * abs(CLAY_TRUTH), (bound, diag)
    assert bound >= CLAY_TRUTH - 1e-4 * abs(CLAY_TRUTH), (bound, diag)
    # CLAUDE.md section 6: the relaxation is what made the difference.
    assert counters["MilpNoiseEntriesRelaxed"] > 0, counters
    assert counters["MilpTinyEntryDecert"] == 0, counters


@pytest.mark.slow
@pytest.mark.correctness
def test_inhouse_engine_alone_confirms_the_rsyn0820m_claim(monkeypatch):
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    h, seed = _load("rsyn0820m", min_tiny=0)
    status, bound, diag, counters = _confirm(h, RSYN_CLAIM, seed)
    assert diag["ran"] and diag["runs"][0]["engine"] == "simplex", diag
    # Before #2107 the in-house tree stopped near -1203 at any budget; the claim was
    # lowered and ``lp_nlp_bb`` lost its certificate without SCIP.
    assert diag["confirmed_by"] == "simplex", diag
    assert status == SolveStatus.OPTIMAL, diag
    tol = 1e-6 * abs(RSYN_TRUTH)
    assert bound is not None and bound <= RSYN_TRUTH + tol, (bound, diag)
    assert bound >= RSYN_TRUTH - 1e-4 * abs(RSYN_TRUTH), (bound, diag)
    # CLAUDE.md section 6: c-MIR separated cuts in this confirmation.
    assert counters["SepCmirCuts"] > 0, counters
    # Deterministic work, not wall clock (the tree finishes inside the budget either
    # way): measured 11540 nodes with c-MIR (bit-identical across repetitions) and
    # 62372 with ``DISCOPT_MILP_CMIR=0``, which needs ~8x the wall time.
    nodes = diag["runs"][0]["nodes"]
    assert nodes <= 25_000, f"{nodes} nodes: c-MIR no longer closes this master's root"

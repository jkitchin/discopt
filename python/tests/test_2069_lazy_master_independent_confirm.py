"""#2069: a certifying HiGHS lazy-master verdict is held against the in-house engine.

HiGHS can return a false ``Optimal`` on an OA master. On the
``constrained_layout_clay0203`` big-M master (``data/issue2069/clay0203_master.mps.gz``:
30 columns, 18 integer, 525 rows) HiGHS 1.10-1.15 report 41907-56725 with presolve on
or off and under two random seeds; SCIP 6.3 proves 41420.27. Inside ``lp_nlp_bb`` the
master claimed ``Optimal`` at 42098.844879 and its presolve-free #1634 cross-solve,
HiGHS again, did not refute it. Only the bound/incumbent inversion guard stood between
that and a certificate: a false bound within the gap tolerance of a suboptimal
incumbent would have certified it.

The lazy master now re-derives any verdict its caller would certify on with the
in-house Rust MILP engine (``milp_highs._confirm_lazy_master_independently``) and
publishes the lower of the two bounds; one the engine cannot reach is withdrawn.
"""

from __future__ import annotations

import gzip
import pathlib
import sys
import warnings

import numpy as np
import pytest
from discopt.solvers import SolveStatus, milp_highs

highspy = pytest.importorskip("highspy")

MPS_GZ = pathlib.Path(__file__).parent / "data" / "issue2069" / "clay0203_master.mps.gz"
#: SCIP 6.3.0's proven optimum of the master (issue #2069).
TRUTH = 41420.2719511956
#: The bound HiGHS's lazy master claimed for this master inside ``lp_nlp_bb``.
FALSE_CLAIM = 42098.844879
#: The certified optimum of the clay0203 MINLP (the GAMS oracle).
CLAY_OPT = 41573.2625


def _slack(v: float) -> float:
    return 1e-6 + 1e-6 * abs(v)


@pytest.fixture(scope="module")
def mps(tmp_path_factory) -> str:
    out = tmp_path_factory.mktemp("issue2069") / "clay0203_master.mps"
    out.write_bytes(gzip.decompress(MPS_GZ.read_bytes()))
    return str(out)


def _highs(path: str):
    h = highspy.Highs()
    h.setOptionValue("output_flag", False)
    assert h.readModel(path) == highspy.HighsStatus.kOk
    return h


def _confirm(h, status, bound, *, time_left=60.0, objective=None, x_seed=None, engines=None):
    return milp_highs._confirm_lazy_master_independently(
        h, highspy, status, bound, time_left, objective, x_seed, 1e-4, engines=engines
    )


ENGINES = ["simplex"] + (["scip"] if milp_highs._scip_available() else [])


@pytest.mark.parametrize("engine", ENGINES)
def test_a_false_claim_is_lowered_to_the_true_master_optimum(mps, engine):
    status, bound, diag = _confirm(
        _highs(mps), SolveStatus.OPTIMAL, FALSE_CLAIM, objective=FALSE_CLAIM, engines=(engine,)
    )
    assert diag["ran"], diag  # CLAUDE.md section 6: the engine actually ran
    assert bound is not None and bound <= TRUTH + _slack(TRUTH), (bound, diag)
    # The engine reaches the optimum, so the certificate is lost for the right reason
    # (the gap to the master's incumbent is open), not because nothing was proved.
    assert bound >= TRUTH - 1e-4 * abs(TRUTH), (bound, diag)
    assert diag["bound_lowered"] > 600.0
    assert status == SolveStatus.ITERATION_LIMIT


@pytest.mark.parametrize("engine", ENGINES)
def test_a_true_claim_is_confirmed_and_keeps_optimal(mps, engine):
    claim = TRUTH - 1e-7
    status, bound, diag = _confirm(
        _highs(mps), SolveStatus.OPTIMAL, claim, objective=TRUTH, engines=(engine,)
    )
    assert diag["ran"] and diag["confirmed_by"] == engine and "withdrawn" not in diag, diag
    assert status == SolveStatus.OPTIMAL
    assert bound is not None and bound <= claim
    assert bound >= TRUTH - 1e-4 * abs(TRUTH)


def test_no_budget_withdraws_the_claim(mps):
    status, bound, diag = _confirm(_highs(mps), SolveStatus.OPTIMAL, TRUTH, time_left=0.0)
    assert bound is None and status == SolveStatus.ITERATION_LIMIT and not diag["ran"]
    status, bound, _ = _confirm(_highs(mps), SolveStatus.INFEASIBLE, None, time_left=0.0)
    assert status == SolveStatus.ERROR and bound is None


def test_an_infeasibility_claim_on_a_feasible_master_is_withdrawn(mps):
    status, bound, diag = _confirm(_highs(mps), SolveStatus.INFEASIBLE, None)
    assert diag["ran"] and diag["confirmed_by"] is None, diag
    assert all(r["status"] != "infeasible" for r in diag["runs"]), diag
    assert status == SolveStatus.ERROR and bound is None


def test_an_engine_with_no_bound_hands_over_to_the_next(mps, monkeypatch):
    """The in-house engine cannot bound every master (``m6``: 62.83 after 38 594
    nodes against a claim SCIP confirms in 0.3 s), so the chain continues to SCIP;
    without another engine the claim is withdrawn, never kept on HiGHS's word."""
    monkeypatch.setattr(
        milp_highs,
        "_run_simplex_engine",
        lambda arrays, budget, gap, seed: (SolveStatus.ITERATION_LIMIT, None, 7),
    )
    status, bound, diag = _confirm(
        _highs(mps), SolveStatus.OPTIMAL, TRUTH, objective=TRUTH, engines=("simplex",)
    )
    assert bound is None and status == SolveStatus.ITERATION_LIMIT, diag
    assert diag["withdrawn"] == "no engine reached a bound"
    pytest.importorskip("pyscipopt")
    status, bound, diag = _confirm(
        _highs(mps), SolveStatus.OPTIMAL, TRUTH, objective=TRUTH, engines=("simplex", "scip")
    )
    assert [r["engine"] for r in diag["runs"]] == ["simplex", "scip"], diag
    assert diag["confirmed_by"] == "scip" and status == SolveStatus.OPTIMAL, diag
    assert bound is not None and TRUTH - 1e-4 * TRUTH <= bound <= TRUTH + _slack(TRUTH)


def test_the_first_engine_that_confirms_ends_the_chain(mps):
    status, bound, diag = _confirm(
        _highs(mps), SolveStatus.OPTIMAL, TRUTH, objective=TRUTH, engines=("simplex", "scip")
    )
    assert [r["engine"] for r in diag["runs"]] == ["simplex"], diag
    assert diag["confirmed_by"] == "simplex" and status == SolveStatus.OPTIMAL


def test_engine_arrays_are_the_master_rows_scaled_by_powers_of_two(mps):
    """Row scaling is by exact powers of two (no rounding), and opened column sides
    are only discopt's near-infinite box: the engine sees the master or a relaxation
    of it, never a tighter model."""
    h = _highs(mps)
    lp = h.getLp()
    a = milp_highs._lp_as_arrays(lp, highspy)
    for M in (a["A_ub"], a["A_eq"]):
        if M is None:
            continue
        for i in range(M.shape[0]):
            row = np.abs(M.data[M.indptr[i] : M.indptr[i + 1]])
            if row.size:
                assert 0.5 <= row.max() < 1.0
    # Every scaled inequality row is an original row times 2^k: the ratio is exact.
    import scipy.sparse as sp

    A = sp.csc_matrix(
        (lp.a_matrix_.value_, lp.a_matrix_.index_, lp.a_matrix_.start_),
        shape=(lp.num_row_, lp.num_col_),
    ).tocsr()
    rl, ru = np.asarray(lp.row_lower_), np.asarray(lp.row_upper_)
    up = (ru < highspy.kHighsInf) & (rl != ru)  # equality rows go to ``A_eq``
    assert up.any()
    orig = A[up].toarray()
    scaled = a["A_ub"][: int(up.sum())].toarray()
    nz = orig != 0
    ratio = scaled[nz] / orig[nz]
    mant, _ = np.frexp(ratio)
    assert np.all(mant == 0.5), "a row scale is not an exact power of two"
    opened = [j for j, (lo, hi) in enumerate(a["bounds"]) if lo is None or hi is None]
    for j in opened:
        lo, hi = lp.col_lower_[j], lp.col_upper_[j]
        assert (a["bounds"][j][0] is not None or lo <= -1e15) and (
            a["bounds"][j][1] is not None or hi >= 1e15
        )


def _lazy(mps_path: str, confirm):
    h = _highs(mps_path)
    a = milp_highs._lp_as_arrays(h.getLp(), highspy)
    a.pop("offset")
    return milp_highs.solve_milp_with_lazy_cuts(
        **a, time_limit=60, lazy_callback=lambda x: None, confirm_bound_if=confirm
    )


def test_lazy_master_confirms_only_when_asked(mps):
    r = _lazy(mps, None)
    assert r.callback_stats["independent_confirmation"] == {"ran": False, "requested": False}
    asked: list[float] = []
    r = _lazy(mps, lambda b: bool(asked.append(b)) or True)
    diag = r.callback_stats["independent_confirmation"]
    assert asked and diag["requested"] and diag["ran"], diag
    assert r.bound is not None and r.bound <= TRUTH + _slack(TRUTH)


def test_lp_nlp_bb_certificate_is_independently_confirmed():
    """``tls2`` with rows scaled by 10^U(-6,6) certifies on the HiGHS lazy master, and
    the certificate now passes through the engine: the scaled rows are what the exact
    power-of-two row scaling exists for (unscaled, the engine had no bound after 444
    nodes)."""
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from discopt.solvers.mip_nlp import solve_mip_nlp
    from test_1537_row_scaling import _per_row_scaled

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = solve_mip_nlp(
            _per_row_scaled("tls2.nl", 6.0), method="lp_nlp_bb", milp_solver="highs", time_limit=60
        )
    assert r.status == "optimal" and r.gap_certified, (r.status, r.objective, r.bound)
    assert r.objective == pytest.approx(5.3, rel=1e-6)
    stats = r.solver_stats or {}
    assert stats.get("oa/independent_confirm_ran") == 1.0, stats
    assert stats.get("oa/independent_confirm_withdrawn") == 0.0, stats


def test_the_opt_out_restores_the_highs_only_certificate(monkeypatch):
    from discopt.solvers import oa

    monkeypatch.setenv("DISCOPT_LP_NLP_BB_INDEPENDENT_CONFIRM", "0")
    assert not oa._lp_nlp_bb_independent_confirm_enabled()
    monkeypatch.delenv("DISCOPT_LP_NLP_BB_INDEPENDENT_CONFIRM")
    assert oa._lp_nlp_bb_independent_confirm_enabled()


@pytest.mark.slow
@pytest.mark.correctness
def test_clay0203_never_certifies_above_the_optimum():
    """End to end: the instance the false master bound came from. The master's
    42098.84 is lowered to 41420.27 by the engine instead of tripping the inversion
    guard, and the solve still certifies the true optimum (through the fallback)."""
    pytest.importorskip("pyomo")
    bench = pathlib.Path(__file__).resolve().parents[2] / "discopt_benchmarks"
    sys.path.insert(0, str(bench))
    from benchmarks import pyomo_gdp_examples as px
    from discopt.pyomo.gdp import convert_pyomo_gdp

    conv = convert_pyomo_gdp(px.builder("constrained_layout_clay0203")())
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = conv.model.solve(gdp_method="big-m", time_limit=240)
    assert r.objective is not None and r.objective >= CLAY_OPT - 1e-2
    if r.bound is not None:
        assert r.bound <= CLAY_OPT + 1e-2, (r.bound, r.objective)
    if r.gap_certified:
        assert r.objective == pytest.approx(CLAY_OPT, abs=1e-2)

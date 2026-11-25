"""#2332: one HiGHS presolve-off ``kInfeasible`` is not an oracle.

HiGHS 1.11.0-1.15.1 return ``kInfeasible`` with presolve off on feasible MILPs that
presolve on solves (fixed in 1.16.0). Two consequences, both tested here:

* The HiGHS MILP route's #1634 cross-check is a second opinion only if it runs the
  other configuration. Under ``solve(presolve=False)`` (#1751) the primary is
  presolve-off, so the cross-solve now runs presolve ON, and #1655's exemption of
  the integer-data class (an argument about presolve) no longer skips it.
* Test helpers decide "is this subsystem infeasible?" with :mod:`_highs_oracle`,
  which needs a verified point or both presolve settings agreeing.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
from _highs_oracle import ISSUE_2332, milp_verdict, satisfies

highspy = pytest.importorskip("highspy")
from discopt.solvers import lp_milp_highs as H  # noqa: E402

pytestmark = pytest.mark.requires_highs

#: Found by a random search over the IIS sweep generator (scratch hunt, seed 1,
#: instance 29614): HiGHS 1.15.1 presolve off says ``kInfeasible``, presolve on
#: returns a point that verifies. Before #2332 the route under ``presolve=False``
#: ran both solves presolve-off and lost the certificate (uncertified "feasible").
SWEEP_29614 = {
    "A": np.array(
        [
            [0.0, 0.0, 0.5, 2.0, -1.5, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 3.0],
            [0.5, 0.0, 0.0, -2.0, 0.0, -1.5],
        ]
    ),
    "row_lo": np.array([-2.5, -1e20, -4.0]),
    "row_hi": np.array([3.0, 0.0, -2.0]),
    "lb": np.array([-2.0, 0.0, -2.0, -2.0, -2.0, -2.0]),
    "ub": np.array([1.0, 2.5, 3.0, 1.0, 2.5, 1e20]),
    "ints": (2, 3, 4),
}


def _std_form(inst, c=None) -> H.StdForm:
    """``row_lo <= A x <= row_hi`` as the route's standard form ``[A, -I] (x, s) = 0``."""
    A = np.asarray(inst["A"], dtype=np.float64)
    m, n = A.shape
    full = sp.hstack([sp.csc_matrix(A), -sp.identity(m, format="csc")], format="csc")
    lo = np.where(inst["row_lo"] <= -1e20, -H.INF, inst["row_lo"])
    hi = np.where(inst["row_hi"] >= 1e20, H.INF, inst["row_hi"])
    cost = np.zeros(n + m) if c is None else np.concatenate([c, np.zeros(m)])
    return H.StdForm.from_arrays(
        cost,
        full,
        np.zeros(m),
        np.concatenate([inst["lb"], lo]),
        np.concatenate([np.where(inst["ub"] >= 1e20, H.INF, inst["ub"]), hi]),
        0.0,
        int_idx=list(inst["ints"]),
    )


def _solve(sf, presolve_off: bool):
    pairs = (("presolve", "off"),) if presolve_off else ()
    with H.user_options(pairs):
        return H.solve_milp_std(sf, time_limit=30.0, gap_tolerance=1e-4, max_nodes=10**6)


def _spy_milp_presolve(monkeypatch) -> list[str]:
    """The effective ``presolve`` of every HiGHS MILP instance the route builds (the
    last write wins: base, then user options, then the solve's own)."""
    seen: list[str] = []
    real = H._new_highs

    def spy(highspy_mod, opts):
        opts = list(opts)
        if any(k == "mip_rel_gap" for k, _ in opts):  # a MILP solve, not an LP check
            val = "choose"
            for k, v in [*H._USER_OPTIONS.get(), *opts]:
                if k == "presolve":
                    val = v
            seen.append(val)
        return real(highspy_mod, opts)

    monkeypatch.setattr(H, "_new_highs", spy)
    return seen


# ---------------------------------------------------------------- the test oracle


@pytest.mark.smoke
def test_issue_witness_is_exactly_feasible():
    inst = ISSUE_2332
    args = (inst["A"], inst["row_lo"], inst["row_hi"], inst["lb"], inst["ub"], inst["ints"])
    assert satisfies(*args, inst["witness"], tol=0.0)


@pytest.mark.smoke
@pytest.mark.parametrize("inst", [ISSUE_2332, SWEEP_29614], ids=["issue", "sweep29614"])
def test_oracle_never_calls_a_feasible_model_infeasible(inst):
    """A single presolve-off solve says ``kInfeasible`` here on HiGHS <= 1.15.1; the
    oracle answers ``"feasible"`` from the verified presolve-on point."""
    args = (inst["A"], inst["row_lo"], inst["row_hi"], inst["lb"], inst["ub"], inst["ints"])
    assert milp_verdict(*args) == "feasible"


@pytest.mark.smoke
def test_oracle_needs_both_settings_to_call_a_model_infeasible(monkeypatch):
    """A presolve-off ``kInfeasible`` that presolve on does not repeat is ``None``,
    not ``"infeasible"`` (here presolve on is made to hit its time limit)."""
    inst = ISSUE_2332
    args = (inst["A"], inst["row_lo"], inst["row_hi"], inst["lb"], inst["ub"], inst["ints"])
    real_run = highspy.Highs.run

    def run(self):
        self._p = self.getOptionValue("presolve")[1]
        return real_run(self)

    def status(self):
        if self._p == "on":
            return highspy.HighsModelStatus.kTimeLimit
        return highspy.HighsModelStatus.kInfeasible  # the #2332 false label

    monkeypatch.setattr(highspy.Highs, "run", run)
    monkeypatch.setattr(highspy.Highs, "getModelStatus", status)
    assert milp_verdict(*args) is None


# ------------------------------------------------- the route's #1634 cross-check


@pytest.mark.smoke
@pytest.mark.parametrize("presolve_off", [False, True])
def test_cross_solve_runs_the_configuration_the_primary_did_not(monkeypatch, presolve_off):
    seen = _spy_milp_presolve(monkeypatch)
    out = _solve(_std_form(SWEEP_29614), presolve_off)
    assert out.stats.get("milp/presolve_cross_check_ran") == 1.0
    assert len(seen) >= 2, seen
    assert "off" in seen and any(v != "off" for v in seen), seen
    assert ("milp/presolve_cross_presolve_on" in out.stats) == presolve_off


@pytest.mark.smoke
def test_presolve_off_solve_is_certified_by_a_presolve_on_cross_check():
    """Before #2332 both solves ran presolve-off and the route lost the certificate
    on this feasible model (measured on HiGHS 1.15.1: uncertified "feasible")."""
    out = _solve(_std_form(SWEEP_29614), presolve_off=True)
    assert out.status == "optimal" and out.gap_certified, (out.status, out.message)
    assert out.stats.get("milp/presolve_cross_presolve_on") == 1.0
    inst = SWEEP_29614
    x = np.asarray(out.x)[: inst["A"].shape[1]]
    assert satisfies(
        inst["A"], inst["row_lo"], inst["row_hi"], inst["lb"], inst["ub"], inst["ints"], x, 1e-6
    )


def _integer_knapsack():
    """Integer data throughout: :func:`H.presolve_exact_class` holds."""
    inst = {
        "A": np.array([[3.0, 5.0, 4.0, 2.0], [1.0, 1.0, 1.0, 1.0]]),
        "row_lo": np.array([-1e20, 1.0]),
        "row_hi": np.array([9.0, 1e20]),
        "lb": np.zeros(4),
        "ub": np.full(4, 3.0),
        "ints": (0, 1, 2, 3),
    }
    return _std_form(inst, c=-np.array([4.0, 7.0, 5.0, 3.0]))


@pytest.mark.smoke
def test_integer_data_class_is_still_exempt_with_presolve_on():
    sf = _integer_knapsack()
    assert H.presolve_exact_class(sf)
    out = _solve(sf, presolve_off=False)
    assert out.status == "optimal" and out.gap_certified
    assert "milp/presolve_cross_check_ran" not in out.stats


@pytest.mark.smoke
def test_integer_data_class_is_cross_checked_when_the_primary_ran_presolve_off(monkeypatch):
    """#1655 exempts the class because HiGHS PRESOLVE is exact on it; a presolve-off
    primary is not covered by that argument, so its verdict gets a presolve-on check."""
    seen = _spy_milp_presolve(monkeypatch)
    out = _solve(_integer_knapsack(), presolve_off=True)
    assert out.status == "optimal" and out.gap_certified
    assert out.stats.get("milp/presolve_cross_check_ran") == 1.0
    assert out.stats.get("milp/presolve_cross_presolve_on") == 1.0
    assert "off" in seen and "choose" in seen, seen

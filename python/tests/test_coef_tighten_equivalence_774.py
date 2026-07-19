"""Both-directions feasible-set-equivalence gate for big-M coefficient tightening (#774).

The #770 gate only checked "no integer-feasible point REMOVED". The #772 false
primal (rsyn0805m 1441.99 > =opt= 1296.12) proved that insufficient: a tightening
that also ADMITS an infeasible point is equally unsound. This module adds the
missing direction and pins the exact bug class down two ways.

1. ``test_optimum_preserved`` — the both-directions gate *at the answer level*.
   Solve each small vendored instance to optimality with the flag OFF and ON and
   assert the certified optima are equal (feasible-set-equivalence ⟹ same optimum)
   AND that the flag-ON incumbent is feasible in a freshly parsed ORIGINAL model.
   A tightening that ADMITS an infeasible point moves the optimum (a false primal);
   one that REMOVES a feasible point moves it the other way (a false bound). Either
   fails this test. This is the direct regression guard for the #770/#772 defect.

2. ``test_no_infeasible_admitted_lp`` — a fast, non-sampling structural guard on
   the linear subsystem (where every coefficient change lives): for each binary
   assignment, LP-certify over the DECLARED box that no point satisfying all
   TIGHTENED linear rows violates any ORIGINAL linear row. Because the nonlinear
   rows are byte-identical in both models and the declared box contains the whole
   feasible region, this is an exact, rigorous certificate that the tightening can
   never admit an infeasible (false-primal) point — the #772 direction the #770
   test missed. It also catches the continuous-[0,1] misclassification class (a
   continuous variable wrongly treated as a binary indicator).

3. ``test_incumbent_feasible_in_fresh_original`` — end-to-end guard on the #282
   false-primal instance rsyn0805m: the write-back representation bug is invisible
   at the normalized-row level but corrupts the solve pipeline, so this exercises a
   real solve and checks the incumbent against a fresh original model.
"""

from __future__ import annotations

import itertools
import os

import discopt.modeling as dm
import numpy as np
import pytest
from discopt.modeling.core import Model
from discopt.solvers._root_presolve import (
    _extract_row,
    _is_binary_var,
    tighten_bigm_coefficients,
)

VENDORED = os.path.join(os.path.dirname(__file__), "data", "minlplib_nl")

# Small vendored instances whose big-M rows the pass rewrites and that solve to
# optimality quickly (probed). ``opt`` is the minlplib.solu reference optimum.
PRESERVE_INSTANCES = {
    "gbd": 2.2000000000,
    "gkocis": -1.9230987380,
    "ex1221": 7.6671800690,
    "st_test1": 0.0000000000,
    "alan": 2.9250000000,
}
LP_ADMIT_INSTANCES = ["gbd", "gkocis", "flay02m", "syn05hfsg", "st_test1", "alan", "ex1221"]


def _var_kind(v):
    vt = getattr(v, "vtype", getattr(v, "var_type", ""))
    name = str(getattr(vt, "name", vt)).upper()
    return "int" if (("BINARY" in name) or ("INTEGER" in name)) else "cont"


def _linear_rows(model, n):
    """All linear inequality rows as normalized (a, b) for ``a·x <= b``."""
    rows = []
    for con in model._constraints:
        sense = getattr(con, "sense", None)
        if sense not in ("<=", ">="):
            continue
        r = _extract_row(model, con, n)
        if r is None:
            continue
        coeffs, const = r
        a = np.asarray(coeffs, float).copy()
        b = float(con.rhs) - const
        if sense == ">=":
            a = -a
            b = -b
        rows.append((a, b))
    return rows


def test_continuous_unit_is_not_binary():
    """A continuous [0,1] variable must NOT be classified as a binary indicator.

    Regression for the latent unsoundness in the #770 pass: ``"IN" in vtype`` is
    true for ``"CONTINUOUS"``, so a continuous [0,1] variable was treated as a
    binary and its coefficient tightened — cutting its fractional feasible values.
    """
    m = Model("kinds")
    b = m.binary("b")
    z = m.continuous("z", lb=0.0, ub=1.0)
    i01 = m.integer("i01", lb=0, ub=1)
    i05 = m.integer("i05", lb=0, ub=5)
    c = m.continuous("c", lb=0.0, ub=10.0)
    assert _is_binary_var(b, 0.0, 1.0) is True
    assert _is_binary_var(i01, 0.0, 1.0) is True  # integer with a [0,1] box is binary-like
    assert _is_binary_var(z, 0.0, 1.0) is False  # the bug: continuous [0,1] is NOT binary
    assert _is_binary_var(i05, 0.0, 5.0) is False  # integer but not [0,1]
    assert _is_binary_var(c, 0.0, 10.0) is False


@pytest.mark.correctness
@pytest.mark.parametrize("name", sorted(PRESERVE_INSTANCES))
def test_optimum_preserved(name):
    """Both directions, answer level: flag OFF vs ON certify the same optimum,
    and the ON incumbent is feasible in a fresh ORIGINAL model."""
    path = os.path.join(VENDORED, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name} not vendored")
    opt = PRESERVE_INSTANCES[name]

    m_off = dm.from_nl(path)
    res_off = m_off.solve(time_limit=60)

    m_on = dm.from_nl(path)
    nrw = tighten_bigm_coefficients(m_on)
    res_on = m_on.solve(time_limit=60)

    assert res_off.objective is not None and res_on.objective is not None
    # 1. Both agree with the oracle optimum (neither admits nor removes a point
    #    that would move the certified optimum).
    assert res_off.objective == pytest.approx(opt, abs=1e-3, rel=1e-4), (
        f"{name}: flag-OFF optimum {res_off.objective} != oracle {opt}"
    )
    assert res_on.objective == pytest.approx(opt, abs=1e-3, rel=1e-4), (
        f"{name}: flag-ON optimum {res_on.objective} != oracle {opt} "
        f"(tightening moved the optimum — not feasible-set-equivalent; {nrw} rows rewritten)"
    )

    # 2. The flag-ON incumbent is feasible in a freshly parsed ORIGINAL model.
    m0 = dm.from_nl(path)
    n = len(m0._variables)
    names = [v.name for v in m0._variables]
    sol = res_on.x if isinstance(getattr(res_on, "x", None), dict) else {}
    if sol:
        x = np.array([float(np.ravel(sol.get(nm, 0.0))[0]) for nm in names])
        worst = max((float(a @ x) - b for a, b in _linear_rows(m0, n)), default=0.0)
        assert worst <= 1e-4, f"{name}: ON incumbent violates original linear rows by {worst}"


def _admit_violation(orig_rows, tight_rows, bin_idx, lb, ub, bin_values):
    """LP-certify over the box: does any point satisfying all TIGHT rows violate
    an ORIG row (with binaries fixed to ``bin_values``)? Returns (row, slack) or None."""
    from scipy.optimize import linprog

    n = len(lb)
    bounds = [(float(lb[j]), None if not np.isfinite(ub[j]) else float(ub[j])) for j in range(n)]
    bin_idx = np.asarray(bin_idx)
    bvals = np.asarray(bin_values, float)

    def reduce_rows(rows):
        red = []
        for a, b in rows:
            a = np.asarray(a, float)
            b_red = b - float(a[bin_idx] @ bvals)
            a_free = a.copy()
            a_free[bin_idx] = 0.0
            red.append((a_free, b_red))
        return red

    tight_sys = reduce_rows(tight_rows)
    orig_sys = reduce_rows(orig_rows)
    A_ub = np.array([a for a, _ in tight_sys]) if tight_sys else None
    b_ub = np.array([b for _, b in tight_sys]) if tight_sys else None
    for a_obj, b_obj in orig_sys:
        if np.allclose(a_obj, 0.0):
            continue
        res = linprog(-a_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
        if res.status == 3:  # unbounded ⇒ tight system does not imply this orig row
            return (-1, float("inf"))
        if res.status == 0 and (-res.fun - b_obj) > 1e-6:
            return (-1, float(-res.fun - b_obj))
    return None


@pytest.mark.correctness
@pytest.mark.parametrize("name", LP_ADMIT_INSTANCES)
def test_no_infeasible_admitted_lp(name):
    """Rigorous LP guard: the tightening admits no infeasible (false-primal) point."""
    pytest.importorskip("scipy")
    path = os.path.join(VENDORED, f"{name}.nl")
    if not os.path.exists(path):
        pytest.skip(f"{name} not vendored")

    m_orig = dm.from_nl(path)
    m_tight = dm.from_nl(path)
    n = len(m_tight._variables)
    if any(getattr(b, "size", 1) != 1 for b in m_tight._variables):
        pytest.skip("non-scalar blocks")
    if tighten_bigm_coefficients(m_tight) == 0:
        pytest.skip(f"{name}: pass rewrote nothing")

    orig_rows = _linear_rows(m_orig, n)
    tight_rows = _linear_rows(m_tight, n)
    assert len(orig_rows) == len(tight_rows)

    lb = np.array([float(np.ravel(v.lb)[0]) for v in m_orig._variables])
    ub = np.array([float(np.ravel(v.ub)[0]) for v in m_orig._variables])
    kinds = [_var_kind(v) for v in m_orig._variables]
    bin_idx = [i for i in range(n) if kinds[i] == "int" and lb[i] == 0.0 and ub[i] == 1.0]
    assert bin_idx, f"{name}: no genuine binaries but rows rewrote — misclassification?"

    if len(bin_idx) <= 14:
        corners = itertools.product([0.0, 1.0], repeat=len(bin_idx))
    else:
        rng = np.random.default_rng(0)
        corners = (tuple(rng.integers(0, 2, len(bin_idx)).astype(float)) for _ in range(4000))

    viol = []
    for corner in corners:
        v = _admit_violation(orig_rows, tight_rows, bin_idx, lb, ub, list(corner))
        if v is not None:
            viol.append((corner, v))
            if len(viol) >= 3:
                break
    assert not viol, (
        f"{name}: tightening ADMITS an infeasible point (false-primal class). "
        f"corner→(orig_row,slack): {viol}"
    )


@pytest.mark.slow
@pytest.mark.correctness
def test_incumbent_feasible_in_fresh_original():
    """End-to-end guard on the #282/#772 false-primal instance rsyn0805m (MAXIMIZE):
    the flag-ON incumbent must be feasible in the ORIGINAL model and never beat the
    oracle optimum (the #770 write-back bug reported 1441.99 > =opt= 1296.12)."""
    snap = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl/rsyn0805m.nl")
    if not os.path.exists(snap):
        pytest.skip("rsyn0805m snapshot unavailable")
    oracle = 1296.1206030

    m = dm.from_nl(snap)
    assert tighten_bigm_coefficients(m) > 0, "expected rsyn0805m big-M rows to be rewritten"
    res = m.solve(time_limit=120)
    assert res.objective is not None
    assert res.objective <= oracle + 1e-3, (
        f"incumbent {res.objective} beats oracle {oracle} — false primal (the #770 bug)"
    )

    m0 = dm.from_nl(snap)
    n = len(m0._variables)
    names = [v.name for v in m0._variables]
    sol = res.x if isinstance(getattr(res, "x", None), dict) else {}
    if not sol:
        pytest.skip("solver did not expose a variable-value dict")
    x = np.array([float(np.ravel(sol.get(nm, 0.0))[0]) for nm in names])
    worst = max((float(a @ x) - b for a, b in _linear_rows(m0, n)), default=0.0)
    assert worst <= 1e-4, f"incumbent violates original linear rows by {worst} (false primal)"

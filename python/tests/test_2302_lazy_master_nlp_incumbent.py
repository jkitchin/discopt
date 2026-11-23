"""#2302: the in-house LP/NLP-BB master never learned the fixed-NLP incumbent.

Its tree only knew the points its separator accepted, so after a veto whose fixed NLP
found a better incumbent it went on diving to integer points that incumbent had already
beaten: on the issue's gate-sizing model, 114 of the 119 points it offered had a master
objective at or above the running NLP incumbent (310 nodes, 119 fixed NLPs, against 36
and 7 on the HiGHS master). The separator now offers that incumbent to the tree
(``incumbent_callback`` / ``MilpLazyHook::incumbent_candidate``), which checks it against
its rows and re-separates it before adopting it.
"""

from __future__ import annotations

import math

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.solvers import SolveStatus
from discopt.solvers.milp_simplex import solve_milp_with_lazy_cuts

# ── the driver hook ──────────────────────────────────────────────────────────


def _box_knapsack():
    # min -x0 - x1 over x integer in [0, 10]^2 with x0 + x1 <= 12.
    c = np.array([-1.0, -1.0])
    a_ub = sp.csr_matrix(np.array([[1.0, 1.0]]))
    return c, a_ub, np.array([12.0]), [(0.0, 10.0), (0.0, 10.0)], np.ones(2, dtype=np.int64)


@pytest.mark.smoke
def test_offered_incumbent_is_adopted_after_a_veto(monkeypatch):
    """A separator that vetoes ``x0 + x1 > 9`` (a lazy row) and offers the verified
    point (5, 4) gets it adopted, and the search still returns the true optimum 9."""
    monkeypatch.setenv("DISCOPT_MILP_LAZY_RESTARTS", "0")  # one tree, one counter set
    c, a_ub, b_ub, bounds, integ = _box_knapsack()
    offered = {"n": 0}

    def lazy(x):
        if x[0] + x[1] > 9.0 + 1e-9:
            return [(np.array([1.0, 1.0]), 9.0)]
        return []

    def offer():
        offered["n"] += 1
        return np.array([5.0, 4.0])

    r = solve_milp_with_lazy_cuts(
        c, a_ub, b_ub, None, None, bounds, integ, lazy_callback=lazy, incumbent_callback=offer
    )
    assert offered["n"] >= 1, "the driver never asked for an offer (CLAUDE.md §6)"
    assert r.callback_stats["incumbent_offers"] >= 1
    assert r.status == SolveStatus.OPTIMAL
    assert r.objective == pytest.approx(-9.0)


@pytest.mark.smoke
def test_an_infeasible_offer_is_never_adopted(monkeypatch):
    """An offer outside the rows is dropped by the driver's check; one the separator
    rejects is not adopted. Either way the answer is the true optimum."""
    monkeypatch.setenv("DISCOPT_MILP_LAZY_RESTARTS", "0")
    c, a_ub, b_ub, bounds, integ = _box_knapsack()
    seen: list[np.ndarray] = []

    def lazy(x):
        seen.append(np.asarray(x, dtype=float).copy())
        if x[0] + x[1] > 9.0 + 1e-9:
            return [(np.array([1.0, 1.0]), 9.0)]
        return []

    for bad in (np.array([10.0, 10.0]), np.array([6.0, 6.0])):
        seen.clear()
        r = solve_milp_with_lazy_cuts(
            c,
            a_ub,
            b_ub,
            None,
            None,
            bounds,
            integ,
            lazy_callback=lazy,
            incumbent_callback=lambda bad=bad: bad,
        )
        assert r.callback_stats["incumbent_offers"] >= 1
        assert r.status == SolveStatus.OPTIMAL
        assert r.objective == pytest.approx(-9.0)
        assert r.x[0] + r.x[1] <= 9.0 + 1e-6
    # (10, 10) violates x0 + x1 <= 12 and never reaches the separator; (6, 6) does,
    # and is vetoed there.
    assert any(np.allclose(p, [6.0, 6.0]) for p in seen)


def test_offer_of_the_wrong_length_raises():
    c, a_ub, b_ub, bounds, integ = _box_knapsack()

    def lazy(x):
        return [(np.array([1.0, 1.0]), 9.0)] if x[0] + x[1] > 9.0 + 1e-9 else []

    with pytest.raises(Exception, match="structural columns"):
        solve_milp_with_lazy_cuts(
            c,
            a_ub,
            b_ub,
            None,
            None,
            bounds,
            integ,
            lazy_callback=lazy,
            incumbent_callback=lambda: np.zeros(3),
        )


# ── the issue's model ─────────────────────────────────────────────────────────

_GATE = ["INV", "NAND2", "NOR2", "NAND2", "NOR2", "INV", "NAND2"]
_C_IN = {"INV": 3.0, "NAND2": 4.0, "NOR2": 5.0}
_C_PAR = {"INV": 3.0, "NAND2": 6.0, "NOR2": 6.0}
_A_UNIT = {"INV": 1.0, "NAND2": 1.5, "NOR2": 1.8}
_FANOUT = {0: [3], 1: [3, 4], 2: [4], 3: [5, 6], 4: [6], 5: [], 6: []}
_OUTPUTS = [5, 6]


def _gate_sizing(k_max: int = 5) -> dm.Model:
    """#2302's convex MINLP: log-form GP gate sizing, sizes restricted to 2**k."""
    r_drv, c_l, a_max, n_g = 4.0, 30.0, 40.0, len(_GATE)
    fanin = {i: [j for j in _FANOUT if i in _FANOUT[j]] for i in _FANOUT}

    def delay_terms(i):
        k = 0.69 * r_drv
        terms = [(k * _C_PAR[_GATE[i]], {})]
        terms += [(k * _C_IN[_GATE[j]], {f"x{j + 1}": 1, f"x{i + 1}": -1}) for j in _FANOUT[i]]
        if i in _OUTPUTS:
            terms.append((k * c_l, {f"x{i + 1}": -1}))
        return terms

    bounds = {
        **{f"x{i + 1}": (1.0, 32.0) for i in range(n_g)},
        **{f"t{i + 1}": (1.0, 1e4) for i in range(n_g)},
        "T": (1.0, 1e4),
    }
    rows = [[(1.0, {"T": 1})]]
    for i in range(n_g):
        d = delay_terms(i)
        tail = [[]] if not fanin[i] else [[(1.0, {f"t{j + 1}": 1})] for j in fanin[i]]
        for extra in tail:
            rows.append([(c, {**ex, f"t{i + 1}": -1}) for c, ex in d + extra])
    for i in _OUTPUTS:
        rows.append([(1.0, {f"t{i + 1}": 1, "T": -1})])
    rows.append([(_A_UNIT[_GATE[i]] / a_max, {f"x{i + 1}": 1}) for i in range(n_g)])

    names = list(bounds)
    m = dm.Model("gates_library")
    k = m.integer("k", shape=(n_g,), lb=0, ub=k_max)
    u = m.continuous(
        "u",
        shape=(len(names) - n_g,),
        lb=[math.log(bounds[v][0]) for v in names[n_g:]],
        ub=[math.log(bounds[v][1]) for v in names[n_g:]],
    )
    y = {v: (math.log(2.0) * k[j] if j < n_g else u[j - n_g]) for j, v in enumerate(names)}

    def aff(c, ex):
        return dm.sum([float(a) * y[v] for v, a in ex.items()]) + math.log(c) if ex else math.log(c)

    m.minimize(aff(*rows[0][0]))
    for terms in rows[1:]:
        z = [aff(c, ex) for c, ex in terms]
        m.subject_to(z[0] <= 0 if len(z) == 1 else dm.sum([dm.exp(t) for t in z]) <= 1)
    return m


def test_gate_sizing_master_prunes_on_the_nlp_incumbent(monkeypatch):
    """Certified T = 106.26 on the in-house master, with the offer adopted and far
    fewer fixed NLPs than the 119 the tree spent without it (13 measured)."""
    monkeypatch.delenv("DISCOPT_CONVEX_ROUTE_OA_MASTER", raising=False)
    monkeypatch.delenv("DISCOPT_OA_MASTER_NLP_INCUMBENT", raising=False)
    r = _gate_sizing().solve()
    assert r.status == "optimal" and r.gap_certified
    assert math.exp(r.objective) == pytest.approx(106.26, abs=5e-3)
    tr = r.mip_nlp_trace
    assert tr["milp_backend"] == "simplex"
    cs = tr["summary"]["callback_stats"]
    assert cs["nlp_incumbent_offer_wired"] is True
    assert cs["nlp_incumbent_adopted"] >= 1, cs
    assert tr["summary"]["nlp_subproblem_count"] <= 40


def test_opt_out_wires_no_offer(monkeypatch):
    monkeypatch.delenv("DISCOPT_CONVEX_ROUTE_OA_MASTER", raising=False)
    monkeypatch.setenv("DISCOPT_OA_MASTER_NLP_INCUMBENT", "0")
    r = _gate_sizing().solve()
    assert r.status == "optimal" and r.gap_certified
    cs = r.mip_nlp_trace["summary"]["callback_stats"]
    assert cs["nlp_incumbent_offer_wired"] is False
    assert cs["nlp_incumbent_offered"] == 0
    assert cs.get("incumbent_offer_calls", 0) == 0

"""Differential soundness test for ``DISCOPT_MILP_SYM_LEX`` (verified formulation
symmetry + lexicographic-leader rows) and ``DISCOPT_MILP_CONFLICT`` (conflict analysis
in the HiGHS-design search) on the Rust MILP engine.

Both are bound-changing (CLAUDE.md §5): every arm (off, sym, conflict, both) must
reach the same verdict and optimum as HiGHS, the independent oracle, on symmetric
families where the rows and the conflicts actually fire, feasible and infeasible.
The test asserts that each mechanism fired at least once over the family (§6: prove
the probe fired) and that every solve was compared.
"""

from __future__ import annotations

import discopt._rust as _rust
import discopt.modeling as dm
import numpy as np
import pytest

pytest.importorskip("highspy")

ARMS = {
    "off": {},
    "sym": {"DISCOPT_MILP_SYM_LEX": "1"},
    "conf": {"DISCOPT_MILP_CONFLICT": "1"},
    "both": {"DISCOPT_MILP_SYM_LEX": "1", "DISCOPT_MILP_CONFLICT": "1"},
}
assert list(ARMS)[0] == "off"  # the flags-off optimum is read before the other arms


def _bin_packing(sizes, nbins, cap) -> dm.Model:
    """Identical bins: every bin permutation is a formulation symmetry."""
    n = len(sizes)
    m = dm.Model("bp")
    x = m.binary("x", shape=(n, nbins))
    y = m.binary("y", shape=(nbins,))
    for i in range(n):
        m.subject_to(sum(x[i, b] for b in range(nbins)) == 1)
    for b in range(nbins):
        m.subject_to(sum(int(sizes[i]) * x[i, b] for i in range(n)) <= cap * y[b])
    m.minimize(sum(y[b] for b in range(nbins)))
    return m


def _int_assign(demand, nmach, cap, ub) -> dm.Model:
    """General-integer copies on identical machines (radix > 2 in the lex rows)."""
    k = len(demand)
    m = dm.Model("ia")
    z = m.integer("z", shape=(k, nmach), lb=0, ub=ub)
    for t in range(k):
        m.subject_to(sum(z[t, j] for j in range(nmach)) == int(demand[t]))
    for j in range(nmach):
        m.subject_to(sum((t + 2) * z[t, j] for t in range(k)) <= cap)
    m.minimize(sum((t + 1) * z[t, j] * (t % 2 + 1) for t in range(k) for j in range(nmach)))
    return m


def _makespan(p, cnt, nmach):
    """P||Cmax with job types: z[t,j] jobs of type t on identical machine j."""
    m = dm.Model("cmax")
    k = len(p)
    z = m.integer("z", shape=(k, nmach), lb=0, ub=max(cnt))
    C = m.continuous("C", lb=0, ub=float(sum(pi * ci for pi, ci in zip(p, cnt))))
    for t in range(k):
        m.subject_to(sum(z[t, j] for j in range(nmach)) == int(cnt[t]))
    for j in range(nmach):
        m.subject_to(sum(int(p[t]) * z[t, j] for t in range(k)) <= C)
    m.minimize(C)
    return m


def _coloring(nv, edges, ncol):
    """Minimum colouring with identical colours."""
    m = dm.Model("col")
    x = m.binary("x", shape=(nv, ncol))
    w = m.binary("w", shape=(ncol,))
    for v in range(nv):
        m.subject_to(sum(x[v, c] for c in range(ncol)) == 1)
    for a, b in edges:
        for c in range(ncol):
            m.subject_to(x[a, c] + x[b, c] <= w[c])
    m.minimize(sum(w[c] for c in range(ncol)))
    return m


def _rand_graph(nv, pr, seed):
    rng = np.random.default_rng(seed)
    return [(a, b) for a in range(nv) for b in range(a + 1, nv) if rng.random() < pr]


def _cases():
    """Bin packing with more bins than needed (feasible, a symmetric search), with
    exactly the volume bound's bins (often infeasible only combinatorially: the LP is
    feasible, the packing is not), and general-integer assignments either way. The
    verdict comes from the oracle, not from the construction."""
    rng = np.random.default_rng(7)
    out = []
    for s in range(4):
        sizes = rng.integers(5, 11, 8)
        cap = 16
        lb = int(np.ceil(sizes.sum() / cap))
        out.append((f"bp{s}-slack", _bin_packing, (sizes, lb + 2, cap)))
        out.append((f"bp{s}-tight", _bin_packing, (sizes, lb, cap)))
    out.append(("ia-a", _int_assign, ([3, 4, 2], 3, 14, 3)))
    out.append(("ia-b", _int_assign, ([4, 4, 3], 3, 13, 3)))
    # Feasible optimisation with symmetry and a real search under both flags
    # (measured: cmax-a 73 -> 37 nodes, col-b 104 -> 12; conflicts learned on both).
    out.append(("cmax-a", _makespan, ([7, 11, 13, 17], [3, 3, 2, 2], 4)))
    out.append(("cmax-b", _makespan, ([5, 8, 9, 14, 19], [3, 2, 3, 2, 2], 3)))
    out.append(("col-a", _coloring, (12, _rand_graph(12, 0.4, 1), 6)))
    out.append(("col-b", _coloring, (14, _rand_graph(14, 0.35, 2), 6)))
    return out


def _solve(build, args, env, monkeypatch):
    for k in ("DISCOPT_MILP_SYM_LEX", "DISCOPT_MILP_CONFLICT"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
    _rust.profile_reset_py()
    r = build(*args).solve(milp_backend="native", milp_cuts=False, time_limit=60)
    return r, dict(_rust.profile_counters_py())


@pytest.mark.slow
@pytest.mark.correctness
def test_sym_and_conflict_arms_match_the_oracle(monkeypatch):
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    compared = 0
    fired = {"rows": 0, "conflicts": 0}
    for name, build, args in _cases():
        monkeypatch.delenv("DISCOPT_LP_ENGINE", raising=False)
        ref = build(*args).solve(milp_backend="highs")
        assert ref.status in ("optimal", "infeasible"), (name, ref.status)
        off = None
        for arm, env in ARMS.items():
            r, c = _solve(build, args, env, monkeypatch)
            assert r.status == ref.status, (name, arm, r.status, ref.status)
            if ref.status == "optimal":
                assert r.objective == pytest.approx(ref.objective, abs=1e-6), (name, arm)
                assert r.gap_certified, (name, arm)
                if arm == "off":
                    off = r.objective
                # The certified optimum of every arm, "both" included, is the
                # flags-off one: the rows and the conflicts change the search, never
                # the answer.
                assert r.objective == pytest.approx(off, abs=1e-9), (name, arm, off)
            fired["rows"] += c.get("MilpSymLexRows", 0)
            fired["conflicts"] += c.get("HighsMipSearchConflicts", 0) + c.get(
                "HighsMipSearchProofConflicts", 0
            )
            compared += 1
    assert compared == len(_cases()) * len(ARMS)
    assert fired["rows"] > 0, "no lex row was ever added: the sym arm compared nothing"
    assert fired["conflicts"] > 0, "no conflict was ever learned: the conflict arm compared nothing"


def test_flags_refuse_garbage(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
    for flag in ("DISCOPT_MILP_SYM_LEX", "DISCOPT_MILP_CONFLICT"):
        monkeypatch.setenv(flag, "maybe")
        with pytest.raises(BaseException):
            _bin_packing([3, 4, 5, 6], 3, 10).solve(milp_backend="native", milp_cuts=False)
        monkeypatch.delenv(flag)


@pytest.mark.correctness
def test_conflict_cut_pool_fires_and_matches_the_oracle(monkeypatch):
    """HiGHS's cut pool under ``DISCOPT_MILP_CONFLICT``: ``generateConflict`` cuts from
    the LP dual proofs, and the root and node cuts (rewritten over the structurals),
    propagated by the search's domains. Bound-changing (§5): every verdict and optimum
    must match HiGHS, with cuts on and off, and both pools must fire (§6)."""
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    compared = 0
    fired = {"conflict_cuts": 0, "pool_cuts": 0}
    cases = [c for c in _cases() if c[0] in ("bp0-tight", "bp1-tight", "cmax-a", "col-a")]
    assert len(cases) == 4
    for name, build, args in cases:
        monkeypatch.delenv("DISCOPT_LP_ENGINE", raising=False)
        monkeypatch.delenv("DISCOPT_MILP_CONFLICT", raising=False)
        ref = build(*args).solve(milp_backend="highs")
        assert ref.status in ("optimal", "infeasible"), (name, ref.status)
        for cuts in (False, True):
            monkeypatch.setenv("DISCOPT_MILP_CONFLICT", "1")
            monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
            _rust.profile_reset_py()
            r = build(*args).solve(milp_backend="native", milp_cuts=cuts, time_limit=60)
            c = dict(_rust.profile_counters_py())
            assert r.status == ref.status, (name, cuts, r.status, ref.status)
            if ref.status == "optimal":
                assert r.objective == pytest.approx(ref.objective, abs=1e-6), (name, cuts)
                assert r.gap_certified, (name, cuts)
            fired["conflict_cuts"] += c.get("HighsMipSearchConflictCuts", 0)
            fired["pool_cuts"] += c.get("HighsMipSearchPropagatedPoolCuts", 0)
            compared += 1
    assert compared == 2 * len(cases)
    assert fired["conflict_cuts"] > 0, "no generateConflict cut was added"
    assert fired["pool_cuts"] > 0, "no pool cut was propagated"

"""Differential soundness test for ``DISCOPT_MILP_CLIQUE_DOMAIN`` on the Rust MILP
engine: the root clique table, extended by HiGHS's presolve probing of every binary
(``HPresolve::runProbing``) and by probing the root LP's fractional binaries,
attached with the probing implications to the search's domains (HiGHS
``HighsDomain::changeBound`` -> ``cliquetable.addImplications`` and
``implications.applyImplications``).

Bound-changing (CLAUDE.md §5): every arm must reach HiGHS's verdict and optimum, the
independent oracle, with and without conflict analysis (which explains the clique
fixings through their ``Reason::cliqueTable``). The test asserts that the hook fixed
something and that presolve probing ran (§6: prove the probe fired), and that every
solve was compared.

``test_c513_tables_and_node_probing`` is the regression test for the probing schedule:
on ``c513_lex_hpre`` the extracted table holds 8 cliques (HiGHS's extraction finds
none) and HiGHS's 404 come from presolve probing, so the table must reach that size;
with conflict analysis the search has a global domain, and node separation runs
``separateImpliedBounds`` on it (probing the node LP's unprobed fractional binaries,
then the implied-bound cuts the LP point violates).
"""

from __future__ import annotations

from pathlib import Path

import discopt._rust as _rust
import discopt.modeling as dm
import numpy as np
import pytest

pytest.importorskip("highspy")

FLAGS = ("DISCOPT_MILP_CLIQUE_DOMAIN", "DISCOPT_MILP_CONFLICT")
ARMS = {
    "off": {},
    "clq": {"DISCOPT_MILP_CLIQUE_DOMAIN": "1"},
    "clq+conf": {"DISCOPT_MILP_CLIQUE_DOMAIN": "1", "DISCOPT_MILP_CONFLICT": "1"},
}


def _coloring(nv, edges, ncol):
    """Minimum colouring: every edge row is a clique on the binaries."""
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


def _knapsack_packing(n, seed):
    """Weighted set packing plus knapsack rows: cliques come from both the packing
    rows and probing the knapsacks (two heavy items exclude each other)."""
    rng = np.random.default_rng(seed)
    m = dm.Model("kp")
    x = m.binary("x", shape=(n,))
    for _ in range(n // 2):
        s = rng.choice(n, size=3, replace=False)
        m.subject_to(sum(x[int(i)] for i in s) <= 1)
    for _ in range(3):
        w = rng.integers(3, 12, n)
        m.subject_to(sum(int(w[i]) * x[i] for i in range(n)) <= int(w.sum() // 3))
    p = rng.integers(5, 20, n)
    m.maximize(sum(int(p[i]) * x[i] for i in range(n)))
    return m


def _rand_graph(nv, pr, seed):
    rng = np.random.default_rng(seed)
    return [(a, b) for a in range(nv) for b in range(a + 1, nv) if rng.random() < pr]


def _cases():
    return [
        ("col-a", _coloring, (12, _rand_graph(12, 0.4, 1), 6)),
        ("col-b", _coloring, (14, _rand_graph(14, 0.35, 2), 6)),
        ("kp-a", _knapsack_packing, (30, 3)),
        ("kp-b", _knapsack_packing, (40, 5)),
    ]


@pytest.mark.correctness
def test_clique_domain_arms_match_the_oracle(monkeypatch):
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    compared = 0
    keys = {
        "fixings": "MilpCliqueDomainFixings",
        "probed": "MilpProbeProbed",
        "presolve": "MilpProbePresolve",
        "node": "MilpProbeNode",
        "ibcuts": "MilpImpliedBoundCuts",
    }
    fired = dict.fromkeys(keys, 0)
    for name, build, args in _cases():
        for k in FLAGS + ("DISCOPT_LP_ENGINE",):
            monkeypatch.delenv(k, raising=False)
        ref = build(*args).solve(milp_backend="highs")
        assert ref.status == "optimal", (name, ref.status)
        for arm, env in ARMS.items():
            for k in FLAGS:  # both default ON since #2350: an arm turns off what it omits
                monkeypatch.setenv(k, "0")
            for k, v in env.items():
                monkeypatch.setenv(k, v)
            monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
            _rust.profile_reset_py()
            r = build(*args).solve(milp_backend="native", time_limit=60)
            c = dict(_rust.profile_counters_py())
            assert r.status == ref.status, (name, arm, r.status, ref.status)
            assert r.objective == pytest.approx(ref.objective, abs=1e-6), (name, arm)
            assert r.gap_certified, (name, arm)
            if arm == "off":
                for key in keys.values():
                    assert c.get(key, 0) == 0, (name, key, c)
            else:
                for f, key in keys.items():
                    fired[f] += c.get(key, 0)
                if arm == "clq":
                    # Node probing needs the global domain conflict analysis builds.
                    assert c.get("MilpProbeNode", 0) == 0, (name, c)
            compared += 1
    assert compared == len(_cases()) * len(ARMS)
    assert fired["fixings"] > 0, "the clique hook never fixed a literal: nothing was compared"
    assert fired["probed"] > 0, "probing never ran: the implications were never compared"
    assert fired["presolve"] > 0, "presolve probing never ran"


def test_clique_domain_flag_refuses_garbage(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
    monkeypatch.setenv("DISCOPT_MILP_CLIQUE_DOMAIN", "maybe")
    with pytest.raises(BaseException):
        _coloring(6, _rand_graph(6, 0.5, 0), 4).solve(milp_backend="native")


C513 = Path(__file__).parent / "data" / "c513_lex_hpre.mps.gz"


@pytest.mark.correctness
def test_c513_tables_and_node_probing(monkeypatch):
    """``c513_lex_hpre`` (an infeasible subsystem, 144 binaries): presolve probing fills
    the clique table to HiGHS 1.16's size (404 cliques / 808 entries after its presolve,
    measured with ``threads=1``, no symmetry), node separation probes and separates
    implied-bound cuts, and no incumbent is reported for the infeasible program."""
    for k in FLAGS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
    monkeypatch.setenv("DISCOPT_MILP_CLIQUE_DOMAIN", "1")
    monkeypatch.setenv("DISCOPT_MILP_CONFLICT", "1")
    _rust.profile_reset_py()
    r = dm.from_mps(str(C513)).solve(threads=1, max_nodes=300, deterministic=True)
    c = dict(_rust.profile_counters_py())
    assert r.objective is None, (r.status, r.objective)
    assert r.status != "optimal", r.status
    assert c.get("MilpCliqueTableCliques0", 0) < 50, c.get("MilpCliqueTableCliques0")
    assert c.get("MilpCliqueTableCliques1", 0) >= 380, c.get("MilpCliqueTableCliques1")
    assert c.get("MilpProbePresolve", 0) >= 144, c.get("MilpProbePresolve")
    assert c.get("MilpProbeNode", 0) > 0, "node separation never probed"
    assert c.get("MilpImpliedBoundCuts", 0) > 0, "no implied-bound cut was separated"


@pytest.mark.correctness
def test_defaults_on_and_zero_opts_out(monkeypatch):
    """#2350 graduated ``DISCOPT_MILP_CLIQUE_DOMAIN`` and ``DISCOPT_MILP_CONFLICT``
    jointly: with neither set, both mechanisms run (the clique hook, probing, conflict
    analysis); ``=0`` turns each off and keeps the legacy search. Same verdict and
    optimum as HiGHS in all three arms."""
    for k in FLAGS + ("DISCOPT_LP_ENGINE",):
        monkeypatch.delenv(k, raising=False)
    # kp-b reaches the search (measured: 45 hook calls, 28 probes); col-a is settled at
    # the root with nothing to probe, so it could not tell ON from OFF.
    name, build, args = next(c for c in _cases() if c[0] == "kp-b")
    ref = build(*args).solve(milp_backend="highs")
    assert ref.status == "optimal", (name, ref.status)
    monkeypatch.setenv("DISCOPT_PROFILE", "1")
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
    seen = {}
    for arm, env in {"default": {}, "both=0": dict.fromkeys(FLAGS, "0")}.items():
        for k in FLAGS:
            monkeypatch.delenv(k, raising=False)
        for k, v in env.items():
            monkeypatch.setenv(k, v)
        _rust.profile_reset_py()
        r = build(*args).solve(milp_backend="native", time_limit=60)
        c = dict(_rust.profile_counters_py())
        assert r.status == "optimal" and r.gap_certified, (arm, r.status)
        assert r.objective == pytest.approx(ref.objective, abs=1e-6), arm
        seen[arm] = (c.get("MilpCliqueDomainCalls", 0), c.get("MilpProbeProbed", 0))
    assert seen["default"][0] > 0 and seen["default"][1] > 0, seen
    assert seen["both=0"] == (0, 0), seen

"""Differential soundness test for ``DISCOPT_MILP_CLIQUE_DOMAIN`` on the Rust MILP
engine: the root clique table, extended by probing the root LP's fractional binaries,
attached with the probing implications to the search's domains (HiGHS
``HighsDomain::changeBound`` -> ``cliquetable.addImplications`` and
``implications.applyImplications``).

Bound-changing (CLAUDE.md §5): every arm must reach HiGHS's verdict and optimum, the
independent oracle, with and without conflict analysis (which explains the clique
fixings through their ``Reason::cliqueTable``). The test asserts that the hook fixed
something and that probing ran (§6: prove the probe fired), and that every solve was
compared.
"""

from __future__ import annotations

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
    fired = {"fixings": 0, "probed": 0}
    for name, build, args in _cases():
        for k in FLAGS + ("DISCOPT_LP_ENGINE",):
            monkeypatch.delenv(k, raising=False)
        ref = build(*args).solve(milp_backend="highs")
        assert ref.status == "optimal", (name, ref.status)
        for arm, env in ARMS.items():
            for k in FLAGS:
                monkeypatch.delenv(k, raising=False)
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
                assert c.get("MilpCliqueDomainFixings", 0) == 0, (name, c)
                assert c.get("MilpProbeProbed", 0) == 0, (name, c)
            else:
                fired["fixings"] += c.get("MilpCliqueDomainFixings", 0)
                fired["probed"] += c.get("MilpProbeProbed", 0)
            compared += 1
    assert compared == len(_cases()) * len(ARMS)
    assert fired["fixings"] > 0, "the clique hook never fixed a literal: nothing was compared"
    assert fired["probed"] > 0, "probing never ran: the implications were never compared"


def test_clique_domain_flag_refuses_garbage(monkeypatch):
    monkeypatch.setenv("DISCOPT_LP_ENGINE", "rust")
    monkeypatch.setenv("DISCOPT_MILP_CLIQUE_DOMAIN", "maybe")
    with pytest.raises(BaseException):
        _coloring(6, _rand_graph(6, 0.5, 0), 4).solve(milp_backend="native")

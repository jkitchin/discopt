"""#843: a real QUBO/Ising local-search primal for chimera_k64ising.

chimera_k64ising is an unconstrained binary quadratic program (1192 binary vars, 0
constraints, indefinite MAXIMIZE Ising) — discopt's dense B&B never lands a good
binary point and the #827 trivial seed only gave the useless all-zeros floor (obj 0),
so it returned NO incumbent. ``primal_heuristics.qubo_local_search`` (greedy-1opt +
tabu on the quadratic form) constructs a real feasible incumbent, injected as
``initial_point`` behind ``DISCOPT_QUBO_PRIMAL`` (default off).

Sound by construction: an unconstrained QUBO has no feasibility to violate, so any
binary point is a valid incumbent (a MAXIMIZE incumbent can never exceed the optimum).
"""

from __future__ import annotations

import os
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt._jax.primal_heuristics import is_qubo, qubo_local_search

BENCH = Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl"))
_CHIMERA = BENCH / "chimera_k64ising-01.nl"


def _small_qubo(seed: int = 1, n: int = 12):
    rng = np.random.default_rng(seed)
    m = dm.Model("qubo")
    x = [m.binary(f"x{i}") for i in range(n)]
    expr = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = float(rng.integers(-3, 4))
            if w:
                expr = expr + w * x[i] * x[j]
    m.maximize(expr)
    return m


def test_843_is_qubo_detection():
    """``is_qubo`` accepts an unconstrained binary quadratic model and rejects
    anything with constraints or non-binary variables."""
    assert is_qubo(_small_qubo()) is True
    # add a constraint -> not a QUBO
    m = _small_qubo()
    m.subject_to(sum(m._variables[0][()] for _ in range(1)) <= 5)  # any constraint
    assert is_qubo(m) is False
    # a continuous model -> not a QUBO
    mc = dm.Model("c")
    y = mc.continuous("y", lb=0.0, ub=1.0)
    mc.maximize(y * y)
    assert is_qubo(mc) is False


def test_843_qubo_local_search_finds_optimum_small():
    """On a small QUBO the local search reaches the brute-force optimum, and the value
    is sound (a MAXIMIZE incumbent never exceeds the optimum)."""
    m = _small_qubo(seed=2, n=12)
    ev = NLPEvaluator(m)
    x = qubo_local_search(m, evaluator=ev, deadline=None)
    assert x is not None and np.all(np.isin(x, [0.0, 1.0]))
    got = -float(ev.evaluate_objective(x))  # true = -internal (MAXIMIZE)
    n = ev.n_variables
    brute = max(
        -float(ev.evaluate_objective(np.array([(k >> b) & 1 for b in range(n)], float)))
        for k in range(1 << n)
    )
    assert abs(got - brute) < 1e-6, f"#843: local search {got} != optimum {brute}"


def test_843_small_qubo_full_solve_seeds_incumbent(monkeypatch):
    """End-to-end: with the flag on, the QUBO primal seeds the incumbent and the solve
    certifies the optimum — soundly (obj <= optimum)."""
    monkeypatch.setenv("DISCOPT_QUBO_PRIMAL", "1")
    m = _small_qubo(seed=3, n=10)
    ev = NLPEvaluator(m)
    nn = ev.n_variables
    brute = max(
        -float(ev.evaluate_objective(np.array([(k >> b) & 1 for b in range(nn)], float)))
        for k in range(1 << nn)
    )
    r = m.solve(time_limit=15)
    assert r.objective is not None, "#843: QUBO primal produced no incumbent"
    assert r.objective <= brute + 1e-4, f"#843: unsound incumbent {r.objective} > optimum {brute}"


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(not _CHIMERA.exists(), reason="chimera_k64ising-01.nl (corpus) absent")
def test_843_qubo_primal_incumbent_on_chimera():
    """chimera-01: the QUBO local search lands a real feasible incumbent (was NONE),
    strictly better than the trivial obj-0 floor and sound (<= optimum 24.3)."""
    m = dm.from_nl(str(_CHIMERA))
    ev = NLPEvaluator(m)
    x = qubo_local_search(m, evaluator=ev, deadline=None)
    assert x is not None and np.all(np.isin(x, [0.0, 1.0]))
    true_obj = -float(ev.evaluate_objective(x))
    assert true_obj > 1.0, f"#843: chimera incumbent {true_obj} not better than the trivial floor"
    assert true_obj <= 24.3 + 1e-3, f"#843: unsound chimera incumbent {true_obj} > optimum 24.3"

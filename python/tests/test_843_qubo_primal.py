"""#843: a real QUBO/Ising local-search primal for chimera_k64ising.

chimera_k64ising is an unconstrained binary quadratic program (1192 binary vars, 0
constraints, indefinite MAXIMIZE Ising) — discopt's dense B&B never lands a good
binary point and the #827 trivial seed only gave the useless all-zeros floor (obj 0),
so it returned NO incumbent. ``discopt.qubo_primal.qubo_local_search`` (greedy-1opt +
tabu on the quadratic form) constructs a real feasible incumbent, injected as
``initial_point``. **Default ON** (graduated per the §5 panel in
``docs/dev/data/issue843-qubo-primal-graduation.md``); ``DISCOPT_QUBO_PRIMAL=0``
opts out and restores the legacy no-seed path.

Sound by construction: an unconstrained QUBO has no feasibility to violate, so any
binary point is a valid incumbent (a MAXIMIZE incumbent can never exceed the optimum).

The graduated seed is JAX-free end to end (structural gate + algebraic quadratic
extraction + numpy search): it runs on every default solve now, so it must never
pull the JAX cold start onto the pure LP/MILP path (``test_lazy_jax_linear_path``).
The subprocess guards below pin that.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._jax.nlp_evaluator import NLPEvaluator
from discopt.qubo_primal import is_qubo, qubo_local_search

# The corpus lives canonically in Dropbox; measurement machines keep a local mirror
# (scripts/refresh_benchmark_mirror.sh) selectable via DISCOPT_MINLP_BENCH.
_BENCH_ROOTS = [
    Path(
        os.path.expanduser(
            os.environ.get("DISCOPT_MINLP_BENCH", "~/projects/discopt-minlp-benchmark")
        )
    ),
    Path(os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark")),
]
_CHIMERA = next(
    (
        r / "minlplib" / "nl" / "chimera_k64ising-01.nl"
        for r in _BENCH_ROOTS
        if (r / "minlplib" / "nl" / "chimera_k64ising-01.nl").exists()
    ),
    _BENCH_ROOTS[0] / "minlplib" / "nl" / "chimera_k64ising-01.nl",
)


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
    x = qubo_local_search(m, deadline=None)
    assert x is not None and np.all(np.isin(x, [0.0, 1.0]))
    got = -float(ev.evaluate_objective(x))  # true = -internal (MAXIMIZE)
    n = ev.n_variables
    brute = max(
        -float(ev.evaluate_objective(np.array([(k >> b) & 1 for b in range(n)], float)))
        for k in range(1 << n)
    )
    assert abs(got - brute) < 1e-6, f"#843: local search {got} != optimum {brute}"


def test_843_degree_cap_and_linear_are_not_qubo():
    """The delta bookkeeping assumes a constant Hessian: a degree>2 objective must be
    refused (None), and a pure-linear objective is left to the (JAX-free, exact) MILP
    path rather than seeded."""
    mc = dm.Model("cubic")
    x = [mc.binary(f"x{i}") for i in range(4)]
    mc.maximize(x[0] * x[1] * x[2] + x[1] * x[3])
    assert qubo_local_search(mc, deadline=None) is None

    ml = dm.Model("lin")
    y = [ml.binary(f"y{i}") for i in range(4)]
    ml.maximize(y[0] + 2 * y[1] - y[2])
    assert qubo_local_search(ml, deadline=None) is None


def test_843_small_qubo_full_solve_seeds_incumbent_by_default(monkeypatch, caplog):
    """End-to-end at the graduated DEFAULT (env unset): the QUBO primal seeds the
    incumbent and the solve certifies the optimum — soundly (obj <= optimum)."""
    monkeypatch.delenv("DISCOPT_QUBO_PRIMAL", raising=False)
    m = _small_qubo(seed=3, n=10)
    ev = NLPEvaluator(m)
    nn = ev.n_variables
    brute = max(
        -float(ev.evaluate_objective(np.array([(k >> b) & 1 for b in range(nn)], float)))
        for k in range(1 << nn)
    )
    with caplog.at_level("INFO", logger="discopt.solver"):
        r = m.solve(time_limit=15)
    assert r.objective is not None, "#843: QUBO primal produced no incumbent"
    assert r.objective <= brute + 1e-4, f"#843: unsound incumbent {r.objective} > optimum {brute}"
    assert any("QUBO local-search primal seed" in rec.message for rec in caplog.records), (
        "#843: the default-ON seed did not fire on a QUBO"
    )


def test_843_optout_restores_legacy_path(monkeypatch):
    """DISCOPT_QUBO_PRIMAL=0 is the graduation opt-out: the search is never invoked."""
    import discopt.qubo_primal as qp

    calls: list[int] = []
    real = qp.qubo_local_search

    def _spy(model, **kw):
        calls.append(1)
        return real(model, **kw)

    monkeypatch.setattr(qp, "qubo_local_search", _spy)
    monkeypatch.setenv("DISCOPT_QUBO_PRIMAL", "0")
    _small_qubo(seed=4, n=8).solve(time_limit=15)
    assert calls == [], "#843: opt-out (=0) must not invoke the QUBO primal"
    monkeypatch.setenv("DISCOPT_QUBO_PRIMAL", "1")
    _small_qubo(seed=4, n=8).solve(time_limit=15)
    assert calls == [1], "#843: opt-in (=1) must invoke the QUBO primal"


def test_843_qubo_search_is_jax_free():
    """The graduated (default-ON) seed runs on every solve, so the gate AND the
    search must not import JAX (the LP/MILP cold-start invariant). Fresh
    subprocess: build a QUBO, run the search, assert no ``jax`` in sys.modules."""
    script = textwrap.dedent(
        """
        import sys
        import numpy as np
        import discopt.modeling as dm
        from discopt.qubo_primal import qubo_local_search
        m = dm.Model('q')
        x = [m.binary(f'x{i}') for i in range(10)]
        expr = x[0] * x[1] - 2 * x[1] * x[2] + 3 * x[2] * x[3] - x[3] * x[4]
        for i in range(4, 9):
            expr = expr + (1 if i % 2 else -2) * x[i] * x[i + 1]
        m.maximize(expr)
        pt = qubo_local_search(m, deadline=None)
        assert pt is not None and np.all(np.isin(pt, [0.0, 1.0])), pt
        print('JAX_LOADED' if 'jax' in sys.modules else 'JAX_FREE')
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, f"qubo_local_search failed:\n{out.stderr}"
    assert out.stdout.strip().splitlines()[-1] == "JAX_FREE", (
        f"#843: the QUBO local search imported JAX (cold-start regression):\n{out.stdout}"
    )


def test_843_unconstrained_binary_milp_solve_stays_jax_free():
    """An unconstrained all-binary LINEAR model is the nearest miss to the QUBO gate
    (it fires the structural check but has no quadratic term). With the seed ON by
    default, its simplex-MILP solve must stay JAX-free end to end — the linear
    early-out in ``qubo_local_search`` is what this pins."""
    script = textwrap.dedent(
        """
        import os
        os.environ['DISCOPT_DISABLE_JAX_CACHE'] = '1'
        os.environ.pop('DISCOPT_QUBO_PRIMAL', None)
        import sys
        import discopt.modeling as dm
        m = dm.Model('ubl')
        x = m.binary('x', shape=(6,))
        m.maximize(dm.sum([float(i - 2) * x[i] for i in range(6)]))
        r = m.solve(time_limit=60, nlp_solver='simplex')
        assert r.status == 'optimal', r.status
        print('JAX_LOADED' if 'jax' in sys.modules else 'JAX_FREE')
        """
    )
    out = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=180
    )
    assert out.returncode == 0, f"solve failed:\n{out.stderr}"
    assert out.stdout.strip().splitlines()[-1] == "JAX_FREE", (
        f"#843: unconstrained binary MILP solve imported JAX with the seed ON:\n{out.stdout}"
    )


@pytest.mark.slow
@pytest.mark.correctness
@pytest.mark.skipif(not _CHIMERA.exists(), reason="chimera_k64ising-01.nl (corpus) absent")
def test_843_qubo_primal_incumbent_on_chimera():
    """chimera-01: the QUBO local search lands a real feasible incumbent (was NONE),
    strictly better than the trivial obj-0 floor and sound (<= optimum 24.3)."""
    m = dm.from_nl(str(_CHIMERA))
    ev = NLPEvaluator(m)
    x = qubo_local_search(m, deadline=None)
    assert x is not None and np.all(np.isin(x, [0.0, 1.0]))
    true_obj = -float(ev.evaluate_objective(x))
    assert true_obj > 1.0, f"#843: chimera incumbent {true_obj} not better than the trivial floor"
    assert true_obj <= 24.3 + 1e-3, f"#843: unsound chimera incumbent {true_obj} > optimum 24.3"

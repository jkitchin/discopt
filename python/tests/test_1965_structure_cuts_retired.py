"""#1965: the auto-engaged SymPy structure-cut presolve is retired (CLAUDE.md §5).

``structure_cuts=True`` used to be the default: every small model with a nonlinear
equality ran the square-difference recognizer before the relaxation build. Measured
for #1965 (3 interleaved reps, ``time_limit=60``, ``threads=1``):

* in-repo corpus: 0 of 59 instances under the size gate received a cut;
* the gas-network variants it fires on: status, objective and certificate identical
  with and without it, wall 1.6-4.8x higher (``W_CS_MAX=90``: 6.76 +- 0.19 s ON vs
  2.00 +- 0.08 s OFF).

Its soundness check is sampled, not proved, so it was retired rather than kept as a
default. The recognizer stays as an explicit opt-in tool.
"""

from __future__ import annotations

import discopt.modeling as dm
import numpy as np
import pytest


def _square_difference_model() -> dm.Model:
    """A model with a nonlinear equality: the shape that used to enter the pass."""
    m = dm.Model("sq_1965")
    p = m.continuous("p", lb=1.0, ub=10.0)
    q = m.continuous("q", lb=1.0, ub=10.0)
    f = m.continuous("f", lb=0.0, ub=5.0)
    m.subject_to(p**2 - q**2 == f**2)
    m.subject_to(p >= 2.0)
    m.minimize(f + p)
    return m


def test_default_solve_never_runs_the_recognizer(monkeypatch):
    """The solve path no longer reaches the recognizer at all (sympy or not)."""
    import discopt._relax.symbolic.cut_recognizer as cr

    calls: list[str] = []

    def _forbidden(name):
        def _f(*a, **k):
            calls.append(name)
            raise AssertionError(f"{name} ran on the default solve path")

        return _f

    for name in ("recognize_and_inject", "recognize_and_derive_cuts", "model_to_sympy"):
        monkeypatch.setattr(cr, name, _forbidden(name))
    r = _square_difference_model().solve(time_limit=30, deterministic=True)
    assert r.status == "optimal", r.status
    assert r.objective == pytest.approx(2.0, abs=1e-5)
    assert calls == []


def test_structure_cuts_true_is_refused():
    with pytest.raises(ValueError, match=r"structure_cuts=True.*retired.*#1965"):
        _square_difference_model().solve(time_limit=30, structure_cuts=True)


@pytest.mark.parametrize("value", [False, None])
def test_structure_cuts_false_or_none_is_accepted(value):
    r = _square_difference_model().solve(time_limit=30, structure_cuts=value)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(2.0, abs=1e-5)


def test_explicit_opt_in_still_injects_and_certifies():
    """The recognizer, called by the user before ``solve``, still works."""
    pytest.importorskip("sympy")
    from discopt._relax.symbolic import cut_recognizer as cr
    from discopt.benchmarks.problems.gas_network_minlp import build_gas_network_minlp

    m = build_gas_network_minlp()
    w = next(v for v in m._variables if v.name == "w_cs")
    w.ub = np.full(w.shape, 90.0)  # the box on which the cut is provable
    n_cons = len(m._constraints)
    assert cr.recognize_and_inject(m) == 2
    assert len(m._constraints) > n_cons
    r = m.solve(time_limit=60, threads=1)
    assert r.status == "optimal" and r.gap_certified
    assert r.objective == pytest.approx(3.0025943, abs=1e-5)

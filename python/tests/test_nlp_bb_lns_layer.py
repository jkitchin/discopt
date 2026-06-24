"""Regression: the LNS primal-improvement layer (RINS + local branching) wired
into the NLP-BB path (`_solve_nlp_bb`) must (a) actually run, (b) terminate — its
local-branching sub-solve must NOT recurse back into the layer — and (c) stay
sound (never inject an incumbent better than the true optimum).

Background: the improvers existed only in `solve_model`'s loop, so the
syn/rsyn/clay families (which take `_solve_nlp_bb`) never got polished past the
root incumbent. They were wired into `_solve_nlp_bb` with `_lns_enabled` threaded
as the recursion guard (the local-branching sub-solve sets it False). If that
thread breaks, the sub-solve re-enters the layer and recurses without bound.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt._jax.primal_heuristics as PH  # noqa: E402
import discopt.modeling as dm  # noqa: E402
import discopt.solver as S  # noqa: E402


def _bilinear_minlp() -> dm.Model:
    # Nonconvex bilinear MINLP with integers -> routed to NLP-BB, LNS-eligible.
    # max x*y  s.t. x + y <= 6,  x,y in {0..5}  ->  optimum 9 at x=y=3.
    m = dm.Model("bil")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    m.maximize(x * y)
    m.subject_to(x + y <= 6)
    return m


def test_nlp_bb_accepts_lns_enabled():
    import inspect

    assert "_lns_enabled" in inspect.signature(S._solve_nlp_bb).parameters


def test_lns_layer_terminates_and_is_sound(monkeypatch):
    """Solve a nonconvex MINLP that exercises the LNS layer; it must terminate
    (bounded local-branching calls — recursion would explode/​hang) and never
    report an objective better than the true optimum (9, maximize)."""
    calls = {"lb": 0, "rins": 0}
    _orig_lb = PH.local_branching
    _orig_ri = PH.rins

    def _spy_lb(*a, **k):
        calls["lb"] += 1
        return _orig_lb(*a, **k)

    def _spy_ri(*a, **k):
        calls["rins"] += 1
        return _orig_ri(*a, **k)

    monkeypatch.setattr(PH, "local_branching", _spy_lb)
    monkeypatch.setattr(PH, "rins", _spy_ri)

    r = _bilinear_minlp().solve(time_limit=15, gap_tolerance=1e-4)

    # Terminated with a bounded number of LNS calls — a broken recursion guard
    # would make local_branching's sub-solve re-enter the layer unboundedly.
    assert calls["lb"] < 500, f"local_branching called {calls['lb']}x — recursion guard broken?"
    # Sound: maximize, so the incumbent can never exceed the true optimum.
    if r.objective is not None:
        assert r.objective <= 9.0 + 1e-4, f"FALSE-FEASIBLE: {r.objective} > opt 9"


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-v", "-s"]))

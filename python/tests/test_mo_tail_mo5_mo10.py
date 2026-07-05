"""Regression tests for the MO tail findings MO5-MO10 (issue #413).

* **MO5** (perf, bound-neutral): the sweep helpers compile each objective once
  (an :class:`~discopt.mo.utils.ObjectiveEvaluator`) instead of recompiling it
  on every evaluation. These tests assert the recompile count drops and that the
  cached evaluator returns the *same* value as the per-call
  :func:`~discopt.mo.utils.evaluate_expression`.
* **MO6**: ``weighted_sum(ideal=..., normalize=False)`` (or with an explicit
  ``nadir``) honors the supplied ideal instead of discarding it and re-running
  ``k`` anchor solves.
* **MO7**: ``total_time_limit`` truncates a sweep (front tagged
  ``".../truncated"``) and a large grid warns.
* **MO8**: scalarizers remove exactly the constraints they added, by identity —
  a row appended by other code during the sweep survives.
* **MO9**: ``_payoff_matrix`` no longer takes the dead ``senses_list`` argument.

Most are solve-backed on tiny LP/QP models and run sub-second (``smoke``).
"""

from __future__ import annotations

import warnings

import discopt._jax.dag_compiler as dag_compiler
import discopt.modeling as dm
import numpy as np
import pytest
from discopt.mo import (
    ParetoFront,
    objective_evaluator,
    weighted_sum,
)
from discopt.mo.scalarization import (
    _tag,
    _TimeBudget,
    _warn_large_grid,
    epsilon_constraint,
    weighted_tchebycheff,
)
from discopt.mo.utils import ObjectiveEvaluator, evaluate_expression


def _biobj_qp():
    m = dm.Model("biobj")
    x = m.continuous("x", shape=(2,), lb=0.0, ub=5.0)
    f1 = x[0] ** 2 + x[1] ** 2
    f2 = (x[0] - 2.0) ** 2 + (x[1] - 1.0) ** 2
    return m, [f1, f2]


# ─────────────────────────────────────────────────────────────
# MO5 — compile once, reuse across the sweep (bound-neutral perf)
# ─────────────────────────────────────────────────────────────


class TestMO5CompileCache:
    @pytest.mark.smoke
    def test_evaluator_matches_evaluate_expression(self):
        """The cached evaluator reproduces evaluate_expression bit-for-bit."""
        m, objs = _biobj_qp()
        x_dict = {"x": np.array([1.3, 0.7])}
        ev = objective_evaluator(m, objs)
        for i, e in enumerate(objs):
            direct = evaluate_expression(e, m, x_dict)
            cached = ev.at(i, x_dict)
            assert cached == direct  # exact, not approx
        vec = ev.vector_at(x_dict)
        assert vec.tolist() == [evaluate_expression(e, m, x_dict) for e in objs]

    @pytest.mark.smoke
    def test_evaluator_tracks_parameter_mutation(self):
        """vector_at reads current parameter values on each call."""
        m = dm.Model("param")
        x = m.continuous("x", shape=(1,), lb=-5.0, ub=5.0)
        p = m.parameter("p", value=1.0)
        f = (x[0] - p) ** 2
        ev = ObjectiveEvaluator(m, [f])
        x_dict = {"x": np.array([0.0])}
        assert ev.at(0, x_dict) == pytest.approx(1.0)  # (0 - 1)^2
        p.value = np.asarray(3.0)
        assert ev.at(0, x_dict) == pytest.approx(9.0)  # (0 - 3)^2 — new param

    @pytest.mark.smoke
    def test_sweep_does_not_recompile_per_point(self, monkeypatch):
        """A weighted_sum sweep compiles each objective once, not per point.

        Pre-fix, ``evaluate_expression`` called ``compile_expression`` k times
        per accepted point plus k^2 for the payoff table (26 for an 11-point
        bi-objective sweep). With the evaluator, the legacy per-call
        ``compile_expression`` is not used on the objective-evaluation path and
        ``compile_expression_params`` is called exactly k times (once per
        objective).
        """
        ce_calls = {"n": 0}
        cep_calls = {"n": 0}
        orig_ce = dag_compiler.compile_expression
        orig_cep = dag_compiler.compile_expression_params

        def counting_ce(*a, **k):
            ce_calls["n"] += 1
            return orig_ce(*a, **k)

        def counting_cep(*a, **k):
            cep_calls["n"] += 1
            return orig_cep(*a, **k)

        monkeypatch.setattr(dag_compiler, "compile_expression", counting_ce)
        monkeypatch.setattr(dag_compiler, "compile_expression_params", counting_cep)

        m, objs = _biobj_qp()
        front = weighted_sum(m, objs, n_weights=11, time_limit=10)
        assert front.n >= 3
        # Objective evaluation no longer goes through the per-call recompile.
        assert ce_calls["n"] == 0
        # Exactly one param-agnostic compile per objective (k == 2).
        assert cep_calls["n"] == len(objs)


# ─────────────────────────────────────────────────────────────
# MO6 — honor a supplied ideal without recomputing anchors
# ─────────────────────────────────────────────────────────────


class TestMO6HonorIdeal:
    @pytest.mark.smoke
    def test_ideal_only_no_normalize_skips_ideal_point(self, monkeypatch):
        """weighted_sum(ideal=..., normalize=False) must not recompute the ideal."""
        import discopt.mo.scalarization as scal

        calls = {"n": 0}
        orig = scal.ideal_point

        def counting_ideal_point(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(scal, "ideal_point", counting_ideal_point)

        m, objs = _biobj_qp()
        supplied_ideal = np.array([0.0, 0.0])
        front = weighted_sum(
            m, objs, ideal=supplied_ideal, normalize=False, n_weights=5, time_limit=10
        )
        assert calls["n"] == 0  # ideal honored, no anchor solves
        assert front.ideal is not None
        assert np.allclose(front.ideal, supplied_ideal)

    @pytest.mark.smoke
    def test_ideal_and_nadir_skips_ideal_point(self, monkeypatch):
        """Supplying both ideal and nadir needs no anchor solves either."""
        import discopt.mo.scalarization as scal

        calls = {"n": 0}
        orig = scal.ideal_point

        def counting_ideal_point(*a, **k):
            calls["n"] += 1
            return orig(*a, **k)

        monkeypatch.setattr(scal, "ideal_point", counting_ideal_point)

        m, objs = _biobj_qp()
        front = weighted_sum(
            m,
            objs,
            ideal=np.array([0.0, 0.0]),
            nadir=np.array([25.0, 20.0]),
            normalize=True,
            n_weights=5,
            time_limit=10,
        )
        assert calls["n"] == 0
        assert front.n >= 2


# ─────────────────────────────────────────────────────────────
# MO7 — overall time budget + large-grid warning
# ─────────────────────────────────────────────────────────────


class TestMO7TimeBudget:
    @pytest.mark.smoke
    def test_timebudget_none_never_expires(self):
        b = _TimeBudget(None)
        assert not b.expired()
        assert not b.truncated
        assert _tag("weighted_sum", b) == "weighted_sum"

    @pytest.mark.smoke
    def test_timebudget_expires_and_tags(self):
        b = _TimeBudget(0.0)  # zero budget: already expired
        assert b.expired()
        assert b.truncated
        assert _tag("nbi", b) == "nbi/truncated"

    @pytest.mark.smoke
    def test_warn_large_grid(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_large_grid(201, "epsilon_constraint")
        assert len(w) == 1
        assert "global solves" in str(w[0].message)

    @pytest.mark.smoke
    def test_no_warn_small_grid(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_large_grid(200, "weighted_sum")
        assert len(w) == 0

    @pytest.mark.smoke
    def test_total_time_limit_truncates_sweep(self):
        """A zero budget returns a partial front tagged '/truncated'."""
        m, objs = _biobj_qp()
        front = weighted_sum(m, objs, n_weights=21, total_time_limit=0.0, time_limit=10)
        assert front.method.endswith("/truncated")
        # Anchors are computed before the loop budget check, so the front may be
        # empty; the tag is the load-bearing assertion.
        assert isinstance(front, ParetoFront)


# ─────────────────────────────────────────────────────────────
# MO8 — remove exactly the constraints the scalarizer added
# ─────────────────────────────────────────────────────────────


class TestMO8IdentityCleanup:
    @pytest.mark.smoke
    def test_epsilon_constraint_restores_constraint_list(self):
        m, objs = _biobj_qp()
        # A pre-existing constraint the sweep must not disturb.
        m.subject_to(objs[0] + objs[1] <= 1e6, name="user_pre")
        before = list(m._constraints)
        epsilon_constraint(m, objs, n_points=5, time_limit=10)
        after = list(m._constraints)
        # Exactly the same constraint objects remain, in the same order.
        assert [id(c) for c in after] == [id(c) for c in before]

    @pytest.mark.smoke
    def test_tchebycheff_restores_constraint_list(self):
        m, objs = _biobj_qp()
        m.subject_to(objs[0] <= 1e6, name="user_pre")
        before = list(m._constraints)
        weighted_tchebycheff(m, objs, n_weights=5, time_limit=10)
        after = list(m._constraints)
        assert [id(c) for c in after] == [id(c) for c in before]

    @pytest.mark.smoke
    def test_identity_removal_keeps_foreign_rows(self):
        """A row appended during the sweep by *other* code must survive.

        Simulates a callback/reformulation that mutates ``_constraints`` while a
        scalarizer is running: with the old positional
        ``del model._constraints[saved_n:]`` such a row would be dropped; with
        identity-based removal it is preserved.
        """
        from discopt.mo.scalarization import _add_tracked, _remove_tracked

        m, objs = _biobj_qp()
        tracked: list = []
        _add_tracked(m, objs[0] <= 5.0, "mo_row", tracked)
        # Foreign row appended after the tracked one.
        foreign = objs[1] <= 5.0
        m.subject_to(foreign, name="foreign")
        _remove_tracked(m, tracked)
        remaining_ids = {id(c) for c in m._constraints}
        assert id(foreign) in remaining_ids  # foreign row survives
        assert id(tracked[0]) not in remaining_ids  # tracked row removed


# ─────────────────────────────────────────────────────────────
# MO9 — _payoff_matrix no longer takes the dead senses_list arg
# ─────────────────────────────────────────────────────────────


class TestMO9DeadArg:
    @pytest.mark.smoke
    def test_payoff_matrix_signature(self):
        import inspect

        from discopt.mo.nbi import _payoff_matrix

        params = list(inspect.signature(_payoff_matrix).parameters)
        assert "senses_list" not in params
        assert params == ["model", "objectives", "anchors", "evaluator"]

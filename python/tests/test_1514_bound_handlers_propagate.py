"""#1514: solve-time relaxation/bound/presolve handlers no longer swallow defects.

About forty ``except Exception`` handlers in ``solver.py`` (plus three in
``_relax/disjunctive_config_bound.py``) logged a failed bound attempt at DEBUG and
carried on. Each was individually "sound" -- a bound that is not computed cannot be
wrong -- but together they turned every crash in the relaxation layer into a silent
loss of strength (#1493's vector-norm ``TypeError`` was the example). Each site is
now one of:

* **removed** -- the callee already declines by return value, so an exception is a
  defect and fails the solve;
* **narrowed** -- the callee documents a specific decline exception, which is still
  absorbed; everything else propagates;
* **kept as a sound fallback** -- a genuine external failure (a POUNCE IPM raising
  from native code), now reported through ``_warn_fallback_once``: WARNING on the
  first failure per site per solve, DEBUG afterwards.

Every test counts the injected calls and asserts the count is non-zero, so a model
that stopped reaching the site fails here instead of passing vacuously (CLAUDE.md §6).
"""

from __future__ import annotations

import logging

import discopt.modeling as dm
import discopt.solver as solver
import numpy as np
import pytest
from discopt._relax import disjunctive_config_bound as dcb
from discopt._relax import nonlinear_bound_tightening as nbt
from discopt._relax import obbt as obbt_mod
from discopt._relax import presolve_pipeline as pp
from discopt._relax.mccormick_lp import MccormickLPRelaxer
from discopt._relax.symbolic import cut_recognizer as cr
from discopt.symbolic import SymbolicTranslationError


class _Boom(RuntimeError):
    """The injected defect; a distinct type so an assertion cannot match by luck."""


def _counting_raiser(exc_factory, calls):
    def _raise(*_a, **_k):
        calls.append(1)
        raise exc_factory()

    return _raise


def _bilinear() -> dm.Model:
    """Small nonconvex continuous model: exercises the spatial B&B path."""
    m = dm.Model("bilinear_1514")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    y = m.continuous("y", lb=-1.0, ub=2.0)
    m.minimize(x * y + 0.1 * x)
    m.subject_to(x + y >= 0.5)
    return m


def _transcendental() -> dm.Model:
    """Nonconvex, non-quadratic: engages the McCormick LP relaxer (root probe and
    node loop)."""
    m = dm.Model("transc_1514")
    x = m.continuous("x", lb=0.5, ub=3.0)
    y = m.continuous("y", lb=0.5, ub=3.0)
    m.minimize(x * dm.exp(-y) + y * dm.log(x))
    m.subject_to(x + y >= 2.0)
    return m


# ── removed: the callee declines by return value ────────────────────────────


def test_node_nonlinear_tightening_defect_propagates(monkeypatch):
    """``_apply_nonlinear_tightening_with_status`` (the per-node entry point) used to
    log the exception and return the box untightened."""
    calls: list[int] = []
    monkeypatch.setattr(
        nbt, "tighten_nonlinear_bounds", _counting_raiser(lambda: _Boom("nbt"), calls)
    )
    with pytest.raises(_Boom, match="nbt"):
        solver._apply_nonlinear_tightening_with_status(
            _bilinear(), np.array([-1.0, -1.0]), np.array([2.0, 2.0])
        )
    assert calls


def test_declared_box_tightening_defect_propagates(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        nbt, "tighten_nonlinear_bounds", _counting_raiser(lambda: _Boom("declared"), calls)
    )
    with pytest.raises(_Boom, match="declared"):
        solver._declared_box_tightening(_bilinear())
    assert calls


def test_node_nonlinear_tightening_still_reports_infeasibility():
    """The decline path is unchanged: an empty box comes back as ``infeasible``."""
    m = _bilinear()
    lb = np.array([1.0, 0.0])
    ub = np.array([0.0, 1.0])  # x's interval is empty
    _, _, infeasible = solver._apply_nonlinear_tightening_with_status(m, lb, ub)
    assert infeasible is True


def test_node_mccormick_lp_defect_fails_the_solve(monkeypatch):
    """The per-node ``solve_at_node`` handlers (serial and batched) are gone."""
    calls: list[int] = []
    monkeypatch.setattr(
        MccormickLPRelaxer, "solve_at_node", _counting_raiser(lambda: _Boom("mc"), calls)
    )
    with pytest.raises(_Boom, match="mc"):
        _transcendental().solve(time_limit=30, deterministic=True)
    assert calls, "solve_at_node was never reached; the test proves nothing"


def test_node_loop_mccormick_lp_defect_fails_the_solve(monkeypatch):
    """Past the root probe: the node-loop ``solve_at_node`` handlers are gone too."""
    orig = MccormickLPRelaxer.solve_at_node
    calls: list[int] = []

    def _second_call_raises(self, *a, **k):
        calls.append(1)
        if len(calls) == 1:
            return orig(self, *a, **k)  # the root probe succeeds
        raise _Boom("node mc")

    monkeypatch.setattr(MccormickLPRelaxer, "solve_at_node", _second_call_raises)
    with pytest.raises(_Boom, match="node mc"):
        _transcendental().solve(time_limit=30, deterministic=True)
    assert len(calls) >= 2, "no solve_at_node call after the root probe"


def test_root_obbt_defect_fails_the_solve(monkeypatch):
    calls: list[int] = []
    monkeypatch.setattr(
        obbt_mod, "obbt_tighten_root", _counting_raiser(lambda: _Boom("obbt"), calls)
    )
    with pytest.raises(_Boom, match="obbt"):
        _bilinear().solve(time_limit=30, deterministic=True)
    assert calls, "root OBBT was never reached; the test proves nothing"


def test_root_presolve_refusal_is_no_longer_swallowed(monkeypatch):
    """``propagate_bounds_to_model`` RAISES on purpose (an optimality-derived box
    must not become declared bounds). The old catch-all turned that documented
    refusal into "presolve found nothing"."""
    calls: list[int] = []
    monkeypatch.setattr(
        pp,
        "propagate_bounds_to_model",
        _counting_raiser(lambda: ValueError("optimality-derived box"), calls),
    )
    with pytest.raises(ValueError, match="optimality-derived box"):
        _bilinear().solve(time_limit=30, deterministic=True)
    assert calls, "root presolve was never reached; the test proves nothing"


def test_reduce_node_defect_propagates():
    """``_reduce_node_and_stage`` no longer maps a crash to "no tightening"."""
    calls: list[int] = []
    with pytest.raises(_Boom, match="reduce"):
        solver._reduce_node_and_stage(
            _counting_raiser(lambda: _Boom("reduce"), calls),
            None,
            0,
            [[0.0]],
            [[1.0]],
            None,
            None,
            None,
            {},
        )
    assert calls


def test_disjunctive_config_bound_obbt_defect_propagates(monkeypatch):
    """The per-leaf OBBT catch-all in the #732 pass is gone."""
    calls: list[int] = []
    monkeypatch.setattr(
        obbt_mod, "obbt_tighten_root", _counting_raiser(lambda: _Boom("leaf"), calls)
    )
    m = _bilinear()
    m._ipx_config_indicators = [0]  # engage the pass on a free "indicator"
    m._ipx_config_counts = []
    with pytest.raises(_Boom, match="leaf"):
        dcb.compute_disjunctive_config_bound(
            m, np.array([0.0, -1.0]), np.array([1.0, 2.0]), max_leaf_solves=2
        )
    assert calls


# ── narrowed: the documented decline is still absorbed ──────────────────────


def test_box_fbbt_abstains_only_on_missing_repr(monkeypatch):
    import discopt._rust as rust

    m = _bilinear()
    lb = np.array([-1.0, -1.0])
    ub = np.array([2.0, 2.0])
    calls: list[int] = []
    monkeypatch.setattr(
        rust, "model_to_repr", _counting_raiser(lambda: ValueError("no repr"), calls)
    )
    tl, tu, crossed = dcb._box_fbbt(m, lb, ub)
    assert calls and crossed is False
    assert np.array_equal(tl, lb) and np.array_equal(tu, ub)

    monkeypatch.setattr(rust, "model_to_repr", _counting_raiser(lambda: _Boom("repr"), calls))
    with pytest.raises(_Boom, match="repr"):
        dcb._box_fbbt(m, lb, ub)
    assert len(calls) == 2


def test_clique_extraction_declines_only_on_missing_repr(monkeypatch):
    import discopt._rust as rust

    m = _bilinear()
    calls: list[int] = []
    monkeypatch.setattr(
        rust, "model_to_repr", _counting_raiser(lambda: ValueError("no repr"), calls)
    )
    assert solver._extract_clique_edges(m) == []
    monkeypatch.setattr(rust, "model_to_repr", _counting_raiser(lambda: _Boom("clq"), calls))
    with pytest.raises(_Boom, match="clq"):
        solver._extract_clique_edges(m)
    assert len(calls) == 2


# The solve-path structure-cut tests (a recognizer defect fails the solve; its
# documented declines skip) went with the auto-engaged pass in #1965: the solver
# no longer calls the recognizer.


def test_recognizer_declines_non_polynomial_rows_with_the_documented_type():
    """Uncovered by #1514: ``sp.Poly`` raises an untyped ``PolynomialError`` on a
    ``sqrt(...)`` row (nvs01), which the solver's catch-all read as "recognizer
    skipped". It is now the recognizer's typed decline -- same outcome, visible
    type -- so the solver can absorb exactly it."""
    import sympy as sp

    cr._ensure_sympy()
    x = sp.Symbol("x", real=True)
    with pytest.raises(SymbolicTranslationError, match="non-polynomial"):
        cr._linear_terms(sp.sqrt(x**2 + 900.0) - x)
    assert cr._linear_terms(2.0 * x + 1.0) == ({x: 2.0}, 1.0)


def test_recognizer_precheck_skips_non_algebraic_rows():
    """A disjunctive/logical row has no ``sense``; the precheck raised
    AttributeError on every such model and the solver swallowed it."""

    class _Row:
        pass

    m = _bilinear()
    m._constraints.append(_Row())
    assert cr.has_square_difference_candidate(m) is False
    with pytest.raises(SymbolicTranslationError, match="non-algebraic row"):
        cr.model_to_sympy(m)


def test_nonlinear_tightening_cache_tracks_added_variables():
    """The structural cache token ignored the variable list, so adding a variable
    and restoring the constraint list (``epsilon_constraint`` does) reused a stale
    offset map -> ``KeyError`` inside a rule, swallowed by the old handlers."""
    m = _bilinear()
    lb, ub = np.array([-1.0, -1.0]), np.array([2.0, 2.0])
    nbt.tighten_nonlinear_bounds(m, lb, ub)  # populate the cache
    saved = list(m._constraints)
    z = m.continuous("z", lb=0.0, ub=1.0)
    m.subject_to(z * z <= 0.25)
    m._constraints[:] = saved  # restore the list in place: same id, same length
    m.subject_to(z * z + z <= 0.5)
    m._constraints.pop()
    lb3, ub3 = np.array([-1.0, -1.0, 0.0]), np.array([2.0, 2.0, 1.0])
    out_lb, out_ub, _ = nbt.tighten_nonlinear_bounds(m, lb3, ub3)
    assert out_lb.shape == (3,)


class _UnknownNode:
    """A node type the recognizer's translator does not know."""


def test_recognizer_translator_refuses_with_the_documented_type():
    """An unsupported node used to raise a bare ``TypeError`` -- the very type a
    real defect raises, so the solver could only absorb it with a catch-all."""
    cr._ensure_sympy()
    with pytest.raises(SymbolicTranslationError, match="node type _UnknownNode"):
        cr._to_sympy(_UnknownNode(), {})


def test_nlpbb_root_cuts_narrowed(monkeypatch):
    from discopt.solvers import _root_cuts as rc

    def _convex_minlp() -> dm.Model:
        # The #781 fixture: fixed-charge network + one convex quadratic row with a
        # linear objective, which ``nlp_bb=True`` routes through the root-cut stage.
        m = dm.Model("rc_1514")
        f0 = m.continuous("f0", lb=0.0, ub=10.0)
        f1 = m.continuous("f1", lb=0.0, ub=10.0)
        y0 = m.binary("y0")
        y1 = m.binary("y1")
        m.subject_to(f0 - 8.0 * y0 <= 0.0)
        m.subject_to(f1 - 8.0 * y1 <= 0.0)
        m.subject_to(f0 + f1 >= 3.0)
        m.subject_to(f0 * f0 + f1 * f1 <= 16.0)
        m.minimize(f0 + 2.0 * f1 + 2.5 * y0 + 2.5 * y1)
        return m

    calls: list[int] = []
    monkeypatch.setattr(
        rc,
        "generate_root_cuts",
        _counting_raiser(lambda: rc.RootCutsNotApplicable("nonlinear objective"), calls),
    )
    res = _convex_minlp().solve(time_limit=30, deterministic=True, nlp_bb=True)
    assert calls, "the NLP-BB root-cut stage was never reached"
    assert res.status == "optimal"

    monkeypatch.setattr(rc, "generate_root_cuts", _counting_raiser(lambda: _Boom("gmi"), calls))
    with pytest.raises(_Boom, match="gmi"):
        _convex_minlp().solve(time_limit=30, deterministic=True, nlp_bb=True)


def test_eigenvalue_root_bound_narrowed(monkeypatch):
    from discopt._relax.convexity import eigenvalue_arith as ea

    def _qp() -> dm.Model:
        m = dm.Model("qp_1514")
        x = m.continuous("x", lb=-1.0, ub=1.0)
        y = m.continuous("y", lb=-1.0, ub=1.0)
        m.minimize(x * x - y * y + x * y)
        return m

    calls: list[int] = []
    monkeypatch.setattr(
        ea, "quadratic_form_bound", _counting_raiser(lambda: np.linalg.LinAlgError("eigh"), calls)
    )
    res = _qp().solve(time_limit=30, deterministic=True, eigenvalue_root_bound=True)
    assert calls, "the eigenvalue root bound was never reached"
    assert res.status == "optimal"

    monkeypatch.setattr(ea, "quadratic_form_bound", _counting_raiser(lambda: _Boom("eig"), calls))
    with pytest.raises(_Boom, match="eig"):
        _qp().solve(time_limit=30, deterministic=True, eigenvalue_root_bound=True)


# ── kept: sound fallback, reported once per solve at WARNING ────────────────


def _warnings_from(caplog, needle):
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING and r.name == "discopt.solver" and needle in r.getMessage()
    ]


def test_warn_fallback_once_is_once_per_site_per_solve(caplog):
    caplog.set_level(logging.DEBUG, logger="discopt.solver")

    @solver._scoped_fallback_warnings
    def _fake_solve():
        for _ in range(3):
            solver._warn_fallback_once("site A", _Boom("a"), "falling back")
        solver._warn_fallback_once("site B", _Boom("b"), "falling back")
        # A nested solve is part of the same user-visible solve.
        _nested()

    @solver._scoped_fallback_warnings
    def _nested():
        solver._warn_fallback_once("site A", _Boom("a"), "falling back")

    _fake_solve()
    assert len(_warnings_from(caplog, "site A failed")) == 1
    assert len(_warnings_from(caplog, "site B failed")) == 1
    a = _warnings_from(caplog, "site A failed")[0].getMessage()
    assert "_Boom" in a and "falling back" in a
    repeats = [
        r
        for r in caplog.records
        if r.levelno == logging.DEBUG and "site A failed again" in r.getMessage()
    ]
    assert len(repeats) == 3

    # A second top-level solve starts a fresh record.
    caplog.clear()
    _fake_solve()
    assert len(_warnings_from(caplog, "site A failed")) == 1


def test_pounce_cut_loop_failure_warns_once(monkeypatch, caplog):
    """A kept site end to end: the MILP cut loop's POUNCE seed solve."""
    from discopt.solvers import lp_pounce

    if not lp_pounce.POUNCE_AVAILABLE:
        pytest.skip("POUNCE not installed")
    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    calls: list[int] = []
    monkeypatch.setattr(lp_pounce, "solve_lp", _counting_raiser(lambda: _Boom("ipm"), calls))

    class _LP:
        c = np.array([1.0, 1.0])
        A_eq = np.array([[1.0, 1.0]])
        b_eq = np.array([1.0])
        x_l = np.array([0.0, 0.0])
        x_u = np.array([1.0, 1.0])

    @solver._scoped_fallback_warnings
    def _run():
        for _ in range(2):
            assert solver._cut_loop_relaxation_x(_LP(), prefer_pounce=True) is None

    _run()
    assert len(calls) == 2
    warned = _warnings_from(caplog, "root cut loop: POUNCE LP failed")
    assert len(warned) == 1 and "_Boom" in warned[0].getMessage()


def test_undifferentiable_model_falls_back_with_one_warning(caplog):
    """A kept site: the linear-row FBBT Jacobian. A model the evaluator cannot
    differentiate (e.g. a numpy-only ``dm.custom`` callable under the JAX fallback)
    falls back to interval tightening -- sound -- and now says so once."""
    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    m = _bilinear()
    calls: list[int] = []

    class _Ev:
        n_constraints = 1
        _model = m

        def evaluate_jacobian(self, x):
            calls.append(1)
            raise TypeError("cannot trace user callable")

    @solver._scoped_fallback_warnings
    def _run():
        for _ in range(2):
            lb, ub, infeasible = solver._tighten_node_bounds_with_status(
                _Ev(), np.array([-1.0, -1.0]), np.array([2.0, 2.0]), [0.5], [1e20]
            )
            assert infeasible is False and lb.shape == (2,)

    _run()
    assert len(calls) == 2
    warned = _warnings_from(caplog, "linear-row FBBT: constraint Jacobian failed")
    assert len(warned) == 1 and "TypeError" in warned[0].getMessage()

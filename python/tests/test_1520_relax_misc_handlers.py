"""#1520: broad ``except Exception`` handlers in the misc relaxation-layer modules.

Continues #1514 over the rest of ``python/discopt/_relax`` (the modules not owned
by the classifier / core-bound / solver agents). Each handler is now one of:

* **removed** -- the callee declines by return value (or has no documented
  failure mode), so an exception is a defect and propagates;
* **narrowed** -- the callee's documented decline exception is still absorbed,
  everything else propagates;
* **kept as a sound fallback** -- a genuine external failure (a native solver, an
  optional solver package, a caller-supplied presolve pass) whose fallback cannot
  produce a wrong bound; it now reports through ``warn_fallback_once`` (WARNING on
  ``discopt.solver``) instead of DEBUG or nothing.

Every injection counts its calls and asserts the count is non-zero, so a test
whose fixture stopped reaching the site fails instead of passing vacuously
(CLAUDE.md §6). The one real defect the removals exposed -- ``find_active_set``
crashing on a vector-valued constraint, silently turned into ``l3_failed`` -- has
its own regression test at the bottom.
"""

from __future__ import annotations

import importlib.util
import logging
import sys
import types

import discopt.modeling as dm
import pytest

pytest.importorskip("jax", reason="JAX is the optional discopt[jax] extra (#1873)")

import numpy as np
from discopt._relax._fallback import FALLBACK_WARNINGS


class _Boom(RuntimeError):
    """The injected defect; a distinct type so an assertion cannot match by luck."""


def _counting_raiser(exc_factory, calls):
    def _raise(*_a, **_k):
        calls.append(1)
        raise exc_factory()

    return _raise


@pytest.fixture(autouse=True)
def _fresh_fallback_record():
    """``warn_fallback_once`` is once per site per solve; these tests call the
    sites outside a solve, so reset the record around each test."""
    FALLBACK_WARNINGS.seen.clear()
    yield
    FALLBACK_WARNINGS.seen.clear()


def _warnings_from(caplog, needle):
    return [
        r
        for r in caplog.records
        if r.name == "discopt.solver" and r.levelno == logging.WARNING and needle in r.getMessage()
    ]


# ── small fixtures ───────────────────────────────────────────────────────────


def _lp_with_param():
    m = dm.Model("lp1520")
    p = m.parameter("p", value=2.0)
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(x >= p)
    m.minimize(3.0 * x)
    return m, p


def _tiny_milp():
    m = dm.Model("milp1520")
    x = m.continuous("x", lb=0.0, ub=4.0)
    y = m.integer("y", lb=0, ub=3)
    m.subject_to(x + y >= 1.5)
    m.minimize(2.0 * x + y)
    return m


def _quad_model():
    m = dm.Model("quad1520")
    x = m.continuous("x", lb=-2.0, ub=3.0)
    y = m.continuous("y", lb=-1.0, ub=2.0)
    m.minimize(x * x - x * y + 0.5 * y)
    m.subject_to(x + y <= 2.0)
    m.subject_to(x * y >= -1.0)
    return m


def _gear4():
    m = dm.Model("gear4_1520")
    i1 = m.integer("i1", lb=12, ub=60)
    i2 = m.integer("i2", lb=12, ub=60)
    i3 = m.integer("i3", lb=12, ub=60)
    i4 = m.integer("i4", lb=12, ub=60)
    s = m.continuous("s", lb=0, ub=100000)
    t = m.continuous("t", lb=0, ub=100000)
    m.subject_to(1e6 * (i1 * i2) / (i3 * i4) + s - t == 144279.32477276)
    m.minimize(s + t)
    return m


# ── differentiable_solve.py ─────────────────────────────────────────────────


def test_solve_objective_forward_failure_propagates(monkeypatch):
    """Removed: ``_solve_objective`` returned ``None`` on any exception, which
    ``gradient`` turned into a silent 0.0 sensitivity."""
    from discopt._relax import differentiable_solve as ds
    from discopt._relax.problem_classifier import ProblemClass

    m, _p = _lp_with_param()
    calls: list[int] = []
    monkeypatch.setattr(ds, "_lp_forward", _counting_raiser(lambda: _Boom("lp fwd"), calls))
    with pytest.raises(_Boom):
        ds._solve_objective(m, ProblemClass.LP)
    assert calls


def test_milp_relaxation_objective_defect_propagates_decline_absorbed(monkeypatch):
    """Narrowed: the MILP arm's continuous-relaxation objective absorbs only
    ``ImportError`` / ``PounceKKTError``."""
    from discopt._relax import differentiable_solve as ds
    from discopt.solvers.lp_pounce import PounceKKTError

    calls: list[int] = []
    monkeypatch.setattr(ds, "_lp_forward", _counting_raiser(lambda: _Boom("relax"), calls))
    with pytest.raises(_Boom):
        ds.differentiable_solve(_tiny_milp())
    assert calls

    declines: list[int] = []
    monkeypatch.setattr(
        ds, "_lp_forward", _counting_raiser(lambda: PounceKKTError("no converge"), declines)
    )
    res = ds.differentiable_solve(_tiny_milp())
    assert declines
    assert res.relaxation_objective() is None
    assert res.objective == pytest.approx(2.0, abs=1e-6)  # y=1, x=0.5 (or y=2, x=0)


# ── gdp_advisor.py / gdp_reformulate.py ─────────────────────────────────────


def test_gdp_advisor_bound_defect_propagates(monkeypatch):
    """Removed: a raise from ``_bound_expression`` read as "unbounded M"."""
    from discopt._relax import gdp_reformulate
    from discopt._relax.gdp_advisor import recommend_methods

    m = dm.Model("adv1520")
    x = m.continuous("x", lb=0.0, ub=20.0)
    m.minimize(x)
    m.either_or([[x <= 5.0], [x >= 15.0]], name="modes")
    calls: list[int] = []
    monkeypatch.setattr(
        gdp_reformulate, "_bound_expression", _counting_raiser(lambda: _Boom("bnd"), calls)
    )
    with pytest.raises(_Boom):
        recommend_methods(m)
    assert calls


def test_big_m_lp_oracle_defect_propagates(monkeypatch):
    """Removed: an exception from the exact LP oracle became "use the default
    big-M" at DEBUG."""
    from discopt._relax import gdp_reformulate as G
    from discopt.solvers import lp_backend

    m = dm.Model("bigm1520")
    x = m.continuous("x", lb=0.0, ub=10.0)
    m.subject_to(x <= 4)
    m.minimize(x)
    lp_data = G._precompute_lp_relaxation(m)
    assert lp_data is not None
    calls: list[int] = []
    monkeypatch.setattr(
        lp_backend,
        "get_exact_lp_solver",
        lambda: _counting_raiser(lambda: _Boom("lp"), calls),
    )
    with pytest.raises(_Boom):
        G._compute_big_m_lp(x <= 1, m, lp_data)
    assert calls


# ── quadratic_form.py / edge_concave.py ─────────────────────────────────────


def test_quadratic_support_polynomial_defect_propagates(monkeypatch):
    """Removed: a raise from the polynomial walker read as "not quadratic"."""
    from discopt._relax import milp_relaxation
    from discopt._relax.quadratic_form import extract_quadratic_support

    m = _quad_model()
    assert extract_quadratic_support(m._objective.expression, m) is not None  # reaches it
    calls: list[int] = []
    monkeypatch.setattr(
        milp_relaxation, "_expr_to_polynomial", _counting_raiser(lambda: _Boom("poly"), calls)
    )
    with pytest.raises(_Boom):
        extract_quadratic_support(m._objective.expression, m)
    assert calls


def test_edge_concave_polynomial_defect_propagates(monkeypatch):
    """Removed: a raise skipped the body ("edge-concave found nothing")."""
    from discopt._relax import milp_relaxation
    from discopt._relax.edge_concave import collect_edge_concave_quadratics

    m = _quad_model()
    collect_edge_concave_quadratics(m)  # the unpatched path runs clean
    calls: list[int] = []
    monkeypatch.setattr(
        milp_relaxation, "_expr_to_polynomial", _counting_raiser(lambda: _Boom("poly"), calls)
    )
    with pytest.raises(_Boom):
        collect_edge_concave_quadratics(m)
    assert calls


# ── scalarize.py ────────────────────────────────────────────────────────────


class _NumpyProxy(types.SimpleNamespace):
    """``scalarize.np`` stand-in whose ``arange`` raises; everything else is numpy."""

    def __init__(self, exc_factory, calls):
        super().__init__()
        self._exc_factory = exc_factory
        self._calls = calls

    def arange(self, *_a, **_k):
        self._calls.append(1)
        raise self._exc_factory()

    def __getattr__(self, name):
        return getattr(np, name)


def test_scalarize_index_resolution_narrowed(monkeypatch):
    """Narrowed to numpy's indexing failures (``IndexError``/``TypeError``/
    ``ValueError``), which still become ``_Unscalarizable``."""
    from discopt._relax import scalarize as sc
    from discopt.modeling.core import IndexExpression

    m = dm.Model("sc1520")
    x = m.continuous("x", shape=(3,), lb=0.0, ub=1.0)

    def fresh():
        e = IndexExpression(x, slice(0, 3), shape_hint=(3,))
        e._scalarize_shape = (3,)  # the memo static_shape reads
        return e

    calls: list[int] = []
    monkeypatch.setattr(sc, "np", _NumpyProxy(lambda: _Boom("np"), calls))
    with pytest.raises(_Boom):
        sc._elem(fresh(), (0,))
    assert calls

    declines: list[int] = []
    monkeypatch.setattr(sc, "np", _NumpyProxy(lambda: IndexError("bad index"), declines))
    with pytest.raises(sc._Unscalarizable):
        sc._elem(fresh(), (0,))
    assert declines


# ── presolve_pipeline.py / presolve/*.py ────────────────────────────────────


def test_reverse_ad_tightening_defect_propagates(monkeypatch):
    """Removed: ``run_reverse_ad_tightening`` returned 0 on any exception."""
    from discopt._relax.convexity import interval_ad_reverse
    from discopt._relax.presolve_pipeline import run_reverse_ad_tightening

    calls: list[int] = []
    monkeypatch.setattr(
        interval_ad_reverse, "tighten_box", _counting_raiser(lambda: _Boom("rad"), calls)
    )
    with pytest.raises(_Boom):
        run_reverse_ad_tightening(_quad_model())
    assert calls


def test_propagate_bounds_var_names_defect_propagates():
    """Removed: ``var_names`` raising fell back to the positional block mapping
    (the "older repr" it guarded against cannot be loaded)."""
    from discopt._relax.presolve_pipeline import propagate_bounds_to_model

    calls: list[int] = []

    class _Repr:
        n_var_blocks = 2

        def var_names(self):
            calls.append(1)
            raise _Boom("names")

    with pytest.raises(_Boom):
        propagate_bounds_to_model(_quad_model(), _Repr())
    assert calls


def test_reverse_ad_pass_defect_propagates(monkeypatch):
    """Removed: ``ReverseADPass.run`` returned an empty delta on any exception."""
    from discopt._relax.convexity import interval_ad_reverse
    from discopt._relax.presolve.reverse_ad import ReverseADPass
    from discopt._rust import model_to_repr

    m = _quad_model()
    rep = model_to_repr(m, getattr(m, "_builder", None))
    calls: list[int] = []
    monkeypatch.setattr(
        interval_ad_reverse, "tighten_box", _counting_raiser(lambda: _Boom("rad"), calls)
    )
    with pytest.raises(_Boom):
        ReverseADPass(m).run(rep)
    assert calls


def test_separability_pass_defect_propagates(monkeypatch):
    """Removed: ``SeparabilityPass.run`` returned an empty delta on any exception."""
    from discopt._relax.presolve import separability

    calls: list[int] = []
    monkeypatch.setattr(
        separability, "detect_separability", _counting_raiser(lambda: _Boom("sep"), calls)
    )
    with pytest.raises(_Boom):
        separability.SeparabilityPass(_quad_model()).run(None)
    assert calls


def test_convex_reform_pass_certificate_defect_propagates(monkeypatch):
    """Removed: ``certify_convex`` absorbs its own ``ValueError`` decline, so any
    other raise is a defect, not "this constraint is not convex"."""
    from discopt._relax.convexity import certificate
    from discopt._relax.presolve.convex_reform import ConvexReformPass

    m = _quad_model()
    calls: list[int] = []
    monkeypatch.setattr(
        certificate, "certify_convex", _counting_raiser(lambda: _Boom("cert"), calls)
    )
    with pytest.raises(_Boom):
        ConvexReformPass(m, box={}).run(None)
    assert calls


def test_orchestrator_python_pass_failure_warns_once(caplog):
    """Kept: a caller-supplied presolve pass that raises contributes an empty,
    error-stamped delta (sound: no tightening) -- now with a WARNING."""
    from discopt._relax.presolve.orchestrator import run_orchestrated_presolve
    from discopt._rust import model_to_repr

    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    calls: list[int] = []

    class _RaisingPass:
        name = "boom1520"

        def run(self, model_repr):
            calls.append(1)
            raise _Boom("pass")

    m = _quad_model()
    rep = model_to_repr(m, getattr(m, "_builder", None))
    _, stats = run_orchestrated_presolve(rep, rust_passes=["fbbt"], python_passes=[_RaisingPass()])
    assert calls
    assert any("error" in d for d in stats["deltas"])
    assert len(_warnings_from(caplog, "presolve Python pass 'boom1520' failed")) == 1


# ── partition_selection.py ──────────────────────────────────────────────────


@pytest.mark.parametrize("weighted", [False, True])
def test_vertex_cover_milp_defect_propagates(monkeypatch, weighted):
    """Removed (both covers): ``_solve_vertex_cover_milp`` already falls back to
    the greedy cover itself; a raise is a defect."""
    from discopt._relax import partition_selection as ps
    from discopt._relax.term_classifier import NonlinearTerms

    terms = NonlinearTerms(
        bilinear=[(0, 1), (1, 2), (2, 3)],
        partition_candidates=[0, 1, 2, 3],
    )
    calls: list[int] = []
    monkeypatch.setattr(
        ps, "_solve_vertex_cover_milp", _counting_raiser(lambda: _Boom("vc"), calls)
    )
    with pytest.raises(_Boom):
        if weighted:
            ps._weighted_min_vertex_cover(terms, {0: 1.0, 1: 2.0, 2: 0.5, 3: 1.0})
        else:
            ps._min_vertex_cover(terms)
    assert calls


# ── claim_audit.py ──────────────────────────────────────────────────────────


def test_claim_audit_scipy_import_not_swallowed(monkeypatch):
    """Removed: a failing ``scipy.sparse`` import used to define ``_is_sparse`` as
    "never sparse". Load a fresh copy of the module with the import blocked."""
    import discopt._relax.claim_audit as ca

    monkeypatch.setitem(sys.modules, "scipy.sparse", None)  # import -> ImportError
    spec = importlib.util.spec_from_file_location("_claim_audit_1520", ca.__file__)
    fresh = importlib.util.module_from_spec(spec)
    with pytest.raises(ImportError):
        spec.loader.exec_module(fresh)


# ── least_squares.py ────────────────────────────────────────────────────────


class _EqRaises:
    def __init__(self, exc_factory, calls):
        self._exc_factory = exc_factory
        self._calls = calls

    def __eq__(self, other):
        self._calls.append(1)
        raise self._exc_factory()

    __hash__ = object.__hash__


def test_least_squares_index_equal_narrowed():
    """Narrowed (both handlers) to numpy's comparison failures."""
    from discopt._relax.least_squares import _index_equal

    # Mixed types -> the object-array comparison branch.
    calls: list[int] = []
    with pytest.raises(_Boom):
        _index_equal([_EqRaises(lambda: _Boom("eq"), calls)], (1,))
    assert calls
    declines: list[int] = []
    assert _index_equal([_EqRaises(lambda: ValueError("ambiguous"), declines)], (1,)) is False
    assert declines

    # Same (non-tuple, non-slice) type -> the ``np.array_equal`` branch.
    calls2: list[int] = []
    with pytest.raises(_Boom):
        _index_equal([_EqRaises(lambda: _Boom("eq"), calls2)], [1])
    assert calls2


# ── perspective.py ──────────────────────────────────────────────────────────


def test_perspective_objective_hessian_narrowed(monkeypatch):
    """Narrowed to ``_NotQuadraticError``."""
    from discopt._relax import perspective, problem_classifier

    m = _quad_model()
    calls: list[int] = []
    monkeypatch.setattr(
        problem_classifier,
        "_extract_quadratic_coefficients",
        _counting_raiser(lambda: _Boom("q"), calls),
    )
    with pytest.raises(_Boom):
        perspective._objective_hessian(m)
    assert calls

    declines: list[int] = []
    monkeypatch.setattr(
        problem_classifier,
        "_extract_quadratic_coefficients",
        _counting_raiser(lambda: problem_classifier._NotQuadraticError("cubic"), declines),
    )
    assert perspective._objective_hessian(m) is None
    assert declines


# ── objective_epigraph.py ───────────────────────────────────────────────────


def test_objective_epigraph_classify_defect_propagates(monkeypatch):
    """Removed: ``classify_expr`` answers ``UNKNOWN`` when it cannot prove
    curvature; a raise is a defect, not "not convex"."""
    import discopt._relax.convexity as convexity
    from discopt._relax.objective_epigraph import relax_objective_defining_equality

    m = dm.Model("epi1520")
    x = m.continuous("x", lb=-5, ub=5)
    y = m.continuous("y", lb=-5, ub=5)
    z = m.continuous("z", lb=-1e20, ub=1e20)
    m.subject_to(z == x * x + y * y)
    m.minimize(z)
    calls: list[int] = []
    monkeypatch.setattr(convexity, "classify_expr", _counting_raiser(lambda: _Boom("cls"), calls))
    with pytest.raises(_Boom):
        relax_objective_defining_equality(m)
    assert calls


# ── integer_ratio.py ────────────────────────────────────────────────────────


def test_integer_ratio_node_bound_defect_propagates(monkeypatch):
    """Removed: ``node_bound`` "never raised" -- a defect in the dive became an
    abstention, silently dropping the bound."""
    from discopt._relax.integer_ratio import IntegerRatioPartitioner, detect_integer_ratio_specs

    m = _gear4()
    specs = detect_integer_ratio_specs(m)
    assert specs
    p = IntegerRatioPartitioner(m, specs)
    calls: list[int] = []
    monkeypatch.setattr(p, "_dive", _counting_raiser(lambda: _Boom("dive"), calls))
    lb = np.array([12, 12, 12, 12, 0, 0], dtype=float)
    ub = np.array([60, 60, 60, 60, 1e5, 1e5], dtype=float)
    with pytest.raises(_Boom):
        p.node_bound(lb, ub)
    assert calls


# ── mccormick_subgradient.py ────────────────────────────────────────────────


def _subgrad_model():
    m = dm.Model("sg1520")
    x = m.continuous("x", lb=-1.0, ub=2.0)
    y = m.continuous("y", lb=-1.0, ub=2.0)
    m.minimize(x * y + dm.exp(0.3 * x))
    m.subject_to(x + y >= 0.5)
    return m


def test_reduced_lp_simplex_failure_warns_and_falls_back(monkeypatch, caplog):
    """Kept: the in-house simplex raising re-solves the same Kelley LP on
    scipy/HiGHS; the bound stays valid and a WARNING is emitted."""
    from discopt._relax import milp_relaxation
    from discopt._relax.mccormick_subgradient import reduced_mccormick_lp_bound

    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    monkeypatch.delenv("DISCOPT_REDUCED_LP_BACKEND", raising=False)
    calls: list[int] = []
    monkeypatch.setattr(
        milp_relaxation.MilpRelaxationModel,
        "solve",
        _counting_raiser(lambda: _Boom("simplex"), calls),
    )
    lb, ub = np.array([-1.0, -1.0]), np.array([2.0, 2.0])
    res = reduced_mccormick_lp_bound(_subgrad_model(), lb, ub)
    assert calls
    assert res.status == "optimal"
    # Valid: below the true box optimum (min of x*y + exp(0.3x) with x+y>=0.5 is
    # attained on the box; sample it).
    xs = np.linspace(-1, 2, 61)
    true = min(a * b + np.exp(0.3 * a) for a in xs for b in xs if a + b >= 0.5 - 1e-12)
    assert res.bound <= true + 1e-6
    assert len(_warnings_from(caplog, "reduced-space Kelley LP (in-house simplex) failed")) == 1


def test_reduced_lp_infeasible_crosscheck_failure_refuses_fathom(monkeypatch, caplog):
    """Kept: if the scipy/HiGHS confirmation of a simplex "infeasible" raises, the
    node is NOT fathomed (status ``unsupported``, bound ``-inf``) -- with a WARNING."""
    import scipy.optimize
    from discopt._relax import milp_relaxation
    from discopt._relax.mccormick_subgradient import reduced_mccormick_lp_bound

    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    monkeypatch.delenv("DISCOPT_REDUCED_LP_BACKEND", raising=False)
    monkeypatch.setattr(
        milp_relaxation.MilpRelaxationModel,
        "solve",
        lambda *_a, **_k: types.SimpleNamespace(status="infeasible", objective=None, bound=None),
    )
    calls: list[int] = []
    monkeypatch.setattr(scipy.optimize, "linprog", _counting_raiser(lambda: _Boom("hi"), calls))
    lb, ub = np.array([-1.0, -1.0]), np.array([2.0, 2.0])
    res = reduced_mccormick_lp_bound(_subgrad_model(), lb, ub)
    assert calls
    assert res.status == "unsupported"
    assert res.bound == -np.inf
    assert len(_warnings_from(caplog, "reduced-space infeasibility cross-check")) == 1


# ── shor_sdp.py ─────────────────────────────────────────────────────────────


def test_shor_sdp_solver_failure_warns_and_declines(monkeypatch, caplog):
    """Kept: SCS (optional, native) raising yields no bound, with a WARNING."""
    from discopt._relax.discretization import DiscretizationState
    from discopt._relax.milp_relaxation import (
        build_milp_relaxation,
        sanitize_relaxation_for_conditioning,
    )
    from discopt._relax.model_utils import binary_flat_cols, flat_variable_bounds
    from discopt._relax.shor_sdp import shor_sdp_lower_bound
    from discopt._relax.term_classifier import classify_nonlinear_terms

    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    m = dm.Model("shor1520")
    x = m.binary("x", shape=(3,))
    m.subject_to(x[0] + x[1] + x[2] == 1)
    m.minimize(2.0 * x[0] * x[1] - 3.0 * x[1] * x[2] + x[0] * x[2])
    lb, ub = flat_variable_bounds(m)
    relax, info = build_milp_relaxation(
        m, classify_nonlinear_terms(m), DiscretizationState(), bound_override=(lb, ub)
    )
    relax = sanitize_relaxation_for_conditioning(relax)

    calls: list[int] = []
    fake_scs = types.ModuleType("scs")
    fake_scs.SCS = _counting_raiser(lambda: _Boom("scs"), calls)
    monkeypatch.setitem(sys.modules, "scs", fake_scs)
    bound, dim = shor_sdp_lower_bound(m, relax, info, binary_vars=binary_flat_cols(m))
    assert calls
    assert bound is None
    assert dim == 4
    assert len(_warnings_from(caplog, "Shor SDP (SCS) solve failed")) == 1


# ── nlp_evaluator.py ────────────────────────────────────────────────────────


def _vec_model():
    m = dm.Model("ev1520")
    x = m.continuous("x", shape=(4,), lb=0.1, ub=3.0)
    m.minimize(dm.sum(x * x))
    for i in range(3):
        m.subject_to(x[i] * x[i + 1] <= 2.0)
    return m


def test_nlp_evaluator_sparsity_detection_defect_propagates(monkeypatch):
    """Removed: a raise became ``sparsity_pattern = None`` (silently dense)."""
    from discopt._relax import sparsity
    from discopt._relax.nlp_evaluator import NLPEvaluator

    ev = NLPEvaluator(_vec_model())
    calls: list[int] = []
    monkeypatch.setattr(
        sparsity, "detect_sparsity_dag", _counting_raiser(lambda: _Boom("dag"), calls)
    )
    with pytest.raises(_Boom):
        ev.sparsity_pattern  # noqa: B018 - the property is the call under test
    assert calls


def test_nlp_evaluator_compressed_decision_defect_propagates(monkeypatch):
    """Removed: ``should_use_sparse`` raising defaulted to dense-then-project."""
    from discopt._relax import sparsity
    from discopt._relax.nlp_evaluator import NLPEvaluator

    ev = NLPEvaluator(_vec_model())
    assert ev.sparsity_pattern is not None
    calls: list[int] = []
    monkeypatch.setattr(sparsity, "should_use_sparse", _counting_raiser(lambda: _Boom("s"), calls))
    with pytest.raises(_Boom):
        ev._use_compressed_eval()
    assert calls


@pytest.mark.parametrize("builder", ["_ensure_sparse_jac_fn", "_ensure_sparse_jac_values_fn"])
def test_nlp_evaluator_sparse_jacobian_setup_defect_propagates(monkeypatch, builder):
    """Removed (both Jacobian builders): a coloring failure became a silent switch
    to the dense Jacobian."""
    from discopt._relax import sparsity
    from discopt._relax.nlp_evaluator import NLPEvaluator

    ev = NLPEvaluator(_vec_model())
    assert ev.sparsity_pattern is not None
    monkeypatch.setattr(sparsity, "should_use_sparse", lambda *_a, **_k: True)
    calls: list[int] = []
    monkeypatch.setattr(
        sparsity, "compute_coloring", _counting_raiser(lambda: _Boom("color"), calls)
    )
    with pytest.raises(_Boom):
        getattr(ev, builder)()
    assert calls


def test_nlp_evaluator_sparse_hessian_setup_defect_propagates(monkeypatch):
    """Removed: a Hessian-coloring failure became a silent switch to the dense
    ``n x n`` Hessian."""
    from discopt._relax import sparse_hessian
    from discopt._relax.nlp_evaluator import NLPEvaluator

    ev = NLPEvaluator(_vec_model())
    monkeypatch.setattr(ev, "_use_sparse_hessian", lambda: True)
    calls: list[int] = []
    monkeypatch.setattr(
        sparse_hessian, "build_hessian_coloring", _counting_raiser(lambda: _Boom("hc"), calls)
    )
    with pytest.raises(_Boom):
        ev._ensure_sparse_hess_values_fn()
    assert calls


# ── differentiable.py ───────────────────────────────────────────────────────


def _param_nlp():
    m = dm.Model("sens1520")
    p = m.parameter("p", value=2.0)
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.minimize((x - p) ** 2 + 0.1 * x)
    return m, p


def test_l3_implicit_differentiation_defect_propagates(monkeypatch):
    """Removed: any exception in the active-set / KKT assembly became
    ``l3_failed=True`` ("fallback_to_L1")."""
    from discopt._relax import differentiable as d

    m, _p = _param_nlp()
    calls: list[int] = []
    monkeypatch.setattr(d, "implicit_differentiate", _counting_raiser(lambda: _Boom("kkt"), calls))
    with pytest.raises(_Boom):
        d.differentiable_solve_l3(m)
    assert calls


def test_sensitivity_pounce_fallback_failure_warns(monkeypatch, caplog):
    """Kept: the POUNCE re-solve raising leaves the IPM's non-optimal status to be
    raised (no sensitivity is reported) -- with a WARNING."""
    from discopt._relax import differentiable as d
    from discopt.solvers import SolveStatus

    caplog.set_level(logging.DEBUG, logger="discopt.solver")
    m, _p = _param_nlp()
    calls: list[int] = []

    def fake_dispatch(nlp_solver, evaluator, x0, opts):
        calls.append(1)
        if nlp_solver == "ipm":
            return types.SimpleNamespace(status=SolveStatus.ITERATION_LIMIT)
        raise _Boom("pounce")

    monkeypatch.setattr(d, "_dispatch_nlp_solve", fake_dispatch)
    with pytest.raises(RuntimeError, match="did not converge"):
        d._compute_sensitivity_at_solution(m, {"x": np.asarray(1.95)}, nlp_solver="ipm")
    assert len(calls) == 2
    assert len(_warnings_from(caplog, "POUNCE sensitivity fallback solve failed")) == 1


def test_l3_vector_constraint_keeps_implicit_sensitivities():
    """The defect the removal exposed: ``find_active_set`` called ``float()`` on a
    vector-valued constraint, and the old ``except Exception`` turned the
    ``TypeError`` into ``l3_failed`` -- every model with a vector constraint
    silently lost ``dx_dp``. Active rows are now flat constraint rows."""
    from discopt._relax.differentiable import differentiable_solve_l3

    m = dm.Model("l3vec1520")
    p = m.parameter("p", value=2.0)
    x = m.continuous("x", shape=(2,), lb=-10.0, ub=10.0)
    m.minimize(dm.sum((x - p) ** 2))
    # Vector-valued, with the first row active at the optimum (x0 = 1 < p).
    m.subject_to(x <= np.array([1.0, 5.0]))
    r = differentiable_solve_l3(m)
    assert r._l3_failed is False
    assert r._dx_dp is not None
    dx = np.asarray(r._dx_dp).reshape(2, -1)[:, 0]
    # x0 is pinned by the active row (dx0/dp = 0); x1 = p is interior (dx1/dp = 1).
    assert dx[0] == pytest.approx(0.0, abs=1e-6)
    assert dx[1] == pytest.approx(1.0, abs=1e-6)

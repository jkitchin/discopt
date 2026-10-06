"""The named transformation framework and the model copy it runs on (#1479).

Two things are under test:

* ``Model.__deepcopy__`` / ``Model.clone`` -- before #1479 ``copy.deepcopy`` raised
  on a ``from_nl`` model (``PyModelRepr``), a solved model (a POUNCE module in
  the evaluator cache) and a fast-API model (``PyModelBuilder``). The first three
  tests below fail on the pre-#1479 tree for exactly those reasons.
* ``discopt.transformations`` -- every registered entry must be the *same*
  function object the solver calls (the framework wraps; it reimplements
  nothing; the solver's call sites -- the AMP route's GDP lowering included --
  dispatch through it), ``create_using`` must never touch its input, and ``apply_to``
  must leave a model whose variables all report it as their owner.
"""

from __future__ import annotations

import copy
import pathlib

import discopt.modeling as dm
import discopt.transformations as dt
import numpy as np
import pytest
from discopt import mpec
from discopt._relax import (
    binary_multilinear_reform,
    gdp_reformulate,
    integer_product_reform,
    nonsmooth_lift,
)
from discopt.modeling import _relax_integrality
from discopt.modeling.core import from_nl

CORPUS = sorted((pathlib.Path(__file__).parent / "data" / "minlplib_nl").glob("*.nl"))


def _gdp_model():
    # The #1430 probe: optimum 5.0; dropping the disjunction gives 0.0.
    m = dm.Model("gdp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x + y)
    m.either_or([[x >= 4, y >= 1], [y >= 6, x >= 1]])
    return m


def _mpcc():
    m = dm.Model("mpcc")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    return m, mpec.Complementarity(x, y)


def _fast_api_model():
    m = dm.Model("fast")
    x = m.continuous("x", shape=(2,), lb=0, ub=4)
    m.add_linear_objective(np.array([1.0, 2.0]), x)
    m.add_linear_constraints(np.array([[1.0, 1.0]]), x, ">=", np.array([3.0]))
    return m


# ── Model copy: the three states that used to raise ──────────────────────


def test_deepcopy_of_a_from_nl_model():
    assert CORPUS, "in-repo .nl corpus missing"
    checked = 0
    for path in CORPUS:
        m = from_nl(str(path))
        c = copy.deepcopy(m)
        assert dt.model_fingerprint(c) == dt.model_fingerprint(m), path.name
        assert all(v.model is c for v in c._variables), path.name
        assert getattr(c, "_source_nl_path", None) == getattr(m, "_source_nl_path", None)
        checked += 1
    assert checked == len(CORPUS)


def test_deepcopy_of_a_solved_model_drops_solve_caches():
    m = dm.Model("solved")
    x = m.continuous("x", lb=0, ub=4)
    y = m.integer("y", lb=0, ub=3)
    m.minimize((x - 1.3) ** 2 + y)
    m.subject_to(x + y >= 2)
    ref = m.solve(time_limit=30)
    assert any(k.endswith("_cache") and k != "_flat_var_offsets_cache" for k in vars(m))

    c = m.clone()
    assert not [k for k in vars(c) if k.endswith("_cache") and k != "_flat_var_offsets_cache"]
    assert c._flat_var_offsets_cache is None
    assert dt.model_fingerprint(c) == dt.model_fingerprint(m)
    assert c.solve(time_limit=30).objective == pytest.approx(ref.objective, abs=1e-6)


def test_deepcopy_of_a_fast_api_model_replays_the_builder():
    m = _fast_api_model()
    c = copy.deepcopy(m)
    assert c._builder is not None and c._builder is not m._builder
    assert c.num_constraints == m.num_constraints == 1
    assert dt.model_fingerprint(c) == dt.model_fingerprint(m)
    assert c.solve(time_limit=10).objective == pytest.approx(3.0, abs=1e-6)
    assert m.solve(time_limit=10).objective == pytest.approx(3.0, abs=1e-6)


def test_deepcopy_of_a_fresh_model_is_the_default_copy():
    m = _gdp_model()
    c = copy.deepcopy(m)
    assert set(vars(c)) == set(vars(m))
    assert dt.model_fingerprint(c) == dt.model_fingerprint(m)
    # Independent: the copy's variables are new objects, and #1332's read-only
    # bound arrays survive the copy.
    assert all(cv is not mv for cv, mv in zip(c._variables, m._variables))
    assert not c._variables[0].lb.flags.writeable
    c._variables[0].lb = 3.0
    assert float(m._variables[0].lb) == 0.0


def test_deepcopy_names_an_uncopyable_attribute():
    m = _gdp_model()
    m._not_copyable = np  # a module: deepcopy cannot copy it
    with pytest.raises(TypeError, match="_not_copyable"):
        copy.deepcopy(m)


# ── Registry: wraps the solver's functions, reimplements nothing ──────────


EXPECTED = {
    "gdp": gdp_reformulate.reformulate_gdp,
    "gdp.bigm": gdp_reformulate.reformulate_gdp,
    "gdp.hull": gdp_reformulate.reformulate_gdp,
    "gdp.mbigm": gdp_reformulate.reformulate_gdp,
    "gdp.auto": gdp_reformulate.reformulate_gdp,
    "gdp.simplex": gdp_reformulate.reformulate_gdp,
    "integer.bilinear": integer_product_reform.reformulate_integer_bilinear,
    "integer.multilinear": integer_product_reform.reformulate_integer_multilinear,
    "binary.multilinear": binary_multilinear_reform.reformulate_binary_multilinear,
    "mpec.gdp": mpec.reformulate_gdp,
    "mpec.sos1": mpec.reformulate_sos1,
    "mpec.scholtes": mpec.reformulate_scholtes,
    "nonsmooth.epigraph": nonsmooth_lift.lift_nonsmooth_atoms,
    "core.relax_integrality": _relax_integrality.relax_integrality,
}


def test_every_registered_name_is_the_solvers_own_function():
    assert dt.available() == sorted(EXPECTED)
    for name, fn in EXPECTED.items():
        assert dt.get(name).function is fn, name
    assert dt.TransformationFactory is dt.get
    assert not dt.get("mpec.scholtes").exact
    assert not dt.get("core.relax_integrality").exact


@pytest.mark.parametrize("method", ["big-m", "hull", "mbigm", "auto"])
@pytest.mark.parametrize("respect", [True, False])
def test_apply_is_exactly_the_call_site_call(method, respect):
    """``respect=False`` is the AMP route's call (``solver.py``: ``reformulate_gdp(
    model, method=amp_gdp_method, respect_disjunction_methods=False)``)."""
    direct = gdp_reformulate.reformulate_gdp(
        _gdp_model(), method=method, respect_disjunction_methods=respect
    )
    via = dt.get("gdp").apply(_gdp_model(), method=method, respect_disjunction_methods=respect)
    assert dt.model_fingerprint(via) == dt.model_fingerprint(direct)


# ── create_using / apply_to ───────────────────────────────────────────────


@pytest.mark.parametrize("name", ["gdp", "gdp.bigm", "gdp.hull", "gdp.mbigm", "gdp.auto"])
def test_gdp_create_using_leaves_the_input_alone_and_solves_the_same(name):
    m = _gdp_model()
    report = dt.check(name, m)
    r = report.result
    assert report.diff.removed_constraints and report.diff.added_constraints
    assert report.diff.added_variables
    assert all(v.model is r for v in r._variables)
    assert not {id(v) for v in r._variables} & {id(v) for v in m._variables}
    assert r.solve(time_limit=30).objective == pytest.approx(5.0, abs=1e-5)


def test_apply_to_a_functional_transformation_adopts_the_result():
    m = _gdp_model()
    x = m._variables[0]
    returned = dt.apply_to("gdp.hull", m)
    assert returned is m
    assert all(v.model is m for v in m._variables)
    assert any(v is x for v in m._variables)  # the caller's handles still work
    with m.fixed({x: 4.0}):  # _resolve_variable's ownership check passes
        pass
    assert m.solve(time_limit=30).objective == pytest.approx(5.0, abs=1e-5)


def test_the_raw_functional_result_does_not_own_its_source_variables():
    """Why ``apply_to`` adopts: the pass's own return value shares the input's
    variables, which still name the input as their model."""
    m = _gdp_model()
    r = dt.get("gdp").apply(m)
    assert r is not m
    assert any(v.model is m for v in r._variables)
    assert any(v.model is r for v in r._variables)


def test_integer_and_binary_product_transformations():
    m = dm.Model("ib")
    a = m.integer("a", lb=0, ub=5)
    b = m.continuous("b", lb=0, ub=3)
    m.minimize(-a * b + a)
    m.subject_to(a * b <= 7)
    for name in ("integer.bilinear", "integer.multilinear"):
        report = dt.check(name, m)
        assert report.diff.added_variables, name
        assert report.result.solve(time_limit=30).objective == pytest.approx(-4.0, abs=1e-5)

    m = dm.Model("bm")
    z = m.binary("z", shape=(3,))
    m.minimize(-z[0] * z[1] * z[2] + z[0])
    m.subject_to(z[0] + z[1] >= 1)
    report = dt.check("binary.multilinear", m)
    assert report.diff.added_variables
    assert report.result.solve(time_limit=30).objective == pytest.approx(0.0, abs=1e-6)


def test_a_no_op_transformation_returns_an_unchanged_copy():
    m = _gdp_model()
    report = dt.check("binary.multilinear", m)  # nothing multilinear here
    assert report.diff.unchanged
    assert dt.model_fingerprint(report.result) == dt.model_fingerprint(m)


@pytest.mark.parametrize("name", ["mpec.gdp", "mpec.sos1"])
def test_mpec_create_using_maps_pairs_onto_the_copy(name):
    m, pair = _mpcc()
    report = dt.check(name, m, pairs=[pair])
    r = report.result
    assert report.diff.added_constraints
    assert m._constraints == [] and not m._lowered_complementarities
    assert not pair.is_lowered_into(m)
    # The copy's rows are over the copy's variables, not the original's.
    (lowered,) = r._lowered_complementarities
    assert lowered is not pair and lowered.is_lowered_into(r)
    assert r.solve(time_limit=30).objective == pytest.approx(1.0, abs=1e-3)


def test_mpec_scholtes_is_registered_as_inexact_and_maps_its_parameter():
    m, pair = _mpcc()
    t = m.parameter("t", 0.01)
    report = dt.check("mpec.scholtes", m, pairs=[pair], t=t)
    (tc,) = report.result._parameters
    assert tc is not t and tc.model is report.result
    assert report.result.solve(time_limit=30).objective == pytest.approx(0.98, abs=1e-3)


def test_apply_to_in_place_clears_the_source_nl_path_only_on_change():
    """A transformed model is no longer the ``.nl`` file; the solver must not
    hand POUNCE that file for it (``nlp_native.build_native_base``)."""
    path = next(p for p in CORPUS if hasattr(from_nl(str(p)), "_source_nl_path"))
    m = from_nl(str(path))

    unchanged = dt.create_using("mpec.gdp", m, pairs=[])
    assert unchanged._source_nl_path == m._source_nl_path

    changed = m.clone()
    w1 = changed.continuous("_t1479_w1", lb=0, ub=1)
    w2 = changed.continuous("_t1479_w2", lb=0, ub=1)
    dt.apply_to("mpec.sos1", changed, pairs=[mpec.Complementarity(w1, w2)])
    assert not hasattr(changed, "_source_nl_path")
    assert m._source_nl_path  # the original keeps its own


# ── Refusals ──────────────────────────────────────────────────────────────


def test_fixed_option_clash_unknown_name_and_duplicate_registration_refuse():
    with pytest.raises(TypeError, match="fixes method"):
        dt.create_using("gdp.hull", _gdp_model(), method="big-m")
    with pytest.raises(KeyError, match="gdp.hull"):
        dt.get("gdp.nonsense")
    with pytest.raises(ValueError, match="already registered"):
        dt.register("gdp", lambda m: m, style="functional", summary="dup")


def test_adoption_refuses_a_result_with_a_foreign_owner():
    keep = []

    def leaky(model):
        out = gdp_reformulate.reformulate_gdp(model)
        keep.append(out)  # a second, live owner the adoption cannot re-point
        return out

    t = dt.Transformation("test.leaky", leaky, "functional", "test only")
    m = _gdp_model()
    before = dt.model_fingerprint(m)
    with pytest.raises(TypeError, match="cannot adopt"):
        t.apply_to(m)
    assert dt.model_fingerprint(m) == before  # refused before touching m


def test_style_mismatch_is_refused():
    t = dt.Transformation("test.bad", lambda m: None, "functional", "test only")
    with pytest.raises(TypeError, match="not a Model"):
        t.apply(_gdp_model())
    t = dt.Transformation("test.bad2", lambda m: m, "in_place", "test only")
    with pytest.raises(TypeError, match="registered in-place"):
        t.apply(_gdp_model())


# ── The solver's call sites dispatch through the registry ─────────────────


@pytest.fixture
def dispatched(monkeypatch):
    calls: list[tuple[str, dict]] = []
    orig = dt.Transformation.apply

    def counted(self, model, **options):
        calls.append((self.name, dict(options)))
        return orig(self, model, **options)

    monkeypatch.setattr(dt.Transformation, "apply", counted)
    return calls


def test_default_solve_lowers_gdp_through_the_registry(dispatched):
    r = _gdp_model().solve(time_limit=30)
    assert r.objective == pytest.approx(5.0, abs=1e-5)
    assert ("gdp", {"method": "big-m"}) in dispatched


def test_amp_route_lowers_gdp_through_the_registry_with_its_own_options(dispatched):
    r = _gdp_model().solve(solver="amp", time_limit=30)
    assert r.objective == pytest.approx(5.0, abs=1e-5)
    amp = [o for n, o in dispatched if n == "gdp"]
    assert {"method": "big-m", "respect_disjunction_methods": False} in amp


def test_integer_product_pass_goes_through_the_registry(dispatched):
    m = dm.Model("ib")
    a = m.integer("a", lb=0, ub=5)
    c = m.integer("c", lb=0, ub=5)
    m.minimize(-(a * c) + 2 * a + c)
    m.subject_to(a * c <= 10)
    m.solve(time_limit=30)
    names = {n for n, _ in dispatched}
    assert names & {"integer.bilinear", "integer.multilinear", "binary.multilinear"}


def test_model_complementarity_lowers_through_the_registry(dispatched):
    m = dm.Model("mp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.complementarity(x, y, method="sos1")
    assert [n for n, _ in dispatched] == ["mpec.sos1"]


def test_a_module_monkeypatch_still_reaches_the_call_site(monkeypatch):
    """The registry resolves ``module:attr`` per call, so patching the module
    attribute (what existing tests do) still intercepts the solver's call."""
    seen = []
    real = gdp_reformulate.reformulate_gdp

    def spy(model, *args, **kwargs):
        seen.append(kwargs)
        return real(model, *args, **kwargs)

    monkeypatch.setattr(gdp_reformulate, "reformulate_gdp", spy)
    _gdp_model().solve(time_limit=30)
    assert seen


# ── #1497: the fingerprint and the diff see every coefficient ────────────


def _coef_model(rhs, a):
    m = dm.Model("m")
    x = m.continuous("x", shape=(3,), lb=0, ub=10)
    m.minimize(np.array(a) @ x)
    m.subject_to(x[0] + x[1] + x[2] >= rhs)
    return m


def test_fingerprint_sees_a_seventh_digit_coefficient_change():
    """A scalar ``Constant`` displays with ``.6g``; the fingerprint must not."""
    m1, m2 = _coef_model(1.0000001, [1.0, 2.0, 3.0]), _coef_model(1.0000004, [1.0, 2.0, 3.0])
    assert dt.model_fingerprint(m1) != dt.model_fingerprint(m2)
    d = dt.diff_models(m1, m2)
    assert not d.unchanged
    assert len(d.added_constraints) == 1 and "1.0000004" in d.added_constraints[0]
    assert len(d.removed_constraints) == 1 and "1.0000001" in d.removed_constraints[0]
    assert not d.objective_changed


def test_fingerprint_sees_one_element_of_an_array_constant():
    """An array ``Constant`` displays as its shape; the fingerprint must see its data."""
    m1, m2 = _coef_model(1.0, [1.0, 2.0, 3.0]), _coef_model(1.0, [1.0, 2.0, 3.5])
    assert dt.model_fingerprint(m1) != dt.model_fingerprint(m2)
    d = dt.diff_models(m1, m2)
    assert d.objective_changed and not d.added_constraints and not d.removed_constraints


def test_fingerprint_sees_a_change_inside_a_large_array_and_a_sum_over():
    def build(k, bump):
        m = dm.Model("big")
        x = m.continuous("x", shape=(100,), lb=0, ub=1)
        c = np.linspace(0.0, 1.0, 100)
        c[k] += bump
        m.minimize(c @ x)
        m.subject_to(dm.sum(lambda i: (1.0 + bump * (i == k)) * x[i], over=range(3)) <= 2)
        return m

    base = dt.model_fingerprint(build(57, 0.0))
    assert dt.model_fingerprint(build(57, 0.0)) == base
    assert dt.model_fingerprint(build(57, 1e-12)) != base


def test_fingerprint_equal_for_a_deep_copy_with_array_constants():
    m = _coef_model(1.0000001, [1.0, 2.0, 3.0])
    assert dt.model_fingerprint(copy.deepcopy(m)) == dt.model_fingerprint(m)
    assert dt.diff_models(m, copy.deepcopy(m)).unchanged


def test_apply_to_detects_a_pass_that_only_changes_coefficients(monkeypatch):
    """``apply_to`` clears the source records exactly when the fingerprint moves;
    a coefficient-only mutation must count as a change."""

    def rescale(model):
        model._objective.expression.left.value = np.array([9.0, 9.0, 9.0])
        return model

    t = dt.Transformation("test.rescale", rescale, "functional", "mutates coefficients")
    monkeypatch.setitem(dt._REGISTRY, "test.rescale", t)
    m = _coef_model(1.0, [1.0, 2.0, 3.0])
    m._source_nl_path = "/nonexistent/source.nl"
    before = dt.model_fingerprint(m)
    dt.apply_to("test.rescale", m)
    assert dt.model_fingerprint(m) != before
    assert "_source_nl_path" not in m.__dict__


# ── #1498: rebuilding passes keep the model's validation guards ──────────


def _piecewise_sos2_model():
    m = dm.Model("s")
    x = m.continuous("x", lb=0, ub=4)
    m.piecewise(x, [0, 1, 2, 4], [0, 3, 1, 5], method="sos2", name="y")
    m.maximize(x)
    return m


def _widen_x(model, ub=6.0):
    next(v for v in model._variables if v.name == "x").ub = ub


@pytest.mark.parametrize("name", ["gdp", "gdp.bigm", "gdp.hull", "gdp.auto"])
def test_gdp_copy_of_a_piecewise_model_still_refuses_a_widened_bound(name):
    """Pre-#1498 the lowered copy had no ``_piecewise_domains`` and solved a
    widened ``x <= 6`` to 4.0, the breakpoint span, instead of refusing."""
    from discopt.modeling._piecewise import PiecewiseDomainError

    m = _piecewise_sos2_model()
    out = dt.create_using(name, m)
    assert out is not m
    assert len(out._piecewise_domains) == len(m._piecewise_domains) == 1
    _widen_x(out)
    with pytest.raises(PiecewiseDomainError):
        out.validate()


def test_apply_to_keeps_the_piecewise_guard():
    from discopt.modeling._piecewise import PiecewiseDomainError

    m = _piecewise_sos2_model()
    dt.apply_to("gdp", m)
    assert not any(type(c).__name__ == "_SOSConstraint" for c in m._constraints)
    _widen_x(m)
    with pytest.raises(PiecewiseDomainError):
        m.validate()


def _guarded_model(nonlinear: bool):
    """A model carrying one piecewise domain (and, if nonlinear, one atan2
    precondition) plus the structure each rebuilding pass acts on."""
    m = dm.Model("g")
    x = m.continuous("x", lb=0, ub=4)
    a = m.integer("a", lb=0, ub=5)
    c = m.integer("c", lb=0, ub=5)
    b = m.binary("b")
    d = m.binary("d")
    f = m.binary("f")
    y = m.piecewise(x, [0, 1, 2, 4], [0, 3, 1, 5], method="sos2", name="y")
    m.subject_to(b * d * f <= 0.5)
    if nonlinear:
        q = m.continuous("q", lb=0.5, ub=3)
        r = m.continuous("r", lb=-2, ub=2)
        m.subject_to(dm.atan2(r, q) <= 1.0)
        m.subject_to(a * c <= 10)
        e = m.continuous("e", lb=0.1, ub=2)
        m.subject_to(x / q + e * dm.log(e) <= 20)
        assert len(m._atan2_preconditions) == 1
    m.minimize(-y - x + a + c)
    assert len(m._piecewise_domains) == 1
    return m


def _assert_guards_carried(src, out, label):
    assert out is not src, f"{label} did not rebuild the model; the probe is void"
    for attr in ("_atan2_preconditions", "_piecewise_domains"):
        want = [id(t) for t in getattr(src, attr)]
        assert [id(t) for t in getattr(out, attr)] == want, (label, attr)


def test_every_rebuilding_pass_carries_atan2_and_piecewise_guards():
    """Each pass that rebuilds a model into a fresh ``Model`` must forward both
    guard lists (#1498). Every pass must actually rebuild here, or the probe
    would pass vacuously (CLAUDE.md §6)."""
    from discopt._relax.factorable_reform import canonicalize_entropy, factorable_reformulate

    fired = 0
    m = _guarded_model(nonlinear=True)
    for name in ("gdp", "integer.bilinear", "integer.multilinear"):
        _assert_guards_carried(m, dt.get(name).apply(m), name)
        fired += 1
    for fn in (factorable_reformulate, canonicalize_entropy):
        _assert_guards_carried(m, fn(m), fn.__name__)
        fired += 1
    # binary.multilinear only fires on a pure MILP without SOS records: lower first.
    milp = dt.get("gdp").apply(_guarded_model(nonlinear=False))
    _assert_guards_carried(milp, dt.get("binary.multilinear").apply(milp), "binary.multilinear")
    fired += 1
    assert fired == 6


def test_a_pass_that_drops_a_guard_is_refused(monkeypatch):
    def rebuild_without_guards(model):
        out = dm.Model(model.name)
        out._variables = list(model._variables)
        out._rebuild_name_index()
        out._constraints = list(model._constraints)
        out._objective = model._objective
        return out

    t = dt.Transformation("test.drop", rebuild_without_guards, "functional", "drops guards")
    monkeypatch.setitem(dt._REGISTRY, "test.drop", t)
    with pytest.raises(RuntimeError, match="_piecewise_domains"):
        dt.create_using("test.drop", _piecewise_sos2_model())


# ── #1497 (related): a crashing pass must not read as "unchanged" ─────────


def _int_bilinear_model():
    m = dm.Model("ib")
    a = m.integer("a", lb=0, ub=5)
    c = m.integer("c", lb=0, ub=5)
    m.minimize(-(a * c) + 2 * a + c)
    m.subject_to(a * c <= 10)
    return m


def _bin_trilinear_model():
    m = dm.Model("bt")
    b = m.binary("b", shape=(3,))
    m.minimize(-(b[0] * b[1] * b[2]) + b[0])
    return m


def _boom(*args, **kwargs):
    raise RuntimeError("injected defect")


@pytest.mark.parametrize(
    "name, module, attr, build",
    [
        ("integer.bilinear", integer_product_reform, "_rewrite", _int_bilinear_model),
        ("integer.multilinear", integer_product_reform, "_rewrite", _int_bilinear_model),
        (
            "integer.bilinear",
            integer_product_reform,
            "has_integer_product_work",
            _int_bilinear_model,
        ),
        ("binary.multilinear", binary_multilinear_reform, "_process_body", _bin_trilinear_model),
    ],
)
def test_a_crash_inside_a_pass_propagates_through_check(monkeypatch, name, module, attr, build):
    """Pre-fix, the passes' ``except Exception: return model`` turned an injected
    defect into ``check(...).diff.unchanged is True``."""
    m = build()
    # The unpatched pass does act on this model, so "unchanged" would be a lie.
    assert not dt.check(name, m).diff.unchanged
    monkeypatch.setattr(module, attr, _boom)
    with pytest.raises(RuntimeError, match="injected defect"):
        dt.check(name, m)


def test_binary_multilinear_gate_does_not_swallow_a_scan_crash(monkeypatch):
    assert binary_multilinear_reform.has_binary_multilinear_work(_bin_trilinear_model())
    monkeypatch.setattr(binary_multilinear_reform, "_witness_scan", _boom)
    with pytest.raises(RuntimeError, match="injected defect"):
        binary_multilinear_reform.has_binary_multilinear_work(_bin_trilinear_model())


def test_documented_abstention_still_returns_the_input_model(monkeypatch):
    """An unbounded continuous factor is the #286 "cannot big-M" case: the pass
    abstains (``IntegerProductNotApplicable``) and returns the model unchanged."""
    m = dm.Model("unb")
    a = m.integer("a", lb=0, ub=5)
    x = m.continuous("x", lb=0, ub=1e30)
    m.minimize(a * x - a)
    m.subject_to(a * x <= 3)
    assert integer_product_reform.has_integer_product_work(m)

    abstained = []
    real = integer_product_reform._Expander.bigm_product

    def spy(self, *args, **kwargs):
        try:
            return real(self, *args, **kwargs)
        except integer_product_reform.IntegerProductNotApplicable:
            abstained.append(True)
            raise

    monkeypatch.setattr(integer_product_reform._Expander, "bigm_product", spy)
    for name in ("integer.bilinear", "integer.multilinear"):
        abstained.clear()
        assert dt.get(name).apply(m) is m
        assert abstained, f"{name} did not reach the documented abstention"
        assert dt.check(name, m).diff.unchanged

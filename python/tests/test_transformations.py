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
from discopt._relax import binary_multilinear_reform, gdp_reformulate, integer_product_reform
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
}


def test_every_registered_name_is_the_solvers_own_function():
    assert dt.available() == sorted(EXPECTED)
    for name, fn in EXPECTED.items():
        assert dt.get(name).function is fn, name
    assert dt.TransformationFactory is dt.get
    assert not dt.get("mpec.scholtes").exact


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

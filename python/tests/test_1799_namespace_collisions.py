"""#1799 follow-up: a user variable named inside a pass's generated namespace.

#1823 made the reformulation passes draw their names through
``Model._fresh_name`` (and factorable's ``_new_aux_name`` / gdp's selector loops
check ``Model._names``), but no test declared a user variable whose name a pass
would otherwise generate, so every one of those checks could be deleted with the
suite still green. Before #1823 such a model solved WRONG, measured on the
pre-#1823 base: ``_ipx_e0`` returned objective 6.0, uncertified, with the user's
``u`` at -2.5e-14 outside its box [1, 2] (the optimum is 7.0 at u=1);
``_fr_aux_0`` returned a feasible -2.80, uncertified (the optimum is -12.8918);
``_hull_anon_v_x_0`` came back uncertified; ``_bml_z0`` produced two variables
of that name.

Each case is checked twice: at the pass (no duplicate name, the user's variable
still the one carrying its name), and through ``Model.solve`` (certified optimum
equal to the known value, the user's variable inside its box) -- the solve-level
check is the one the measured wrong answers came from.

Also here: ``SolveResult.value`` on a variable a pass substituted out
(``AffineDecisionRule.apply``) is refused, instead of resolving through the
freed name to a new variable that reuses it.
"""

from __future__ import annotations

import collections
import json
import warnings

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax import binary_multilinear_reform as bml
from discopt._relax import factorable_reform as fr
from discopt._relax import integer_product_reform as ipx
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt.ro.affine_policy import AffineDecisionRule
from discopt.serialize import dumps, loads

pytestmark = pytest.mark.smoke


def _hull():
    m = dm.Model("hull_user_name")
    x = m.continuous("x", lb=0, ub=10)
    u = m.continuous("_hull_anon_v_x_0", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]])
    m.subject_to(u >= 3)
    m.minimize(x + u)
    return m, u


def _ipx():
    m = dm.Model("ipx_user_name")
    x = m.integer("x", lb=0, ub=5)
    y = m.integer("y", lb=0, ub=5)
    u = m.continuous("_ipx_e0", lb=1, ub=2)
    m.subject_to(x * y >= 7)
    m.minimize(x + y + u)
    return m, u


def _bml():
    m = dm.Model("bml_user_name")
    a, b, c = m.binary("a"), m.binary("b"), m.binary("c")
    u = m.continuous("_bml_z0", lb=1, ub=2)
    m.subject_to(a * b + b * c >= 1)
    m.minimize(a + 2 * b + c - a * c + u)
    return m, u


def _fr():
    m = dm.Model("fr_user_name")
    x = m.continuous("x", lb=0.5, ub=3)
    y = m.continuous("y", lb=0.5, ub=3)
    u = m.continuous("_fr_aux_0", lb=1, ub=2)
    m.subject_to(dm.exp(x * y) + (x * y) ** 2 <= 30)
    m.minimize(-((x * y) ** 3) / (1 + x) + u)
    return m, u


#: name -> (builder, the pass that generates into that namespace, solve kwargs,
#: certified optimum). Optima: hull x=0, u=3; ipx x*y>=7 over integers <=5 is
#: x+y=6, plus u=1; bml b=1, a=1, c=0, u=1; factorable measured on #1823 (the
#: pre-#1823 base returned an uncertified -2.80 here).
CASES = {
    "_hull_anon_v_x_0": (_hull, lambda m: reformulate_gdp(m, "hull"), {"gdp_method": "hull"}, 3.0),
    "_ipx_e0": (_ipx, ipx.expand_integer_products, {}, 7.0),
    "_bml_z0": (_bml, bml.reformulate_binary_multilinear, {}, 4.0),
    "_fr_aux_0": (_fr, fr.factorable_reformulate, {}, -12.891775337944784),
}


def _duplicates(model):
    names = [v.name for v in model._variables] + [p.name for p in model._parameters]
    return sorted(n for n, k in collections.Counter(names).items() if k > 1)


@pytest.mark.parametrize("name", sorted(CASES))
def test_pass_keeps_a_user_name_in_its_namespace_unique(name):
    build, reform, _, _ = CASES[name]
    m, u = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = reform(m)
    if isinstance(r, tuple):
        r = r[0]
    assert not _duplicates(r)
    # The pass generated into the namespace (else the case tests nothing) ...
    generated = [v.name for v in r._variables if v.name not in {w.name for w in m._variables}]
    assert generated, f"{reform} generated no variable on this case"
    # ... and exactly one variable still carries the user's name, with its box.
    carriers = [v for v in r._variables if v.name == name]
    assert len(carriers) == 1
    np.testing.assert_array_equal(carriers[0].lb, u.lb)
    np.testing.assert_array_equal(carriers[0].ub, u.ub)


@pytest.mark.parametrize("name", sorted(CASES))
def test_solve_with_a_user_name_in_a_pass_namespace(name):
    build, _, kwargs, optimum = CASES[name]
    m, u = build()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = m.solve(time_limit=60, **kwargs)
    assert s.status == "optimal", (s.status, s.objective)
    assert s.gap_certified
    assert s.objective == pytest.approx(optimum, rel=1e-4, abs=1e-6)
    val = float(s.value(u))
    assert float(u.lb) - 1e-6 <= val <= float(u.ub) + 1e-6, val


def test_gdp_selector_skips_a_parameter_carrying_its_name():
    """The selector loops check ``Model._names``, not only the variables.

    ``Model._check_name`` keeps users out of ``_gdp_aux_``, but a loaded document
    may carry that prefix (a lowered model's selectors must round-trip, so the
    loader does not refuse it) -- and on a *parameter* the name is invisible to a
    loop that looked only at variable names.
    """
    m = dm.Model("g")
    x = m.continuous("x", lb=0, ub=10)
    m.parameter("p", value=1.0)
    m.either_or([[x <= 2], [x >= 5]])
    m.minimize(x)
    doc = json.loads(dumps(m))
    taken = "_gdp_aux_disj_anon_0_0"  # what the first selector is named, measured
    for p in doc["parameters"]:
        p["name"] = taken
    m2 = loads(json.dumps(doc))
    assert taken in {p.name for p in m2._parameters}
    r = reformulate_gdp(m2, "big-m")
    assert not _duplicates(r)
    assert taken not in {v.name for v in r._variables}


def _affine_model_with_reused_name():
    m = dm.Model("ro")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    p = m.parameter("p", value=1.0)
    m.subject_to(x + y >= 2 + p)
    m.minimize(x + 2 * y)
    AffineDecisionRule(y, [p]).apply()
    y_new = m.continuous("y", lb=7, ub=8)
    return m, x, y, y_new


def test_value_refuses_a_variable_substituted_out_before_the_solve():
    """``value(old_y)`` returned the value of the NEW ``y`` that reused the freed name."""
    m, x, y, y_new = _affine_model_with_reused_name()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = m.solve(time_limit=20)
    assert 7.0 - 1e-6 <= float(s.value(y_new)) <= 8.0 + 1e-6  # the new y, unconstrained
    s.value(x)  # a live variable of the solved model still resolves
    with pytest.raises(KeyError, match="no longer a variable of the solved model"):
        s.value(y)


@pytest.mark.requires_jax
def test_differentiable_value_refuses_a_substituted_variable():
    """``DiffSolveResult.value`` had the same name-only lookup."""
    from discopt._relax.differentiable import differentiable_solve

    m, _, y, y_new = _affine_model_with_reused_name()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = differentiable_solve(m)
    assert 7.0 - 1e-6 <= np.ravel(r.value(y_new))[0] <= 8.0 + 1e-6
    with pytest.raises(KeyError, match="no longer a variable of the solved model"):
        r.value(y)


def test_value_by_name_still_serves_other_handles():
    """Strings-by-name handles and a clone's source variables keep resolving."""
    m = dm.Model("c")
    x = m.continuous("x", lb=1, ub=3)
    m.minimize(x)
    c = m.clone()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        s = c.solve(time_limit=20)
    assert float(s.value(x)) == pytest.approx(1.0)  # x belongs to m, not c
    other = dm.Model("o").continuous("nope", lb=0, ub=1)
    with pytest.raises(KeyError, match="No variable named 'nope'"):
        s.value(other)

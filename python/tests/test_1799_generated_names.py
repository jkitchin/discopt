"""#1799: reformulation passes register the names of the variables they create.

The passes appended to ``Model._variables`` without touching ``Model._names``, so a
transformed model accepted a second variable with an auxiliary's name;
``serialize.dumps`` wrote it and ``loads`` refused the document. Measured on main
over the serialize builder cases: 12 of 56 transformed models had names missing
from ``_names`` (or, for ``ro:affine_policy``, a stale one), and two unnamed
disjunctions under hull produced the *same* copy name ``_hull_anon_v_x_0`` twice.

After: each generated variable goes through ``Model._register_generated_variable``
with a name from ``Model._fresh_name``, so names stay unique and registered, and a
user declaration that clashes is refused where it is made.
"""

from __future__ import annotations

import collections
import warnings

import discopt.modeling as dm
import discopt.transformations as dt
import pytest
import serialize_builder_cases as cases
from discopt.serialize import dumps, loads

pytestmark = pytest.mark.smoke


def _two_unnamed_hull_disjunctions():
    m = dm.Model("two_anon")
    x = m.continuous("x", lb=0, ub=10)
    m.either_or([[x <= 2], [x >= 5]])
    m.either_or([[x <= 3], [x >= 7]])
    m.minimize(x)
    return dt.create_using("gdp.hull", m)


#: Every transformed-model case in the serialize tables, plus the duplicate shape.
CASES = {
    **{k: v for k, v in cases.ROUND_TRIPS.items() if k.startswith(("transform:", "ro:"))},
    **{k: r.build for k, r in cases.REFUSED.items() if k.startswith("transform:")},
    "gdp.hull: two unnamed disjunctions": _two_unnamed_hull_disjunctions,
}


def _build(key):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = CASES[key]()
    return m[0] if isinstance(m, tuple) else m


def _names(m):
    return [v.name for v in m._variables] + [p.name for p in m._parameters]


@pytest.mark.parametrize("key", sorted(CASES))
def test_names_registry_matches_the_model(key):
    """``_names`` is exactly the model's variable and parameter names, all distinct."""
    m = _build(key)
    names = _names(m)
    dups = [n for n, c in collections.Counter(names).items() if c > 1]
    assert not dups, dups
    assert m._names == set(names)


ADDS_VARIABLES = [
    k
    for k in sorted(CASES)
    if k.startswith("transform:") and k not in ("transform:core.relax_integrality",)
]


@pytest.mark.parametrize("key", ADDS_VARIABLES)
def test_declaring_a_generated_name_is_refused(key):
    """The issue's repro, for every transformation that adds variables: the
    clash is refused at the declaration, not when the document is loaded."""
    m = _build(key)
    generated = [v.name for v in m._variables if v.name.startswith("_")]
    if not generated:
        pytest.skip(f"{key} adds no variable on this case")
    with pytest.raises(ValueError, match="already used|reserved"):
        m.continuous(generated[0], lb=0, ub=1)


def test_two_unnamed_hull_disjunctions_round_trip():
    """Was two variables named ``_hull_anon_v_x_0``: ``loads`` refused the document."""
    m = _two_unnamed_hull_disjunctions()
    assert {"_hull_anon_v_x_0", "_hull_anon_v_x_0_2"} <= {v.name for v in m._variables}
    back = loads(dumps(m))
    assert [v.name for v in back._variables] == [v.name for v in m._variables]


def test_affine_policy_frees_the_substituted_name():
    """``y`` is substituted out: its name is free, on the model as on a reload."""
    m = _build("ro:affine_policy")
    assert "y" not in {v.name for v in m._variables}
    m.continuous("y", lb=0, ub=1)
    back = loads(dumps(m))
    assert "y" in {v.name for v in back._variables}


def test_fresh_name_suffixes_only_on_a_clash():
    m = dm.Model("f")
    m.continuous("a")
    assert m._fresh_name("b") == "b"
    assert m._fresh_name("a") == "a_2"
    m.continuous("a_2")
    assert m._fresh_name("a") == "a_3"

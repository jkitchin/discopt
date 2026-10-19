"""#1790: follow-ups to #1775 G1.2 (nested disjunctions and local variables).

1. A block disjunction none of whose disjuncts has a row of its own (each holds
   only a nested disjunction, or nothing) tagged no row, so the GDP tree never
   built its node and every solve raised. Its key was also invisible to the next
   emission, so a disjunction nested in it reused the key.
2. A block could sit in two disjunctions, or at top level and nested.
3. A nesting cycle was accepted and recursed without end when lowered.
4. Local variables escaping through SOS / logic rows were not refused; ``serialize``
   dropped local-ness silently; an all-empty nested block got the generic
   unclassified-attribute refusal instead of the G1.2 one.
"""

from __future__ import annotations

import warnings

import discopt.modeling as dm
import pytest
from discopt.gdp import gdp_tree
from discopt.modeling.core import BooleanVar

pytestmark = pytest.mark.smoke

METHODS = ["big-m", "mbigm", "hull", "auto", "simplex"]


def _xy(name):
    m = dm.Model(name)
    return m, m.continuous("x", lb=0, ub=10), m.continuous("y", lb=0, ub=10)


def _both_dc_only():
    """A [x <= 1 v x >= 9] v B [y <= 1 v y >= 9]: neither block has a row."""
    m, x, y = _xy("both_dc_only")
    a = m.make_disjunct("A")
    a.subject_to(m.disjunction([[x <= 1], [x >= 9]], name="ina"))
    b = m.make_disjunct("B")
    b.subject_to(m.disjunction([[y <= 1], [y >= 9]], name="inb"))
    m.add_disjunction([a, b], name="outer")
    return m, x, y


def _two_rowless_levels():
    """A [A1 [x <= 1 v x >= 9] v A2 [y <= 2 v y >= 8]] v B [x <= 4 v x >= 7].

    Row-less at two levels: no block anywhere has a row of its own."""
    m, x, y = _xy("two_rowless_levels")
    a = m.make_disjunct("A")
    a1 = m.make_disjunct("A1")
    a1.subject_to(m.disjunction([[x <= 1], [x >= 9]], name="in_a1"))
    a2 = m.make_disjunct("A2")
    a2.subject_to(m.disjunction([[y <= 2], [y >= 8]], name="in_a2"))
    a.add_disjunction([a1, a2], name="inner")
    b = m.make_disjunct("B")
    b.subject_to(m.disjunction([[x <= 4], [x >= 7]], name="in_b"))
    m.add_disjunction([a, b], name="outer")
    return m, x, y


def _block_only_and_dc_only():
    """A [A1 [x >= 4] v A2 [x >= 5]] v B [y <= 1 v y >= 9]."""
    m, x, y = _xy("block_only_and_dc_only")
    a = m.make_disjunct("A")
    a1 = m.make_disjunct("A1")
    a1.subject_to(x >= 4)
    a2 = m.make_disjunct("A2")
    a2.subject_to(x >= 5)
    a.add_disjunction([a1, a2])
    b = m.make_disjunct("B")
    b.subject_to(m.disjunction([[y <= 1], [y >= 9]], name="inb"))
    m.add_disjunction([a, b])
    return m, x, y


# min (x - 5)^2 + (y - 5)^2, by enumerating the selections by hand. Dropping the
# nesting gives 0 at (5, 5); enforcing every nested disjunction at once gives more.
#   both_dc_only:        A (1 or 9, 5) 16,  B (5, 1 or 9) 16                -> 16
#   two_rowless_levels:  A1 16, A2 (5, 2 or 8) 9, B (4 or 7, 5) 1            -> 1
#   block_only_dc_only:  A1 (5, 5) 0 -- min (x - 2)^2 + (y - 5)^2 instead:
#                        A1 (4, 5) 4, A2 (5, 5) 9, B (2, 1 or 9) 16          -> 4
CASES = [
    (_both_dc_only, lambda x, y: (x - 5) ** 2 + (y - 5) ** 2, 16.0),
    (_two_rowless_levels, lambda x, y: (x - 5) ** 2 + (y - 5) ** 2, 1.0),
    (_block_only_and_dc_only, lambda x, y: (x - 2) ** 2 + (y - 5) ** 2, 4.0),
]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("build, obj, truth", CASES)
def test_rowless_block_disjunction_solves(build, obj, truth, method):
    """Fails before #1790: ``ValueError: disjunction ... is nested in a disjunct
    whose indicator ... belongs to no disjunction of this model``."""
    m, x, y = build()
    m.minimize(obj(x, y))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = m.solve(gdp_method=method, time_limit=60)
    assert r.status == "optimal"
    assert r.objective == pytest.approx(truth, abs=1e-5)


def test_rowless_disjunction_has_a_tree_node_in_model_order():
    m, x, y = _two_rowless_levels()
    c = m.make_disjunct("C")
    c.subject_to(y >= 3)
    d = m.make_disjunct("D")
    d.subject_to(y <= 2)
    m.add_disjunction([c, d], name="after")
    tree = gdp_tree(m)
    assert [r.origin.source.split(":")[1] for r in tree.disjunctions] == ["outer", "after"]
    outer = tree.disjunctions[0]
    assert [d.name for d in outer.disjuncts] == ["A_active", "B_active"]
    a, b = outer.disjuncts
    assert [ch.name for ch in b.children] == ["in_b"]
    (inner,) = a.children
    assert [d.name for d in inner.disjuncts] == ["A1_active", "A2_active"]
    assert [ch.name for ch in inner.disjuncts[0].children] == ["in_a1"]
    # Keys are distinct even though the row-less disjunctions tag no row.
    keys = [n.origin.source for n in tree.walk() if n.kind == "add_disjunction"]
    assert len(keys) == len(set(keys)) == 3


def test_a_block_belongs_to_one_disjunction():
    m, x, y = _xy("two_places")
    a, b, c = (m.make_disjunct(n) for n in "ABC")
    d = m.make_disjunct("D")
    d.subject_to(x >= 1)
    a.add_disjunction([d, m.make_disjunct("E")])
    with pytest.raises(ValueError, match="'D' cannot go into .* already in a disjunction"):
        b.add_disjunction([d, m.make_disjunct("F")])
    with pytest.raises(ValueError, match="'D' .* already in a disjunction of disjunct 'A'"):
        m.add_disjunction([d, c])
    # A refused call claims nothing: C is still free to place.
    m.add_disjunction([a, c], name="top")
    with pytest.raises(ValueError, match="already in a disjunction of the model"):
        m.add_disjunction([c, b])
    with pytest.raises(ValueError, match="listed twice"):
        m.add_disjunction([b, b])


def test_a_nested_disjunction_belongs_to_one_block():
    m, x, y = _xy("dc_two_places")
    inner = m.disjunction([[y <= 1], [y >= 6]], name="inner")
    a, b = m.make_disjunct("A"), m.make_disjunct("B")
    a.subject_to(inner)
    b.subject_to(inner)
    with pytest.raises(ValueError, match="'inner' is nested in more than one Disjunct block"):
        m.add_disjunction([a, b])


def test_a_nesting_cycle_is_refused_when_built():
    """Fails before #1790: accepted here, then ``RecursionError`` when lowered."""
    m, x, y = _xy("cycle")
    a, b, c = (m.make_disjunct(n) for n in "ABC")
    a.add_disjunction([b, m.make_disjunct("A2")])
    with pytest.raises(ValueError, match="nesting cycle"):
        b.add_disjunction([a, m.make_disjunct("B2")])
    b.add_disjunction([c, m.make_disjunct("B3")])
    with pytest.raises(ValueError, match="nesting cycle"):
        c.add_disjunction([a, m.make_disjunct("C2")])


def _local_on_a():
    m, x, y = _xy("local")
    a = m.make_disjunct("A")
    z = a.variable("z", kind="binary")
    a.subject_to(x >= 3 * z)
    b = m.make_disjunct("B")
    b.subject_to(x >= 6)
    m.add_disjunction([a, b])
    m.minimize(x)
    return m, z, y


def test_local_variable_in_an_sos_set_is_refused():
    m, z, y = _local_on_a()
    m.sos1([z, y])
    with pytest.raises(ValueError, match="'z' is declared local .* used in an SOS constraint"):
        m.validate()


def test_local_variable_in_a_logic_row_is_refused():
    m, z, _ = _local_on_a()
    w = m.boolean("w")
    m.logical(BooleanVar(z).implies(w))
    with pytest.raises(ValueError, match="'z' is declared local .* used in a logical constraint"):
        m.validate()


def test_serialize_keeps_local_variables(tmp_path):
    """Refused before schema 1.7 (plan WP-3b-2). The blocks travel with their local
    variables, so the reloaded model still refuses a use of one outside its block."""
    from discopt.serialize import load, save

    m, _, _ = _local_on_a()
    m.validate()
    save(m, tmp_path / "m.json")
    reloaded = load(tmp_path / "m.json")
    reloaded.validate()
    local = {v.name for d in reloaded._gdp_factory_disjuncts for v in d._local_vars}
    assert local == {v.name for d in m._gdp_factory_disjuncts for v in d._local_vars} != set()
    w = next(v for v in reloaded._variables if v.name in local)
    reloaded.subject_to(w <= 1.0, name="escape")
    with pytest.raises(ValueError, match="declared local to disjunct"):
        reloaded.validate()


def test_serialize_keeps_an_all_empty_nested_block(tmp_path):
    """A nested block none of whose disjuncts emits a row is recorded only in the
    block-disjunction record (#1790); refused before schema 1.7, now written."""
    from discopt.serialize import load, save

    m, x, y = _xy("empty_inner")
    a = m.make_disjunct("A")
    a.subject_to(x <= 3)
    a.add_disjunction([m.make_disjunct("A1"), m.make_disjunct("A2")])
    b = m.make_disjunct("B")
    b.subject_to(x >= 7)
    m.add_disjunction([a, b])
    m.minimize(x)
    save(m, tmp_path / "m.json")
    reloaded = load(tmp_path / "m.json")

    def shape(model):
        return [
            (len(n.disjuncts), [len(d.children) for d in n.disjuncts])
            for n in gdp_tree(model).disjunctions
        ]

    assert shape(reloaded) == shape(m) == [(2, [1, 0])]
    r1, r2 = m.solve(time_limit=60), reloaded.solve(time_limit=60)
    assert (r1.status, r1.objective, r1.node_count) == (r2.status, r2.objective, r2.node_count)

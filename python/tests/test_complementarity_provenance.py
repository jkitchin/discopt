"""Durable complementarity provenance through the model-rebuilding passes (#1147).

Before this slice, ``Model.complementarity`` recorded a pair on
``Model._complementarities`` and every pass that constructs a fresh ``Model``
— GDP lowering, integer-product expansion, factorable reformulation,
binary-multilinear linearization — dropped it. What survived a lowering was
string-level only: the pair name baked into generated identifiers
(``_gdp_aux_disj_pair0_0_0``). A source-residual probe written against the
rebuilt model would then have measured the *relaxed row* instead of the source
product, printed a small number, and been believed — the
instrument-that-measures-nothing failure mode CLAUDE.md §6–§7 exist for.

Every test here fails on ``main`` (``after == 0``) except where noted, and the
representation tests exercise the box-MCP form the slice adds.
"""

from __future__ import annotations

import discopt.modeling.core as dm
import numpy as np
import pytest
from discopt._relax.binary_multilinear_reform import reformulate_binary_multilinear
from discopt._relax.factorable_reform import factorable_reformulate
from discopt._relax.gdp_reformulate import reformulate_gdp
from discopt._relax.integer_product_reform import expand_integer_products
from discopt.mpec import (
    Complementarity,
    ComplementarityProvenanceError,
    ComplementarityRole,
    box_mcp,
    carry_complementarities,
    flat_source_indices,
    resolve_source_variables,
    source_variables,
    unlowered_relations,
)


def _mpcc(extra=None, *, name="pair0"):
    """min (x-1)^2 + (y-1)^2 s.t. 0 <= x _|_ y >= 0, plus optional extra structure."""
    m = dm.Model("mpcc")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    if extra is not None:
        extra(m)
    pair = m.complementarity(x, y, name=name)
    return m, x, y, pair


def _assert_resolves_to(model, pair, expected):
    """The relation resolves, by object identity, to exactly ``expected``."""
    got = resolve_source_variables(model, pair, context="test")
    assert [id(v) for v in got] == [id(v) for v in expected]


# ── the defect: the relation set survives every rebuilding pass ──


@pytest.mark.parametrize("method", ["big-m", "hull", "mbigm"])
def test_gdp_lowering_carries_the_relation(method):
    m, x, y, pair = _mpcc()
    assert len(m._complementarities) == 1

    out = reformulate_gdp(m, method)

    assert out is not m, "the pass must have rebuilt the model for this to be a test"
    assert out._complementarities == [pair], f"{method}: relation dropped by the rebuild"
    # The SOURCE operands, not the generated selector/aux columns.
    _assert_resolves_to(out, pair, [x, y])
    assert pair.f is x and pair.g is y


def test_integer_product_expansion_carries_the_relation():
    def extra(m):
        k = m.integer("k", lb=0, ub=3)
        z = m.continuous("z", lb=0, ub=5)
        m.subject_to(k * z <= 4)

    m, x, y, pair = _mpcc(extra)
    out = expand_integer_products(m)

    assert out is not m
    assert out._complementarities == [pair]
    _assert_resolves_to(out, pair, [x, y])


def test_factorable_reformulation_carries_the_relation():
    def extra(m):
        w = m.continuous("w", lb=1, ub=5)
        v = m.continuous("v", lb=1, ub=5)
        m.subject_to(v / w <= 4)

    m, x, y, pair = _mpcc(extra)
    out = factorable_reformulate(m)

    assert out is not m
    assert out._complementarities == [pair]
    _assert_resolves_to(out, pair, [x, y])


def test_binary_multilinear_linearization_carries_the_relation():
    """The pass fires on the GDP-LOWERED model (its rows are ordinary constraints
    by then) and must carry the relation the GDP pass forwarded to it."""
    m = dm.Model("bml_mpcc")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    b = [m.binary(f"b{i}") for i in range(3)]
    # Purely-MILP shape: the only degree>=3 monomial is binary, so the pass is
    # in scope (a continuous square would take it out of scope for its own reasons).
    m.minimize(x + y + 3.0 * b[0] * b[1] * b[2])
    pair = m.complementarity(x, y, name="pair0")

    lowered = reformulate_gdp(m, "big-m")
    assert lowered._complementarities == [pair]

    out = reformulate_binary_multilinear(lowered)
    assert out is not lowered, (
        "the binary-multilinear pass must FIRE here — it used to abstain on any model "
        "carrying a complementarity record, a guard keyed on state an earlier pass emptied"
    )
    assert out._complementarities == [pair]
    _assert_resolves_to(out, pair, [x, y])


def test_relation_survives_a_chain_of_passes_as_the_same_object():
    def extra(m):
        w = m.continuous("w", lb=1, ub=5)
        v = m.continuous("v", lb=1, ub=5)
        m.subject_to(v / w <= 4)

    m, x, y, pair = _mpcc(extra)
    out = factorable_reformulate(reformulate_gdp(m, "big-m"))

    assert out._complementarities[0] is pair, "identity must hold across a chain of passes"
    _assert_resolves_to(out, pair, [x, y])


# ── resolution is by object identity, not by name or index ──


def test_resolution_survives_renumbering_and_renaming_of_variables():
    """Presolve/FBBT eliminate and renumber columns, so an index- or name-keyed
    map is stale exactly when it becomes useful. Identity is not."""
    m, x, y, pair = _mpcc()
    out = reformulate_gdp(m, "big-m")

    idx_before = flat_source_indices(out, pair)

    # Simulate a renumbering pass: prepend a fresh column and renumber every
    # variable, and rename the operands out from under the generated row names.
    victim = dm.Variable("_eliminated", dm.VarType.CONTINUOUS, (), 0.0, 1.0, out)
    out._variables.insert(0, victim)
    for i, v in enumerate(out._variables):
        v._index = i
    out._flat_offsets = None  # invalidate the memoized prefix-sum table
    x.name, y.name = "x_renamed", "y_renamed"
    out._rebuild_name_index()

    _assert_resolves_to(out, pair, [x, y])
    idx_after = flat_source_indices(out, pair)
    assert idx_after != idx_before, "the renumbering must actually have moved the columns"
    assert idx_after == [1, 2]


def test_flat_indices_are_derived_not_persisted():
    """Scope item 4: indices are computed at the boundary, so appending columns
    moves them rather than leaving a stored answer stale."""
    m, x, y, pair = _mpcc()
    before = flat_source_indices(m, pair)
    m.continuous("later", lb=0, ub=1)
    assert flat_source_indices(m, pair) == before  # appended AFTER: prefix unchanged
    assert not hasattr(pair, "_flat_indices"), "indices must not be persisted on the relation"


def test_unresolvable_provenance_raises_naming_pair_and_pass():
    m, x, y, pair = _mpcc(name="mypair")
    dst = dm.Model("rebuilt")
    dst._variables = [v for v in m._variables if v is not y]  # y eliminated by the "pass"

    with pytest.raises(ComplementarityProvenanceError) as exc:
        carry_complementarities(m, dst, pass_name="pretend pass")
    msg = str(exc.value)
    assert "mypair" in msg
    assert "pretend pass" in msg
    assert "y" in msg
    assert dst._complementarities == [], "a relation that cannot be resolved is never carried"


def test_provenance_error_is_not_swallowed_by_a_defensive_pass_handler():
    """``expand_integer_products`` returns the model unchanged on any unexpected
    error. A broken provenance chain must escape that handler, not degrade to a
    silent abstain."""

    def extra(m):
        k = m.integer("k", lb=0, ub=3)
        z = m.continuous("z", lb=0, ub=5)
        m.subject_to(k * z <= 4)

    m, x, y, pair = _mpcc(extra)
    # A relation over a variable the model never held: unresolvable by construction.
    orphan = dm.Model("orphan")
    ghost = orphan.continuous("ghost", lb=0, ub=1)
    m._complementarities.append(Complementarity(ghost, ghost, "ghost_pair"))

    with pytest.raises(ComplementarityProvenanceError, match="ghost_pair"):
        expand_integer_products(m)


def test_source_variables_returns_objects_for_a_nonlinear_operand():
    m = dm.Model("nl")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    z = m.continuous("z", lb=0, ub=10)
    m.minimize(x + y + z)
    pair = m.complementarity(x * z + 1, y, name="nlpair")
    got = source_variables(pair)
    assert {id(v) for v in got} == {id(x), id(z), id(y)}


# ── vector / array relations keep elementwise identity ──


def test_vector_relation_keeps_elementwise_identity_through_a_pass():
    m = dm.Model("vec")
    x = m.continuous("x", shape=3, lb=0, ub=10)
    y = m.continuous("y", shape=3, lb=0, ub=10)
    m.minimize(dm.sum(x) + dm.sum(y))
    pair = m.complementarity(x, y, name="vpair")

    out = reformulate_gdp(m, "big-m")
    assert out._complementarities == [pair]
    assert pair.shape == (3,)

    elems = pair.elements(out)
    assert len(elems) == 3
    assert [e.index for e in elems] == [(0,), (1,), (2,)]
    assert all(e.source is pair for e in elems)
    assert all(e.role is pair.role for e in elems)
    for e in elems:
        # Each element still reads the SOURCE columns, by identity.
        _assert_resolves_to(out, e, [x, y])


def test_broadcast_relation_keeps_elementwise_identity():
    m = dm.Model("bcast")
    s = m.continuous("s", lb=0, ub=10)
    y = m.continuous("y", shape=2, lb=0, ub=10)
    m.minimize(s + dm.sum(y))
    m.complementarity(s, y, name="bpair")

    out = reformulate_gdp(m, "big-m")
    elems = out._complementarities[0].elements(out)
    assert [e.index for e in elems] == [(0,), (1,)]
    assert all(e.f is s for e in elems), "the broadcast side is shared, by identity"


# ── the box-bounded MCP form ──


@pytest.mark.parametrize(
    "lb,ub,case",
    [
        (0.0, 5.0, "lower-active"),
        (-3.0, 3.0, "interior"),
        (-5.0, 0.0, "upper-active"),
        (-np.inf, 2.0, "one-sided-infinite-below"),
        (2.0, np.inf, "one-sided-infinite-above"),
        (1.5, 1.5, "fixed l == u"),
    ],
)
def test_box_mcp_round_trips_through_a_rebuilding_pass(lb, ub, case):
    """Every box configuration is represented and forwarded unchanged. The
    ``l=0, u=+inf`` case is the NCP pair and is covered by its own test, since
    it is the one configuration this slice *does* lower."""
    m = dm.Model("mcp")
    z = m.continuous("z", lb=max(lb, -50.0), ub=min(ub, 50.0))
    w = m.continuous("w", lb=0, ub=10)
    m.minimize(z * z + w)
    m.subject_to(z + w >= 1)
    rel = m.mcp(z + w - 1, z, lb=lb, ub=ub, name="mcp0")

    assert rel.role is ComplementarityRole.BOX_MCP, case
    assert rel.g_bounds == (float(lb), float(ub))
    assert rel.f_bounds == (-np.inf, np.inf), "the residual side of a box MCP is free"

    # An unlowered relation emits no GDP rows, so drive a pass that does rebuild
    # the model for its own reasons: the integer-product expansion.
    k = m.integer("k", lb=0, ub=3)
    v = m.continuous("v", lb=0, ub=5)
    m.subject_to(k * v <= 4)
    out = expand_integer_products(m)
    assert out is not m

    carried = out._complementarities[0]
    assert carried is rel, "the relation object itself must be forwarded"
    assert carried.role is ComplementarityRole.BOX_MCP
    assert carried.g_bounds == (float(lb), float(ub))
    _assert_resolves_to(out, carried, [z, w])


def test_box_mcp_with_l0_uinf_is_recorded_as_the_ncp_pair_and_lowered():
    m = dm.Model("mcp_ncp")
    a = m.continuous("a", lb=0, ub=10)
    b = m.continuous("b", lb=0, ub=10)
    m.minimize((a - 1) ** 2 + (b - 1) ** 2)
    rel = m.mcp(b, a, lb=0.0, ub=np.inf, name="ncp")

    assert rel.role is ComplementarityRole.NCP_PAIR
    assert rel.is_lowered_into(m), "the shared special case must be lowered, not refused"
    assert m.solve().objective == pytest.approx(1.0, abs=1e-3)


def test_box_mcp_defaults_the_box_to_the_variables_declared_bounds():
    m = dm.Model("mcp_default")
    z = m.continuous("z", lb=-2.0, ub=7.0)
    m.minimize(z)
    rel = m.mcp(z - 1, z, name="d")
    assert rel.g_bounds == (-2.0, 7.0)


def test_box_mcp_requires_explicit_bounds_for_a_non_variable_side():
    m = dm.Model("mcp_expr")
    z = m.continuous("z", lb=0, ub=5)
    with pytest.raises(ValueError, match="lb/ub are required"):
        box_mcp(z, z + 1, name="e")


def test_box_mcp_rejects_an_empty_box():
    m = dm.Model("mcp_empty")
    z = m.continuous("z", lb=0, ub=5)
    with pytest.raises(ValueError, match="empty box"):
        box_mcp(z, z, lb=3.0, ub=1.0, name="e")


def test_unbounded_sentinel_bound_reduces_to_the_ncp_pair():
    """``m.continuous('a', lb=0)`` leaves ub at the 9.999e19 sentinel; comparing
    it literally against ``inf`` would misclassify the pair as a general box."""
    m = dm.Model("sentinel")
    a = m.continuous("a", lb=0)
    b = m.continuous("b", lb=0, ub=10)
    m.minimize(a + b)
    rel = box_mcp(b, a, name="s")
    assert rel.role is ComplementarityRole.NCP_PAIR
    assert rel.g_bounds == (0.0, np.inf)


def test_solve_refuses_a_model_carrying_an_unlowered_relation():
    m = dm.Model("unlowered")
    z = m.continuous("z", lb=-2, ub=5)
    w = m.continuous("w", lb=0, ub=10)
    m.minimize(z * z + w)
    rel = m.mcp(z + w - 1, z, name="mcp0")

    assert unlowered_relations(m) == [rel]
    with pytest.raises(NotImplementedError) as exc:
        m.solve(time_limit=5.0)
    assert "mcp0" in str(exc.value)


def test_box_mcp_is_refused_by_every_lowering_rather_than_mis_encoded():
    from discopt.mpec import reformulate_gdp as mpec_gdp
    from discopt.mpec import reformulate_scholtes, reformulate_sos1

    m = dm.Model("box_lower")
    z = m.continuous("z", lb=-2, ub=5)
    m.minimize(z)
    rel = box_mcp(z, z, lb=-2.0, ub=5.0, name="boxed")
    for lower in (mpec_gdp, reformulate_sos1):
        with pytest.raises(NotImplementedError, match="box-MCP"):
            lower(m, [rel])
    with pytest.raises(NotImplementedError, match="box-MCP"):
        reformulate_scholtes(m, [rel], 0.1)


def test_complementarity_bound_tightening_skips_box_relations():
    """The "one side strictly positive => partner is 0" rule is a property of the
    NONNEGATIVE pair; on a box MCP the partner is pinned to a bound of [l, u]."""
    from discopt.mpec import tighten_complementarity_bounds

    m = dm.Model("tighten_box")
    driver = m.continuous("driver", lb=1.0, ub=5.0)  # strictly positive
    partner = m.continuous("partner", lb=-3.0, ub=4.0)
    rel = box_mcp(driver, partner, lb=-3.0, ub=4.0, name="boxed")
    assert tighten_complementarity_bounds(m, [rel]) == 0
    assert float(np.max(np.asarray(partner.ub))) == 4.0, "the box must be untouched"


# ── lowering state ──


def test_lowering_state_is_recorded_and_prevents_duplicate_rows():
    from discopt.mpec import reformulate_gdp as mpec_gdp

    m, x, y, pair = _mpcc()
    assert pair.lowering == "gdp"
    assert pair.is_lowered_into(m)

    n_cons = len(m._constraints)
    n_vars = len(m._variables)
    mpec_gdp(m, [pair])  # a second lowering of the same relation into the same model
    assert len(m._constraints) == n_cons, "re-lowering must not duplicate rows"
    assert len(m._variables) == n_vars


def test_lowering_state_is_per_model_so_a_second_model_still_gets_its_rows():
    from discopt.mpec import complementarity as make_pair
    from discopt.mpec import reformulate_gdp as mpec_gdp

    m1 = dm.Model("m1")
    x = m1.continuous("x", lb=0, ub=10)
    y = m1.continuous("y", lb=0, ub=10)
    m1.minimize(x + y)
    pair = make_pair(x, y, "shared")
    mpec_gdp(m1, [pair])
    assert len(m1._constraints) > 0

    m2 = dm.Model("m2")
    m2._variables = [x, y]
    m2._rebuild_name_index()
    m2.minimize(x + y)
    mpec_gdp(m2, [pair])
    assert len(m2._constraints) > 0, (
        "the same relation lowered into a DIFFERENT model must emit its rows; "
        "the skip is keyed to the model, not to the relation alone"
    )


def test_carry_marks_the_relation_lowered_into_the_rebuilt_model():
    m, x, y, pair = _mpcc()
    out = reformulate_gdp(m, "big-m")
    assert pair.is_lowered_into(out), (
        "the rebuilt model carries the generated rows, so re-lowering into it would duplicate them"
    )
    assert unlowered_relations(out) == []


# ── the richer relation ──


def test_relation_carries_role_bounds_scale_and_parent():
    m = dm.Model("fields")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x + y)
    pair = m.complementarity(x, y, name="p", scale=1e3)

    assert pair.role is ComplementarityRole.NCP_PAIR
    assert pair.f_bounds == (0.0, np.inf)
    assert pair.g_bounds == (0.0, np.inf)
    assert pair.effective_scale == 1e3
    assert pair.parent is None
    assert "p" in pair.describe() and "ncp_pair" in pair.describe()


def test_effective_scale_falls_back_to_the_declared_bounds():
    m = dm.Model("scale")
    z = m.continuous("z", lb=0, ub=250.0)
    rel = box_mcp(z, z, lb=0.0, ub=250.0, name="s")
    assert rel.scale is None
    assert rel.effective_scale == 250.0

    m2 = dm.Model("scale2")
    a = m2.continuous("a", lb=0, ub=1)
    b = m2.continuous("b", lb=0, ub=1)
    m2.minimize(a + b)
    ncp = m2.complementarity(a, b, name="n")
    assert ncp.effective_scale == 1.0, "an unbounded NCP pair keeps the absolute tolerance"


def test_kkt_generated_pairs_declare_their_role_and_parent():
    from discopt.bilevel.kkt import build_kkt

    m = dm.Model("kkt")
    xu = m.continuous("xu", lb=0, ub=10)
    yl = m.continuous("yl", lb=0, ub=10)
    m.minimize(xu + yl)
    con = dm.Constraint(body=yl - xu, sense="<=", rhs=0.0, name="follow")
    sysm = build_kkt(m, lower_vars=[yl], lower_objective=yl, lower_constraints=[con])

    assert len(sysm.comp_pairs) == 1
    pair = sysm.comp_pairs[0]
    assert pair.role is ComplementarityRole.FROM_KKT
    assert pair.parent == "follow"


def test_relation_is_identity_hashed_so_it_keys_a_provenance_map():
    m, x, y, pair = _mpcc()
    out = reformulate_gdp(m, "big-m")
    residuals = {pair: 0.0}
    assert out._complementarities[0] in residuals
    other = Complementarity(x, y, "pair0")
    assert other not in residuals, "two structurally identical relations are two relations"


# ── no behavior change ──


def test_lowered_model_is_row_for_row_unchanged_by_the_provenance_carry():
    """Bound-neutral regime: the carry adds a Python-side record, never a row."""
    m, x, y, pair = _mpcc()
    out = reformulate_gdp(m, "big-m")
    assert len(out._variables) == 4  # x, y and the two GDP selector binaries
    assert out._complementarities == [pair]

    res = out.solve(time_limit=30.0, gap_tolerance=1e-6)
    assert res.objective == pytest.approx(1.0, abs=1e-3)


def test_nl_imported_complementarity_survives_the_gdp_pass():
    from pathlib import Path

    fixture = Path(__file__).parent / "data" / "mpcc_complementarity.nl"
    m = dm.from_nl(str(fixture))
    assert len(m._complementarities) == 1
    pair = m._complementarities[0]

    out = reformulate_gdp(m, "big-m")
    assert out._complementarities == [pair]
    resolve_source_variables(out, pair, context="test")

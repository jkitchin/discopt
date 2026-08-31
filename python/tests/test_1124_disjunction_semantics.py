"""Issue #1124 — lock the disjunction semantics contract.

RFC #1123 asks that ``OR`` / ``SELECT_ONE`` / ``EXACTLY_ONE_TRUE`` be an
explicit contract rather than something inferred from whichever activation a
lowering happens to emit. These tests pin the *current* meaning of
``either_or`` — ``SELECT_ONE``: exactly one disjunct is **selected**, one-way
activation, projection onto ``x`` equal to the **union** of the disjuncts — on
every lowering path, and pin the refusals that keep the other two semantics
from being silently served by the select-one machinery.

The discriminator throughout is

    min (x - 1/2)^2   s.t.   [x <= 1]  or  [x >= 0],   x in [-2, 3]

whose optimum is ``0`` at ``x = 1/2`` under union/select-one and ``0.25`` (at
``x in {0, 1}``) under truth semantics, since ``x = 1/2`` satisfies *both*
predicates. Note that the pre-existing ``test_hull_overlapping_disjuncts`` does
**not** discriminate: its ``min x`` objective is ``0`` under either reading.
"""

import discopt.modeling as dm
import pytest
from discopt._relax.gdp_reformulate import (
    _SUPPORTED_SEMANTICS,
    _semantics_is_supported,
    reformulate_gdp,
)
from discopt.modeling.core import (
    GDP_AUX_PREFIX,
    DisjunctionSemantics,
    SelectorActivation,
    SelectorCardinality,
    Variable,
    VarType,
    _DisjunctiveConstraint,
)

# Objective value under union / select-one semantics vs. under truth semantics.
SELECT_ONE_OPT = 0.0
TRUTH_OPT = 0.25

ALL_METHODS = ["big-m", "mbigm", "hull", "auto"]


def _overlap_model():
    """min (x-1/2)^2 s.t. [x<=1] or [x>=0] — the disjuncts overlap on [0, 1]."""
    m = dm.Model("overlap")
    x = m.continuous("x", lb=-2.0, ub=3.0)
    m.minimize((x - 0.5) ** 2)
    m.either_or([[x <= 1.0], [x >= 0.0]], name="ov")
    return m


# ---------------------------------------------------------------------------
# The enum is a pair, not a flat triple
# ---------------------------------------------------------------------------


class TestSemanticsEnum:
    def test_member_value_is_the_pair_itself(self):
        """The axes are carried BY the member, not by a side lookup table.

        A separate table can drift out of sync with the members; the member's
        own ``value`` cannot. This also pins the public ``.value`` shape.
        """
        assert DisjunctionSemantics.SELECT_ONE.value == (
            SelectorActivation.ONE_WAY,
            SelectorCardinality.EXACTLY_ONE,
        )
        assert DisjunctionSemantics.OR.value == (
            SelectorActivation.ONE_WAY,
            SelectorCardinality.AT_LEAST_ONE,
        )
        assert DisjunctionSemantics.EXACTLY_ONE_TRUE.value == (
            SelectorActivation.REIFIED,
            SelectorCardinality.EXACTLY_ONE,
        )
        for member in DisjunctionSemantics:
            assert member.value == (member.activation, member.cardinality)

    def test_string_spelling_is_a_coercion_alias_not_the_value(self):
        m = dm.Model("alias")
        x = m.continuous("x", lb=0.0, ub=1.0)
        m.either_or([[x <= 0.5], [x >= 0.5]], name="d", semantics="select_one")
        (dc,) = [c for c in m._constraints if isinstance(c, _DisjunctiveConstraint)]
        assert dc.semantics is DisjunctionSemantics.SELECT_ONE
        assert DisjunctionSemantics.SELECT_ONE.label == "select_one"

    def test_lowering_dispatch_is_keyed_on_the_pair_not_the_member(self):
        """The gate asks what the emitted rows encode, never which member it got."""
        assert _SUPPORTED_SEMANTICS == frozenset(
            {(SelectorActivation.ONE_WAY, SelectorCardinality.EXACTLY_ONE)}
        )
        for member in DisjunctionSemantics:
            expected = (member.activation, member.cardinality) in _SUPPORTED_SEMANTICS
            assert _semantics_is_supported(member) is expected
        # Only the select-one pair is served today.
        assert _semantics_is_supported(DisjunctionSemantics.SELECT_ONE)
        assert not _semantics_is_supported(DisjunctionSemantics.OR)
        assert not _semantics_is_supported(DisjunctionSemantics.EXACTLY_ONE_TRUE)

    def test_members_decompose_into_activation_and_cardinality(self):
        """Lowerings must be able to branch on the axes, not on the name."""
        assert DisjunctionSemantics.SELECT_ONE.activation is SelectorActivation.ONE_WAY
        assert DisjunctionSemantics.SELECT_ONE.cardinality is SelectorCardinality.EXACTLY_ONE

        assert DisjunctionSemantics.OR.activation is SelectorActivation.ONE_WAY
        assert DisjunctionSemantics.OR.cardinality is SelectorCardinality.AT_LEAST_ONE

        # The axis that separates SELECT_ONE from EXACTLY_ONE_TRUE is reification,
        # NOT cardinality: both are exactly-one.
        assert DisjunctionSemantics.EXACTLY_ONE_TRUE.activation is SelectorActivation.REIFIED
        assert DisjunctionSemantics.EXACTLY_ONE_TRUE.cardinality is SelectorCardinality.EXACTLY_ONE
        assert (
            DisjunctionSemantics.EXACTLY_ONE_TRUE.cardinality
            == DisjunctionSemantics.SELECT_ONE.cardinality
        )
        assert (
            DisjunctionSemantics.EXACTLY_ONE_TRUE.activation
            != DisjunctionSemantics.SELECT_ONE.activation
        )

    def test_xor_spelling_is_refused(self):
        """Pyomo.GDP's ``xor=True`` means select-one; over n>2 XOR means parity."""
        m = dm.Model("xor_reject")
        x = m.continuous("x", lb=0.0, ub=1.0)
        for spelling in ("xor", "truth_xor", "XOR"):
            with pytest.raises(ValueError, match="ambiguous"):
                m.either_or([[x <= 0.5], [x >= 0.5]], name="d", semantics=spelling)

    def test_unknown_semantics_is_refused(self):
        m = dm.Model("bad")
        x = m.continuous("x", lb=0.0, ub=1.0)
        with pytest.raises(ValueError, match="unknown disjunction semantics"):
            m.either_or([[x <= 0.5], [x >= 0.5]], name="d", semantics="at_least_two")


# ---------------------------------------------------------------------------
# AC-1 / AC-2: existing models keep their feasible set, on every lowering
# ---------------------------------------------------------------------------


class TestSelectOneIsTheDefault:
    def test_either_or_defaults_to_select_one(self):
        m = _overlap_model()
        (dc,) = [c for c in m._constraints if isinstance(c, _DisjunctiveConstraint)]
        assert dc.semantics is DisjunctionSemantics.SELECT_ONE

    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_overlap_point_stays_feasible(self, method):
        """AC-1/AC-2: a point in the overlap is feasible — the projection is the union.

        Under EXACTLY_ONE_TRUE, x=1/2 would be infeasible and the optimum 0.25.
        """
        r = _overlap_model().solve(gdp_method=method)
        assert r.status == "optimal"
        assert r.objective == pytest.approx(SELECT_ONE_OPT, abs=1e-4)
        assert r.objective != pytest.approx(TRUTH_OPT, abs=1e-2)
        assert float(r.x["x"]) == pytest.approx(0.5, abs=1e-3)

    @pytest.mark.parametrize("method", ["big-m", "hull"])
    def test_selector_row_is_exactly_one_not_at_least_one(self, method):
        """The emitted cardinality row is ``sum(y) == 1`` (select-one), not ``>= 1``."""
        ref = reformulate_gdp(_overlap_model(), method=method)
        sel = [c for c in ref._constraints if "select" in (getattr(c, "name", None) or "")]
        assert len(sel) == 1
        assert sel[0].sense == "=="

    def test_add_disjunction_block_path_is_select_one(self):
        m = dm.Model("blocks")
        x = m.continuous("x", lb=-2.0, ub=3.0)
        m.minimize((x - 0.5) ** 2)
        d1 = m.make_disjunct("a")
        d1.subject_to(x <= 1.0)
        d2 = m.make_disjunct("b")
        d2.subject_to(x >= 0.0)
        m.add_disjunction([d1, d2], name="ov")
        r = m.solve()
        assert r.objective == pytest.approx(SELECT_ONE_OPT, abs=1e-4)

    def test_add_disjunction_selector_row_is_not_named_xor(self):
        """The cardinality row is a select-one selector, not a truth-XOR."""
        m = dm.Model("naming")
        x = m.continuous("x", lb=0.0, ub=10.0)
        d1 = m.make_disjunct("a")
        d1.subject_to(x <= 3.0)
        d2 = m.make_disjunct("b")
        d2.subject_to(x >= 7.0)
        m.add_disjunction([d1, d2], name="modes")
        names = [getattr(c, "name", None) for c in m._constraints]
        assert "_disj_modes_select" in names
        assert not any((n or "").endswith("_xor") for n in names)


# ---------------------------------------------------------------------------
# AC-3: SELECT_ONE vs EXACTLY_ONE_TRUE is semantic, not convexity-driven
# ---------------------------------------------------------------------------


class TestSemanticNotConvexityDistinction:
    """A *deselected* disjunct's predicate may still be true — convex and nonconvex.

    Using named indicators makes the distinction directly observable: forcing an
    indicator to 0 means "this mode is not SELECTED", not "this predicate is false".
    """

    def test_convex_deselected_predicate_may_be_true(self):
        m = dm.Model("convex_sel")
        x = m.continuous("x", lb=-2.0, ub=3.0)
        m.minimize((x - 0.5) ** 2)
        da = m.make_disjunct("a")
        da.subject_to(x <= 1.0)  # convex
        db = m.make_disjunct("b")
        db.subject_to(x >= 0.0)  # convex
        m.add_disjunction([da, db], name="ov")
        m.subject_to(da.indicator.variable == 0)  # deselect mode a

        r = m.solve()
        assert r.objective == pytest.approx(SELECT_ONE_OPT, abs=1e-4)
        assert float(r.x["a_active"]) == pytest.approx(0.0, abs=1e-6)
        # ... and yet the deselected disjunct's predicate holds at the optimum.
        assert float(r.x["x"]) <= 1.0 + 1e-6

    def test_nonconvex_deselected_predicate_may_be_true(self):
        m = dm.Model("nonconvex_sel")
        z = m.continuous("z", lb=-3.0, ub=3.0)
        m.minimize((z - 2.0) ** 2)
        ea = m.make_disjunct("a")
        ea.subject_to(z * z >= 1.0)  # nonconvex
        eb = m.make_disjunct("b")
        eb.subject_to(z >= 0.0)
        m.add_disjunction([ea, eb], name="ov2")
        m.subject_to(ea.indicator.variable == 0)  # deselect mode a

        r = m.solve()
        assert r.objective == pytest.approx(0.0, abs=1e-3)
        assert float(r.x["a_active"]) == pytest.approx(0.0, abs=1e-6)
        # Same conclusion as the convex case: the distinction is not about convexity.
        assert float(r.x["z"]) ** 2 >= 1.0 - 1e-5


# ---------------------------------------------------------------------------
# AC-6 / the loud-refusal rule: nothing silently serves another semantics
# ---------------------------------------------------------------------------


class TestUnimplementedSemanticsRefuseLoudly:
    @pytest.mark.parametrize(
        "semantics", [DisjunctionSemantics.OR, DisjunctionSemantics.EXACTLY_ONE_TRUE]
    )
    @pytest.mark.parametrize("method", ALL_METHODS)
    def test_non_select_one_is_refused_by_every_lowering(self, semantics, method):
        m = dm.Model("refuse")
        x = m.continuous("x", lb=-2.0, ub=3.0)
        m.minimize((x - 0.5) ** 2)
        m.either_or([[x <= 1.0], [x >= 0.0]], name="ov", semantics=semantics)
        with pytest.raises(NotImplementedError, match=semantics.name):
            reformulate_gdp(m, method=method)

    def test_refusal_names_the_axes_and_the_tracking_issue(self):
        m = dm.Model("refuse_msg")
        x = m.continuous("x", lb=0.0, ub=1.0)
        m.either_or(
            [[x <= 0.6], [x >= 0.4]],
            name="ov",
            semantics=DisjunctionSemantics.EXACTLY_ONE_TRUE,
        )
        with pytest.raises(NotImplementedError) as exc:
            reformulate_gdp(m, method="big-m")
        msg = str(exc.value)
        assert "reified" in msg and "exactly_one" in msg
        assert "1124" in msg

    def test_nested_disjunction_refused_by_every_semantics_check(self):
        """A nested disjunction carries its own semantics and is checked too."""
        m = dm.Model("nested_refuse")
        z = m.continuous("z", lb=0.0, ub=10.0)
        m.minimize(z)
        inner = m.disjunction(
            [[z >= 3.0], [z >= 7.0]], name="in", semantics=DisjunctionSemantics.OR
        )
        m.either_or([[z >= 1.0, inner], [z >= 9.0]], name="out")
        with pytest.raises(NotImplementedError, match="OR"):
            reformulate_gdp(m, method="big-m")

    def test_add_disjunction_refuses_at_call_time(self):
        """This path lowers immediately, so it must refuse when called."""
        m = dm.Model("blocks_refuse")
        x = m.continuous("x", lb=0.0, ub=10.0)
        d1 = m.make_disjunct("a")
        d1.subject_to(x <= 3.0)
        d2 = m.make_disjunct("b")
        d2.subject_to(x >= 7.0)
        with pytest.raises(NotImplementedError, match="SELECT_ONE"):
            m.add_disjunction([d1, d2], name="m", semantics=DisjunctionSemantics.OR)


class TestNestedUnderHullRefusesClearly:
    def test_nested_hull_raises_not_implemented_not_attribute_error(self):
        """Was a bare ``AttributeError: '_DisjunctiveConstraint' has no attribute 'body'``."""
        m = dm.Model("nested_hull")
        z = m.continuous("z", lb=0.0, ub=10.0)
        m.minimize(z)
        inner = m.disjunction([[z >= 3.0], [z >= 7.0]], name="in")
        m.either_or([[z >= 1.0, inner], [z >= 9.0]], name="out")
        with pytest.raises(NotImplementedError) as exc:
            reformulate_gdp(m, method="hull")
        msg = str(exc.value)
        assert "nested" in msg and "big-m" in msg

    def test_nested_big_m_still_works(self):
        """The refusal is hull-only; big-M implements nesting and must be unaffected."""
        m = dm.Model("nested_bigm")
        z = m.continuous("z", lb=0.0, ub=10.0)
        m.minimize(z)
        inner = m.disjunction([[z >= 3.0], [z >= 7.0]], name="in")
        m.either_or([[z >= 1.0, inner], [z >= 9.0]], name="out")
        r = m.solve(gdp_method="big-m")
        assert r.status == "optimal"
        assert r.objective == pytest.approx(3.0, abs=1e-4)


# ---------------------------------------------------------------------------
# AC-5: user Boolean identities vs. generated existential auxiliaries
# ---------------------------------------------------------------------------


class TestIdentitiesVersusAuxiliaries:
    def test_generated_selectors_are_distinguishable_from_user_booleans(self):
        """A user's named Boolean identity is not a compiler-generated selector.

        The generated selectors are existential auxiliaries: the lowering chooses
        them, and they carry the reserved ``_gdp_aux_`` prefix. A user's Boolean is
        a named identity that survives lowering under its own name.
        """
        m = dm.Model("identities")
        x = m.continuous("x", lb=0.0, ub=10.0)
        y_user = m.boolean("mode_is_hot")  # user-visible Boolean identity
        m.minimize(x + y_user.variable)
        m.either_or([[x <= 3.0], [x >= 7.0]], name="d")  # generates selectors

        ref = reformulate_gdp(m, method="big-m")
        names = [v.name for v in ref._variables]

        generated = [n for n in names if n.startswith("_gdp_aux_")]
        assert len(generated) == 2  # one selector per disjunct

        assert "mode_is_hot" in names
        assert not any(n.startswith("_") for n in ("mode_is_hot",))
        assert "mode_is_hot" not in generated

    @pytest.mark.parametrize(
        "reserved",
        [
            GDP_AUX_PREFIX + "disj_d_0_0",  # the exact name the lowering would mint
            GDP_AUX_PREFIX,  # the bare prefix
            GDP_AUX_PREFIX + "anything",
        ],
    )
    def test_user_name_in_the_reserved_namespace_is_refused(self, reserved):
        """A user must not be able to take the namespace that marks an auxiliary.

        Before this was enforced, ``m.boolean("_gdp_aux_disj_d_0_0")`` followed by
        ``either_or(..., name="d")`` produced FOUR variables under THREE distinct
        names: the lowering minted the identical name for its own selector. Two
        different variables then shared one name, which breaks the AC-5 identity /
        auxiliary distinction and makes name-keyed result lookup ambiguous.
        """
        m = dm.Model("collision")
        with pytest.raises(ValueError, match="reserved"):
            m.boolean(reserved)
        # Every user-facing factory funnels through the same check.
        with pytest.raises(ValueError, match="reserved"):
            m.continuous(reserved)
        with pytest.raises(ValueError, match="reserved"):
            m.binary(reserved)

    def test_the_collision_that_motivated_the_reservation_cannot_recur(self):
        """End-to-end: the exact reviewed reproducer must not produce a duplicate name."""
        m = dm.Model("collide_e2e")
        x = m.continuous("x", lb=0.0, ub=10.0)
        m.minimize(x)
        with pytest.raises(ValueError, match="reserved"):
            m.boolean(GDP_AUX_PREFIX + "disj_d_0_0")
        # ... and with the user name refused, lowering still yields unique names.
        m.either_or([[x <= 3.0], [x >= 7.0]], name="d")
        names = [v.name for v in reformulate_gdp(m, method="big-m")._variables]
        assert len(names) == len(set(names)), f"duplicate variable names: {names}"

    def test_generated_names_are_unique_even_against_preexisting_auxiliaries(self):
        """The allocator, not just the reservation, guarantees uniqueness.

        A model can already carry reserved-prefix names without any user having
        typed one — a second lowering pass, or an imported model. The counter
        alone does not prevent a clash there, so the allocator checks the names
        actually present.
        """
        m = dm.Model("preexisting")
        x = m.continuous("x", lb=0.0, ub=10.0)
        m.minimize(x)
        m.either_or([[x <= 3.0], [x >= 7.0]], name="d")
        # Simulate a model that already carries the names the pass would mint,
        # bypassing the user-facing check exactly as an earlier pass would.
        for k in range(2):
            m._variables.append(
                Variable(f"{GDP_AUX_PREFIX}disj_d_{k}_{k}", VarType.BINARY, (), 0.0, 1.0, m)
            )

        names = [v.name for v in reformulate_gdp(m, method="big-m")._variables]
        assert len(names) == len(set(names)), f"duplicate variable names: {names}"
        generated = [n for n in names if n.startswith(GDP_AUX_PREFIX)]
        assert len(generated) == 4  # 2 pre-existing + 2 freshly allocated
        assert len(set(generated)) == 4

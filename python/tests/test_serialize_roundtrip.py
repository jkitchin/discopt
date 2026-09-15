"""Round-trip tests for the native model format (``discopt.serialize``).

The invariant these tests enforce is deliberately stronger than "the reloaded
model gets the same objective". ``docs/dev/performance-plan.md`` §7226 records
three export defects that a same-optimum round-trip test reported as passing and
a byte-diff caught, so fidelity here is judged by:

1. **Document stability** -- ``dumps(loads(dumps(m))) == dumps(m)``, byte for byte.
   A field that is dropped on read, or re-derived differently on write, shows up
   immediately.
2. **.nl byte-identity** -- writing ``.nl`` from the original model and from the
   reloaded model must produce identical bytes. This compares the two models
   through an independent writer that walks the whole DAG, so a lost coefficient,
   a reordered row, or a re-defaulted bound cannot hide behind a matching optimum.
3. **Bound-neutrality** -- ``node_count`` and the certified ``objective`` must be
   *exactly* unchanged across the round trip (CLAUDE.md §5, bound-neutral regime).

``test_corpus_round_trip_is_nl_byte_identical`` applies (1) and (2) to every
instance in the in-repo MINLPLib corpus, and ``test_corpus_probe_actually_fired``
asserts the corpus was non-empty -- a parametrized test over an empty glob passes
silently while measuring nothing.
"""

from __future__ import annotations

import json
from pathlib import Path

import discopt.modeling as dm
import numpy as np
import pytest
import scipy.sparse as sp
from discopt.modeling.core import (
    _DisjunctiveConstraint,
    _IndicatorConstraint,
    _SOSConstraint,
)
from discopt.serialize import SerializationError, dumps, load, loads

NL_DIR = Path(__file__).parent / "data" / "minlplib_nl"
CORPUS = sorted(NL_DIR.glob("*.nl"))


def _kinetics_model():
    """A small fitting-shaped model: array variable, parameter, integer, binary, nonlinear."""
    m = dm.Model("kinetics")
    k = m.continuous("k", shape=(3,), lb=0.0, ub=10.0)
    T = m.parameter("T", 350.0)
    n = m.integer("n", lb=0, ub=5)
    b = m.binary("b")
    m.minimize(dm.sum((k - 1.5) ** 2) + T * n + b)
    m.subject_to(k[0] + k[1] <= 4.0)
    m.subject_to(dm.exp(k[2]) - 2.0 == 0.0)
    return m


# ── the three fidelity invariants ──────────────────────────────────────────


@pytest.mark.smoke
def test_document_is_stable_under_reload():
    m = _kinetics_model()
    once = dumps(m)
    assert dumps(loads(once)) == once


@pytest.mark.smoke
def test_nl_export_is_byte_identical_after_round_trip():
    m = _kinetics_model()
    assert loads(dumps(m)).to_nl() == m.to_nl()


@pytest.mark.smoke
def test_reloaded_model_keeps_variable_names_unlike_nl():
    """The motivating gap: `.nl` renames every variable positionally."""
    m = _kinetics_model()
    back = loads(dumps(m))
    assert [v.name for v in back._variables] == [v.name for v in m._variables]
    assert [v.name for v in back._variables] == ["k", "n", "b"]
    # Contrast: the .nl route loses them.
    nl_back = dm.from_nl(str(CORPUS[0]))
    assert all(v.name.startswith("x") for v in nl_back._variables)


@pytest.mark.slow
@pytest.mark.parametrize("nl_path", CORPUS, ids=lambda p: p.stem)
def test_corpus_round_trip_is_nl_byte_identical(nl_path):
    """Every in-repo MINLPLib instance survives .nl -> native -> .nl unchanged."""
    m = dm.from_nl(str(nl_path))
    once = dumps(m)
    reloaded = loads(once)
    assert dumps(reloaded) == once, f"{nl_path.stem}: document not stable under reload"
    assert reloaded.to_nl() == m.to_nl(), f"{nl_path.stem}: .nl differs after round trip"


def test_corpus_probe_actually_fired():
    """The corpus test is parametrized over a glob; an empty glob would pass silently."""
    assert len(CORPUS) >= 60, f"expected the in-repo MINLPLib corpus, found {len(CORPUS)} files"


@pytest.mark.slow
@pytest.mark.parametrize("stem", ["nvs01", "st_e13", "st_miqp3", "nvs04"])
def test_solve_is_bound_neutral_after_round_trip(stem):
    """node_count and objective exactly unchanged -- not merely close."""
    path = NL_DIR / f"{stem}.nl"
    if not path.exists():
        pytest.skip(f"{stem} not in the in-repo corpus")
    original = dm.from_nl(str(path))
    reloaded = loads(dumps(original))
    r1 = original.solve(time_limit=60)
    r2 = reloaded.solve(time_limit=60)
    assert r1.status == r2.status
    assert r1.node_count == r2.node_count
    assert r1.objective == r2.objective


# ── the fitted-state use case ──────────────────────────────────────────────


@pytest.mark.smoke
def test_fitted_result_travels_with_the_model(tmp_path):
    m = dm.Model("fit")
    k = m.continuous("k", shape=(2,), lb=0.0, ub=10.0)
    m.minimize(dm.sum((k - 1.5) ** 2))
    result = m.solve(time_limit=30)

    path = tmp_path / "fit.dopt"
    m.save(path, result=result)
    back = load(path)

    assert back.saved_result is not None
    assert back.saved_result.status == result.status
    assert back.saved_result.objective == result.objective
    np.testing.assert_array_equal(back.saved_result.value(back._variables[0]), result.value(k))


@pytest.mark.smoke
def test_model_saved_without_a_result_reloads_with_none(tmp_path):
    m = _kinetics_model()
    path = tmp_path / "bare.dopt"
    m.save(path)
    assert load(path).saved_result is None


@pytest.mark.smoke
def test_gzip_is_detected_by_magic_bytes_not_filename(tmp_path):
    """A document stays readable after someone renames it."""
    m = _kinetics_model()
    plain, gz = tmp_path / "m.dopt", tmp_path / "m.dopt.gz"
    m.save(plain)
    m.save(gz)
    assert gz.stat().st_size < plain.stat().st_size

    renamed = tmp_path / "no_suffix.bin"
    gz.rename(renamed)
    assert dumps(load(renamed)) == dumps(load(plain)) == dumps(m)


@pytest.mark.smoke
def test_parameter_values_round_trip():
    m = dm.Model("param")
    x = m.continuous("x", lb=0, ub=10)
    T = m.parameter("T", 3.25)
    V = m.parameter("V", np.array([[1.5, 2.5], [3.5, 4.5]]))
    m.minimize((x - T) ** 2)
    back = loads(dumps(m))
    assert float(back._parameters[0].value) == 3.25
    np.testing.assert_array_equal(back._parameters[1].value, V.value)


# ── structure that `.nl` cannot carry ──────────────────────────────────────


@pytest.mark.smoke
def test_relations_survive_that_nl_refuses():
    m = dm.Model("relations")
    x = m.continuous("x", shape=(3,), lb=0, ub=20)
    y = m.binary("y", shape=(2,))
    m.minimize(dm.sum(x))
    m.if_then(y[0], [x[0] >= 10, x[1] <= 5], name="unit0")
    m.sos1([x], name="pick_one")
    m.either_or([[x[2] >= 8], [x[2] <= 2]], name="split")

    # The premise: `.nl` refuses this model outright.
    with pytest.raises(ValueError, match="indicator"):
        m.to_nl()

    once = dumps(m)
    back = loads(once)
    assert dumps(back) == once
    assert [type(c).__name__ for c in back._constraints] == [
        type(c).__name__ for c in m._constraints
    ]
    ind = next(c for c in back._constraints if type(c) is _IndicatorConstraint)
    # `if_then(y[0], ...)` stores an IndexExpression, not a bare Variable.
    assert ind.indicator.base is back._variables[1]
    assert ind.active_value == 1
    sos = next(c for c in back._constraints if type(c) is _SOSConstraint)
    assert sos.sos_type == 1 and sos.name == "pick_one"
    dis = next(c for c in back._constraints if type(c) is _DisjunctiveConstraint)
    assert len(dis.disjuncts) == 2
    assert (
        dis.semantics
        is next(c for c in m._constraints if type(c) is _DisjunctiveConstraint).semantics
    )


@pytest.mark.smoke
def test_interleaved_rows_keep_their_declaration_order():
    """`_constraints` is heterogeneous and ordered: relations sit between algebraic rows.

    Reloading section-by-section would silently reorder such a model, which changes
    `.nl` row order and any downstream row indexing.
    """
    m = dm.Model("interleaved")
    x = m.continuous("x", shape=(3,), lb=0, ub=20)
    y = m.binary("y", shape=(2,))
    m.minimize(dm.sum(x))
    m.subject_to(x[0] + x[1] <= 12.0)
    m.either_or([[x[2] >= 8], [x[2] <= 2]], name="split")
    m.subject_to(x[1] - x[2] == 0.0)
    m.if_then(y[0], [x[0] >= 10], name="unit0")
    m.subject_to(x[0] + x[2] <= 15.0)

    kinds = [type(c).__name__ for c in m._constraints]
    assert kinds == [
        "Constraint",
        "_DisjunctiveConstraint",
        "Constraint",
        "_IndicatorConstraint",
        "Constraint",
    ], "the fixture must actually interleave relations with algebraic rows"

    back = loads(dumps(m))
    assert [type(c).__name__ for c in back._constraints] == kinds
    assert [getattr(c, "name", None) for c in back._constraints] == [
        getattr(c, "name", None) for c in m._constraints
    ]


@pytest.mark.smoke
def test_dag_sharing_is_preserved_not_re_expanded():
    """A shared subexpression is written once and comes back as one object."""
    m = dm.Model("shared")
    v = m.continuous("v", lb=0, ub=5)
    shared = dm.exp(v) + 1.0
    m.minimize(shared * shared + shared)
    for _ in range(20):
        m.subject_to(shared <= 1e6)

    doc = json.loads(dumps(m))
    n_exp = sum(1 for nd in doc["nodes"] if nd.get("op") == "call" and nd.get("f") == "exp")
    assert n_exp == 1, "the shared subexpression was re-expanded per reference"

    back = loads(dumps(m))
    shared_objects = {id(c.body.left) for c in back._constraints}
    assert len(shared_objects) == 1, "sharing was not restored on load"


# ── builder-resident rows (the X-1 hazard) ─────────────────────────────────


def _fast_api_model(objective: str):
    """A model whose rows live ONLY in the Rust builder, never in `_constraints`."""
    m = dm.Model("fast")
    x = m.continuous("x", shape=(4,), lb=0.0, ub=10.0)
    A = sp.csr_matrix(np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 2.0], [1.0, 0.0, 0.0, -1.0]]))
    m.add_linear_constraints(A, x, "<=", np.array([5.0, 8.0, 3.0]), name="blk")
    if objective == "linear":
        m.add_linear_objective(np.array([1.0, 2.0, 3.0, 4.0]), x, constant=1.5, sense="minimize")
    else:
        m.add_quadratic_objective(
            sp.csr_matrix(np.diag([2.0, 2.0, 2.0, 2.0])),
            np.array([-1.0, 0.0, 1.0, 0.0]),
            x,
            constant=0.25,
            sense="minimize",
        )
    return m


@pytest.mark.smoke
@pytest.mark.parametrize("objective", ["linear", "quadratic"])
def test_builder_resident_rows_survive(objective):
    m = _fast_api_model(objective)
    assert len(m._constraints) == 0 and len(m._builder_linear_blocks) == 1

    once = dumps(m)
    back = loads(once)
    assert dumps(back) == once
    assert len(back._builder_linear_blocks) == 1, "blocks must stay builder-resident"
    assert len(back._constraints) == 0, "blocks must not be relocated into _constraints"
    assert back.num_constraints == m.num_constraints
    assert back.to_nl() == m.to_nl()

    r1, r2 = m.solve(time_limit=60), back.solve(time_limit=60)
    assert r1.status == r2.status
    assert r1.node_count == r2.node_count
    assert r1.objective == r2.objective


@pytest.mark.smoke
def test_dropping_a_builder_block_changes_the_nl():
    """Negative control: proves the .nl comparison above can actually detect a loss."""
    m = _fast_api_model("linear")
    doc = json.loads(dumps(m))
    doc["builder_blocks"] = []
    assert loads(json.dumps(doc)).to_nl() != m.to_nl()


# ── bounds are preserved literally ─────────────────────────────────────────


@pytest.mark.smoke
def test_non_finite_and_sentinel_bounds_are_preserved_literally():
    """`1e20` is the Rust LP layer's INF sentinel and must not be normalised."""
    m = dm.Model("bounds")
    w = m.continuous(
        "w", shape=(3,), lb=np.array([-np.inf, 1e20, -1e20]), ub=np.array([np.inf, 1e20, 1e20])
    )
    m.minimize(dm.sum(w))
    back = loads(dumps(m))
    np.testing.assert_array_equal(back._variables[0].lb, m._variables[0].lb)
    np.testing.assert_array_equal(back._variables[0].ub, m._variables[0].ub)


@pytest.mark.smoke
def test_integer_bounds_are_not_re_defaulted_on_load():
    """`Model.integer` applies a [0, 1e6] fallback + warning for an unspecified bound.

    A reload must reproduce the bounds that were *saved*, so it goes through the
    low-level constructor rather than the declaration API.
    """
    m = dm.Model("ints")
    n = m.integer("n", lb=-40, ub=-5)
    m.minimize(n * 1.0)
    back = loads(dumps(m))
    assert float(back._variables[0].lb) == -40.0
    assert float(back._variables[0].ub) == -5.0


@pytest.mark.smoke
def test_initial_point_round_trips():
    m = _kinetics_model()
    k = m._variables[0]
    m.set_initial_point({k: np.array([0.5, 1.0, 1.5])})
    back = loads(dumps(m))
    assert back._initial_point == m._initial_point
    assert back._initial_point, "the starting point should not be empty"


# ── refusals: never a silent drop ──────────────────────────────────────────


@pytest.mark.smoke
def test_custom_call_is_refused_by_name():
    m = dm.Model("custom")
    z = m.continuous("z", lb=0, ub=1)
    m.minimize(dm.custom(lambda v: v * 2, name="dbl")(z))
    with pytest.raises(SerializationError, match="dbl"):
        dumps(m)


@pytest.mark.smoke
def test_unknown_op_is_refused_rather_than_skipped():
    doc = json.loads(dumps(_kinetics_model()))
    doc["nodes"][0]["op"] = "quantum_frobnicate"
    with pytest.raises(SerializationError, match="unknown op"):
        loads(json.dumps(doc))


@pytest.mark.smoke
def test_future_schema_major_is_refused():
    doc = json.loads(dumps(_kinetics_model()))
    doc["schema"] = "discopt.model/99"
    with pytest.raises(SerializationError, match="major version"):
        loads(json.dumps(doc))


@pytest.mark.smoke
def test_foreign_document_is_refused():
    with pytest.raises(SerializationError, match="not a discopt model document"):
        loads(json.dumps({"schema": "something/else"}))


# ── complementarity relations ──────────────────────────────────────────────
#
# A relation lives outside `_constraints`, and whether THIS model already carries
# the rows encoding it is separate per-model state. Both halves have to survive:
# losing the marks makes a lowered model refuse to solve, and losing the *method*
# silently promotes a relaxation to an exact encoding.


def _mpcc(name="pair0"):
    """min (x-1)^2 + (y-1)^2 s.t. 0 <= x _|_ y >= 0."""
    m = dm.Model("mpcc")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    pair = m.complementarity(x, y, name=name)
    return m, x, y, pair


@pytest.mark.smoke
def test_complementarity_relation_round_trips():
    from discopt.mpec import resolve_source_variables

    m, x, y, pair = _mpcc()
    once = dumps(m)
    back = loads(once)
    assert dumps(back) == once

    assert len(back._complementarities) == 1
    rel = back._complementarities[0]
    assert rel.name == pair.name
    assert rel.role is pair.role
    assert rel.f_bounds == pair.f_bounds
    assert rel.g_bounds == pair.g_bounds
    assert rel.scale == pair.scale
    assert rel.parent == pair.parent
    assert rel.is_symmetric_nonnegative == pair.is_symmetric_nonnegative

    # Provenance: the relation must resolve to the RELOADED variables by identity,
    # not to a stale object or a name-based match.
    resolved = resolve_source_variables(back, rel, context="test")
    assert [id(v) for v in resolved] == [id(v) for v in back._variables]


@pytest.mark.smoke
def test_lowering_mark_and_method_survive():
    from discopt.mpec import unlowered_relations

    m, _, _, _ = _mpcc()
    before = m.solve(time_limit=60)
    methods = sorted(r.method for r in m._lowered_complementarities.values())
    assert methods, "the fixture must have produced a lowering mark"

    back = loads(dumps(m))
    assert sorted(r.method for r in back._lowered_complementarities.values()) == methods
    assert unlowered_relations(back) == []

    after = back.solve(time_limit=60)
    assert after.status == before.status
    assert after.node_count == before.node_count
    assert after.objective == before.objective


@pytest.mark.smoke
def test_unlowered_relation_is_not_marked_lowered():
    """The negative control: proves the loader does not mark everything as lowered.

    The solver boundary refuses a declared-but-unlowered relation, because solving
    it would silently drop the condition and certify the answer. That refusal must
    survive the round trip.
    """
    from discopt.mpec import complementarity, register_relations, unlowered_relations

    m = dm.Model("declared_only")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    register_relations(m, [complementarity(x, y, name="never_lowered")])
    assert len(unlowered_relations(m)) == 1

    once = dumps(m)
    back = loads(once)
    assert dumps(back) == once
    assert len(unlowered_relations(back)) == 1
    assert back._lowered_complementarities == {}

    with pytest.raises(NotImplementedError, match="no lowering"):
        back.solve(time_limit=30)


@pytest.mark.smoke
def test_a_relaxing_lowering_is_not_promoted_to_exact():
    """`scholtes` RELAXES the relation; reloading must not report it as an encoding.

    `LoweringRecord.is_exact` is derived from the method alone, so a mark that lost
    its method would read as exact and the solver would certify a relaxation as the
    declared model -- a false certificate.
    """
    from discopt.mpec import complementarity, register_relations, relaxed_relations, solve_mpec

    m = dm.Model("scholtes")
    a = m.continuous("a", lb=0, ub=10)
    b = m.continuous("b", lb=0, ub=10)
    m.minimize((a - 1) ** 2 + (b - 1) ** 2)
    pair = complementarity(a, b, name="relaxed_pair")
    register_relations(m, [pair])
    solve_mpec(m, [pair], method="scholtes")

    records = list(m._lowered_complementarities.values())
    assert records and not records[0].is_exact, "fixture must produce a relaxing lowering"
    assert len(relaxed_relations(m)) == 1

    back = loads(dumps(m))
    reloaded = list(back._lowered_complementarities.values())
    assert [r.method for r in reloaded] == [r.method for r in records]
    assert [r.is_exact for r in reloaded] == [r.is_exact for r in records]
    assert len(relaxed_relations(back)) == 1


@pytest.mark.smoke
def test_vector_relation_keeps_its_shape():
    m = dm.Model("vec_mpcc")
    u = m.continuous("u", shape=(3,), lb=0, ub=10)
    v = m.continuous("v", shape=(3,), lb=0, ub=10)
    m.minimize(dm.sum((u - 1.0) ** 2) + dm.sum((v - 2.0) ** 2))
    declared = m.complementarity(u, v, name="vecpair")

    back = loads(dumps(m))
    rel = back._complementarities[0]
    assert rel.f_shape == declared.f_shape == (3,)
    assert rel.g_shape == declared.g_shape
    assert rel.index == declared.index

    before, after = m.solve(time_limit=60), back.solve(time_limit=60)
    assert before.status == after.status
    assert before.node_count == after.node_count
    assert before.objective == after.objective


@pytest.mark.smoke
def test_source_link_carries_a_parent_outside_the_declared_list():
    """An element relation's `source` parent need not itself be declared on the model.

    `Complementarity` is public and constructible, so a caller can register only the
    element. Dropping its parent would break attribution back to the declared
    relation (`describe()`, the #1148 source-residual report).
    """
    from discopt.mpec import (
        Complementarity,
        ComplementarityRole,
        register_relations,
    )

    m = dm.Model("elementwise")
    u = m.continuous("u", shape=(2,), lb=0, ub=10)
    v = m.continuous("v", shape=(2,), lb=0, ub=10)
    m.minimize(dm.sum(u) + dm.sum(v))

    parent = Complementarity(u, v, name="vec", role=ComplementarityRole.NCP_PAIR)
    elements = [
        Complementarity(
            u[i],
            v[i],
            name=f"vec[{i}]",
            role=ComplementarityRole.NCP_PAIR,
            f_shape=(),
            g_shape=(),
            index=(i,),
            source=parent,
        )
        for i in range(2)
    ]
    register_relations(m, elements)  # only the elements are declared
    assert all(p is not parent for p in m._complementarities)

    once = dumps(m)
    back = loads(once)
    assert dumps(back) == once

    assert len(back._complementarities) == 2
    parents = {id(rel.source) for rel in back._complementarities}
    assert len(parents) == 1, "both elements must share ONE reloaded parent object"
    restored_parent = back._complementarities[0].source
    assert restored_parent is not None
    assert restored_parent.name == "vec"
    assert [rel.index for rel in back._complementarities] == [(0,), (1,)]
    assert back._complementarities[0].describe() == elements[0].describe()


# ── review follow-ups (PR #1240) ───────────────────────────────────────────


@pytest.mark.smoke
def test_saving_inside_a_fixed_scope_keeps_the_declared_domain():
    """A temporary `fix()` must not be persisted as the permanent declared box.

    Before this was handled, a model saved inside `Model.fixed(...)` reloaded with
    the pinned value as its declared bounds: it solved a *different* problem with no
    warning (18.0 -> 13.0 on this fixture), `fix_depth` was 0 so the pin could not be
    undone, and re-fixing was refused as "outside its declared bounds".
    """
    m = dm.Model("fixdemo")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.maximize(3 * x + 2 * y)
    m.subject_to(x + y <= 6)

    with m.fixed(x=1.0):
        doc = dumps(m)
        assert (float(x.lb), float(x.ub)) == (1.0, 1.0)

    back = loads(doc)
    bx = back._variables[0]

    # Saved while pinned, so the reloaded model is pinned too -- faithfully.
    assert (float(bx.lb), float(bx.ub)) == (1.0, 1.0)
    assert bx.fix_depth == 1

    # ... and, unlike before, the pin is reversible and the declared domain is intact.
    bx.unfix()
    assert (float(bx.lb), float(bx.ub)) == (0.0, 10.0)
    assert bx.fix_depth == 0
    bx.fix(5.0)  # would have raised "outside its declared bounds"
    assert (float(bx.lb), float(bx.ub)) == (5.0, 5.0)


@pytest.mark.smoke
def test_a_model_saved_outside_a_fixed_scope_solves_identically():
    m = dm.Model("fixdemo2")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.maximize(3 * x + 2 * y)
    m.subject_to(x + y <= 6)
    with m.fixed(x=1.0):
        pass  # scope exited: the box is the declared one again

    back = loads(dumps(m))
    assert back._variables[0].fix_depth == 0
    before, after = m.solve(time_limit=60), back.solve(time_limit=60)
    assert before.objective == after.objective
    assert before.node_count == after.node_count


@pytest.mark.smoke
def test_decomposition_annotations_survive():
    """Declared Benders/Lagrangian structure must not be replaced by auto-detection."""
    m = dm.Model("decomp")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(x + y)
    link = x + y >= 3
    m.subject_to(link)
    m.set_stage(x, 1)
    m.set_stage(y, 2)
    m.set_block(y, 0)
    m.mark_coupling(link)

    once = dumps(m)
    back = loads(once)
    assert dumps(back) == once
    assert back._decomp_stages == m._decomp_stages
    assert back._decomp_blocks == m._decomp_blocks
    # `mark_coupling` records id(constraint) for the object form; a raw id is a
    # process-local address, so it must come back pointing at the REBUILT row.
    assert id(back._constraints[0]) in back._coupling_keys
    assert len(back._coupling_keys) == len(m._coupling_keys)


@pytest.mark.smoke
def test_named_sets_and_aux_counter_survive():
    m = dm.Model("sets")
    plants = m.set("plants", ["pitt", "sf"])
    links = m.set("links", [("pitt", "a"), ("sf", "b")])
    x = m.continuous("x", over=plants, lb=0, ub=10)
    m.minimize(sum(x[k] for k in plants))
    m._aux_counter = 7
    m._zero_spanning_factor_auxes = {"aux3", "aux1"}

    back = loads(dumps(m))
    assert [s.name for s in back._sets] == [s.name for s in m._sets]
    assert [s.dimen for s in back._sets] == [s.dimen for s in m._sets]
    assert list(back._sets[0].members) == list(plants.members)
    assert list(back._sets[1].members) == list(links.members), "tuple members must stay tuples"
    assert back._aux_counter == 7
    assert back._zero_spanning_factor_auxes == {"aux1", "aux3"}


def test_every_model_attribute_is_accounted_for():
    """A field added to `Model.__init__` must not silently start being dropped.

    `loads` restores section by section, so an unhandled attribute is reset to its
    default with no error -- which for the decomposition annotations meant a user's
    declared structure being quietly replaced by auto-detection. This fails until
    the new field is classified in `_MODEL_STATE`.
    """
    from discopt.serialize import _MODEL_STATE

    live = set(dm.Model("probe").__dict__)
    unhandled = live - _MODEL_STATE
    assert not unhandled, (
        f"Model gained attribute(s) {sorted(unhandled)} that discopt.serialize does not "
        "account for. Add each to the right bucket in _MODEL_STATE (carried in its own "
        "section, derived on load, or carried in the 'state' section) -- or refuse it."
    )
    stale = _MODEL_STATE - live
    assert not stale, f"_MODEL_STATE lists attribute(s) Model no longer has: {sorted(stale)}"


def test_every_produced_operator_name_is_accepted():
    """Read-time operator validation must not refuse a name the modeling layer emits.

    Scans the package for every `BinaryOp("...")` / `UnaryOp("...")` / `FunctionCall("...")`
    literal. A name missing from the accepted sets would make `loads` refuse a valid
    document, so this is the guard against the validation drifting shut.
    """
    import re
    from pathlib import Path

    from discopt.serialize import _KNOWN_BINARY_OPS, _KNOWN_UNARY_OPS, _known_funcs

    root = Path(__file__).resolve().parents[1] / "discopt"
    binary, unary, funcs = set(), set(), set()
    scanned = 0
    for path in root.rglob("*.py"):
        text = path.read_text()
        scanned += 1
        binary |= set(re.findall(r'BinaryOp\(\s*"([^"]+)"', text))
        unary |= set(re.findall(r'UnaryOp\(\s*"([^"]+)"', text))
        funcs |= set(re.findall(r'FunctionCall\(\s*"([a-z_0-9]+)"', text))

    assert scanned > 0, "the scan found no source files"
    assert binary and unary and funcs, "the scan matched no operator literals"
    assert binary <= _KNOWN_BINARY_OPS, (
        f"unaccepted binary op(s): {sorted(binary - _KNOWN_BINARY_OPS)}"
    )
    assert unary <= _KNOWN_UNARY_OPS, f"unaccepted unary op(s): {sorted(unary - _KNOWN_UNARY_OPS)}"
    known = _known_funcs()
    assert funcs <= known, f"unaccepted function name(s): {sorted(funcs - known)}"


@pytest.mark.smoke
@pytest.mark.parametrize(
    "field,bad",
    [
        ("binop", {"op": "binop", "o": "frobnicate", "a": 0, "b": 0}),
        ("unop", {"op": "unop", "o": "frobnicate", "a": 0}),
        ("call", {"op": "call", "f": "frobnicate", "args": [0]}),
    ],
)
def test_unknown_operator_name_is_refused_on_read(field, bad):
    """Previously these loaded fine and failed during a solve, where the first
    symptom is the incumbent-verification snapshot failing and the false-primal
    guard being disabled for that solve."""
    doc = json.loads(dumps(_kinetics_model()))
    doc["nodes"].append(bad)
    with pytest.raises(
        SerializationError, match="unknown (binary|unary) operator|unknown function"
    ):
        loads(json.dumps(doc))


@pytest.mark.smoke
def test_a_future_minor_schema_is_readable_but_a_future_major_is_not():
    """A minor bump is additive by construction, so "1.1" reads as major 1."""
    doc = json.loads(dumps(_kinetics_model()))
    doc["schema"] = "discopt.model/1.1"
    assert loads(json.dumps(doc)) is not None

    doc["schema"] = "discopt.model/2.0"
    with pytest.raises(SerializationError, match="major version 2"):
        loads(json.dumps(doc))

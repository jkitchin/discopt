"""Regression tests for modeling-module review findings M4–M8 (issue #413 tail).

Each test pins a *class* of misbehavior the review documented, so a future
reintroduction trips it. All are fast (no or a single tiny solve) and run on
every PR.

- M4: ``Expression.__eq__`` builds a Constraint (the modeling DSL), which used to
  make expressions unhashable and ``expr in [other]`` silently True. Fix:
  identity hashing on expressions + ``Constraint.__bool__`` raising.
- M5: ``subject_to`` mutated the caller's Constraint (``c.name = name``) and never
  validated duplicate constraint names (though duals are name-keyed). Fix:
  copy-on-name + duplicate-name rejection in ``validate()``.
- M6: misspelled ``solve()`` kwargs were silently swallowed. Fix: validate against
  the accepted set and raise ``TypeError`` with a near-match hint.
- M7: ``_check_name`` rebuilt the full name set per declaration (O(n²)). Fix:
  persistent ``_names`` set (O(1) per declaration). Bound-neutral.
- M8: incompatible operand shapes were not caught at build time. Fix: conservative
  broadcast check at ``BinaryOp`` construction (only when both shapes are known).
"""

import time

import discopt.modeling as dm
import numpy as np
import pytest
from discopt import Model
from discopt.modeling.core import Constraint, _known_shape
from discopt.solver import solve_model_accepted_kwargs

# ─────────────────────────────────────────────────────────────
# M4 — hashing / membership / boolean context
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_m4_expressions_are_hashable_by_identity():
    m = Model()
    x = m.continuous("x")
    y = m.continuous("y")
    u = x + y
    v = x * y
    # Before the fix, non-Variable expressions were unhashable (defining __eq__
    # dropped __hash__), so this raised TypeError.
    d = {u: 1, v: 2}
    assert d[u] == 1 and d[v] == 2
    # Identity hashing: two structurally identical expressions are distinct keys.
    assert (x + y) not in d
    # Sets work too.
    s = {u, v, x, y}
    assert len(s) == 4


@pytest.mark.unit
def test_m4_variable_stays_hashable():
    m = Model()
    x = m.continuous("x")
    assert isinstance(hash(x), int)
    assert x in {x}


@pytest.mark.unit
def test_m4_membership_via_list_raises_not_silently_true():
    m = Model()
    x = m.continuous("x")
    y = m.continuous("y")
    u = x + y
    v = x * y
    # ``u in [v]`` truth-tests ``u == v`` which builds a Constraint; that
    # Constraint has no boolean value → loud TypeError instead of silent True.
    with pytest.raises(TypeError, match="no truth value"):
        _ = u in [v]
    # Identical object short-circuits (Python checks identity before __eq__).
    assert u in [u]


@pytest.mark.unit
def test_m4_constraint_has_no_boolean_value():
    m = Model()
    x = m.continuous("x")
    with pytest.raises(TypeError, match="no truth value"):
        bool(x == 5)
    with pytest.raises(TypeError, match="no truth value"):
        if x <= 3:  # noqa: SIM102 - intentionally exercising __bool__
            pass


@pytest.mark.unit
def test_m4_eq_still_builds_a_constraint():
    """The DSL overload must be preserved: ``x == 5`` builds a Constraint."""
    m = Model()
    x = m.continuous("x")
    c = x == 5
    assert isinstance(c, Constraint)
    assert c.sense == "=="


# ─────────────────────────────────────────────────────────────
# M5 — subject_to must not mutate the caller; dup names rejected
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_m5_re_adding_object_does_not_rename_earlier_row():
    """The M5 hazard: re-adding the SAME Constraint object must not rename the
    row already in the model (in-place mutation used to leave BOTH rows with the
    second name). The first placement may stamp the name on the object (so
    object-keyed APIs like ``mark_coupling`` still work), but a second placement
    copies rather than corrupting the earlier row.
    """
    m = Model()
    a = m.continuous("a", lb=0, ub=10)
    c = a <= 5
    m.subject_to(c, name="first")
    m.subject_to(c, name="second")
    # Earlier row keeps "first"; the second add stored an independent copy.
    assert [cc.name for cc in m._constraints] == ["first", "second"]
    # The two stored rows are distinct objects (no aliasing).
    assert m._constraints[0] is not m._constraints[1]


@pytest.mark.unit
def test_m5_named_constraint_is_not_renamed_by_readd():
    """Re-adding a constraint the caller already named under a *different* name
    must not clobber the caller's chosen name in place."""
    m = Model()
    a = m.continuous("a", lb=0, ub=10)
    c = a <= 5
    m.subject_to(c, name="original")
    assert c.name == "original"
    m.subject_to(c, name="other")
    # Caller's object keeps its own name; the model's second row is a renamed copy.
    assert c.name == "original"
    assert [cc.name for cc in m._constraints] == ["original", "other"]


@pytest.mark.unit
def test_m5_object_identity_preserved_for_first_add():
    """First placement keeps object identity so object-keyed APIs work
    (regression guard for the mark_coupling interaction)."""
    m = Model()
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    c = x + y <= 1
    m.subject_to(c, name="link")
    # The stored row IS the caller's object (identity preserved on first add).
    assert m._constraints[0] is c
    m.mark_coupling(c)
    assert id(c) in m._coupling_keys
    assert "link" in m._coupling_keys


@pytest.mark.smoke
def test_m5_duplicate_constraint_names_warn():
    """Duplicate constraint names *warn* (not raise): indexed/array constraint
    families legitimately reuse a base name, so a hard error would break valid
    models. The warning still surfaces the ``constraint_duals`` name-collision
    hazard. (Rejecting broke the GAMS ``supply(i)`` transport family — CI-caught.)
    """
    m = Model()
    b = m.continuous("b", lb=0, ub=10)
    m.minimize(b)
    m.subject_to(b <= 1, name="dup")
    m.subject_to(b >= 0, name="dup")
    with pytest.warns(UserWarning, match="Duplicate constraint name"):
        m.validate()


@pytest.mark.smoke
def test_m5_unnamed_constraints_allowed():
    m = Model()
    d = m.continuous("d", lb=0, ub=10)
    m.minimize(d)
    m.subject_to(d <= 1)
    m.subject_to(d >= 0)
    m.validate()  # anonymous rows are exempt from the uniqueness rule


@pytest.mark.smoke
def test_m5_gams_indexed_equation_family_names_uniquely():
    """A GAMS indexed equation ``supply(i)`` expands to one row per index; each
    row must get a distinct name (``supply[p1]``, ``supply[p2]``) so the M5
    duplicate-name check does not spuriously reject a legitimate model and
    ``constraint_duals`` name-keying stays sound (regression for the gams-link CI
    failure). Without per-index naming all rows were named ``supply`` and
    ``validate()`` raised.
    """
    src = """
Sets i / p1, p2 /
     j / m1, m2 / ;
Parameter cap(i) / p1 20, p2 30 / ;
Parameter dem(j) / m1 25, m2 15 / ;
Table c(i,j)
          m1   m2
    p1     2    3
    p2     4    1 ;
Positive Variable x(i,j) ;
Free Variable z ;
Equations cost, supply(i), demand(j) ;
cost..      z =e= sum((i,j), c(i,j) * x(i,j)) ;
supply(i).. sum(j, x(i,j)) =l= cap(i) ;
demand(j).. sum(i, x(i,j)) =g= dem(j) ;
Model transport / all / ;
Solve transport using LP minimizing z ;
"""
    from discopt.modeling.gams_parser import parse_gams

    m = parse_gams(src)
    names = [c.name for c in m._constraints]
    # Each indexed row has its own key-qualified name; no duplicates.
    assert names == ["supply[p1]", "supply[p2]", "demand[m1]", "demand[m2]"]
    assert len(names) == len(set(names))
    m.validate()  # must not raise a spurious duplicate-name error


# ─────────────────────────────────────────────────────────────
# M6 — solve() kwarg validation
# ─────────────────────────────────────────────────────────────


@pytest.mark.smoke
def test_m6_misspelled_solve_kwarg_raises():
    m = Model()
    z = m.continuous("z", lb=0, ub=5)
    m.minimize(z)
    m.subject_to(z >= 1)
    with pytest.raises(TypeError, match="unexpected keyword argument 'gap_tolerence'"):
        m.solve(gap_tolerence=1e-3)


@pytest.mark.smoke
def test_m6_near_match_suggestion():
    m = Model()
    z = m.continuous("z", lb=0, ub=5)
    m.minimize(z)
    m.subject_to(z >= 1)
    with pytest.raises(TypeError, match="Did you mean 'gap_tolerance'"):
        m.solve(gap_tolerence=1e-3)


@pytest.mark.smoke
def test_m6_valid_solve_kwargs_still_work():
    m = Model()
    z = m.continuous("z", lb=0, ub=5)
    m.minimize(z)
    m.subject_to(z >= 1)
    # A known solve() parameter and a forwarded solve_model parameter both work.
    r1 = m.solve(gap_tolerance=1e-3)
    r2 = m.solve(max_nodes=50)
    assert r1.status == "optimal"
    assert r2.status == "optimal"


@pytest.mark.unit
def test_m6_allowlist_covers_solver_and_backend_kwargs():
    """The allowlist must include solve_model params and backend passthrough keys.

    If a legitimate kwarg is missing here, real solves/benchmarks would be
    rejected as typos — a false refusal that breaks the panel.
    """
    allowed = solve_model_accepted_kwargs()
    must_have = {
        # solve_model named params
        "gap_tolerance",
        "max_nodes",
        "strategy",
        "nlp_solver",
        "rlt",
        "cuts",
        "presolve",
        "batch_size",
        "mccormick_bounds",
        "root_cut_rounds",
        # backend passthrough
        "gurobi_options",
        "mip_nlp_method",
        "mip_nlp_options",
        "equality_relaxation",
        "ecp_mode",
        "feasibility_cuts",
        "milp_solver",
        "rel_gap",
        "abs_tol",
        "obbt_at_root",
        "iteration_callback",
        "lp_spatial",
    }
    missing = must_have - allowed
    assert not missing, f"allowlist missing legitimate kwargs: {sorted(missing)}"


# ─────────────────────────────────────────────────────────────
# M7 — O(1) name check (bound-neutral perf)
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_m7_duplicate_name_still_rejected():
    m = Model()
    m.continuous("a")
    with pytest.raises(ValueError, match="already used"):
        m.continuous("a")
    with pytest.raises(ValueError, match="already used"):
        m.parameter("a", value=1.0)


@pytest.mark.unit
def test_m7_name_check_is_not_quadratic():
    """Declaration cost must be ~O(1)/var, not growing with model size (M7).

    Compares per-var declaration time at n and 4n; the old O(n²) name rebuild
    made the 4n cost ~4× the n cost. Allow generous headroom for noise but
    reject clear super-linear growth.
    """

    def per_var_us(n):
        m = Model()
        t0 = time.perf_counter()
        for i in range(n):
            m.continuous(f"x{i}")
        return (time.perf_counter() - t0) / n * 1e6

    small = per_var_us(1000)
    large = per_var_us(4000)
    # O(1): large ≈ small. O(n²): large ≈ 4× small. Fail only on clear quadratic.
    assert large < small * 2.5, (
        f"name check looks super-linear: {small:.1f} us/var at n=1k vs {large:.1f} us/var at n=4k"
    )


@pytest.mark.unit
def test_m7_rebuild_name_index_after_bulk_reassignment():
    """A model that bulk-reassigns _variables must resync its name cache."""
    m = Model()
    m.continuous("keep")
    m.continuous("gone")
    # Simulate a reformulation pass that drops a variable then rebuilds the index.
    m._variables = [v for v in m._variables if v.name != "gone"]
    m._rebuild_name_index()
    assert "gone" not in m._names
    # The freed name is now declarable again.
    m.continuous("gone")
    assert "gone" in m._names


# ─────────────────────────────────────────────────────────────
# M8 — build-time shape checking (conservative)
# ─────────────────────────────────────────────────────────────


@pytest.mark.unit
def test_m8_incompatible_shapes_raise_at_build_time():
    m = Model()
    p = m.continuous("p", shape=(3,))
    q = m.continuous("q", shape=(2,))
    with pytest.raises(ValueError, match="do not broadcast"):
        _ = p + q
    with pytest.raises(ValueError, match="do not broadcast"):
        _ = p * q
    # And in a constraint context.
    with pytest.raises(ValueError, match="do not broadcast"):
        _ = p + q <= 1


@pytest.mark.unit
def test_m8_shape_mismatch_propagates_through_unary():
    m = Model()
    p = m.continuous("p", shape=(3,))
    q = m.continuous("q", shape=(2,))
    with pytest.raises(ValueError, match="do not broadcast"):
        _ = (-p) + q


@pytest.mark.unit
def test_m8_valid_broadcasts_do_not_raise():
    m = Model()
    a = m.continuous("a", shape=(3,))
    b = m.continuous("b", shape=(3,))
    s = m.continuous("s")
    c1 = m.continuous("c1", shape=(1,))
    assert _known_shape(a + b) == (3,)
    assert _known_shape(a + s) == (3,)  # scalar broadcasts
    assert _known_shape(a + 2.0) == (3,)  # python scalar broadcasts
    assert _known_shape(a + c1) == (3,)  # (3,) + (1,) broadcasts
    assert _known_shape(s + 1.0) == ()


@pytest.mark.unit
def test_m8_unknown_shape_never_falsely_raises():
    """Nodes with unknowable shape (matmul, reductions) must not trigger a check."""
    m = Model()
    a = m.continuous("a", shape=(3,))
    q = m.continuous("q", shape=(2,))
    M = np.ones((2, 3))
    mm = M @ a  # MatMul → unknown static shape
    assert _known_shape(mm) is None
    # unknown + (2,) must NOT raise even though (3,)+(2,) would.
    _ = mm + q
    sm = dm.sum(a)  # reduction → unknown static shape
    assert _known_shape(sm) is None
    _ = sm + q  # must not raise

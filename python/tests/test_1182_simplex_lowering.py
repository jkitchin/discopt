"""Issue #1182 — the exact continuous (simplex/CNF) lowering of disjunctions.

Pins Theorem 1 of arXiv:2601.03906v1 as implemented in
:mod:`discopt._relax.simplex_lowering`, and each requirement #1182 carries
forward from @bernalde's comment on #1148:

1. residuals are measured on the **original predicates** at the returned source
   point, and the simplex weights are never reported as failed Boolean
   integrality nor as recovered named Boolean assignments;
3. the overlapping-disjunct and strict-boundary fixtures are reused, with a
   **fractional-witness control** and a **stale-report control** — neither
   weighted-row feasibility nor a report taken at another point may stand in for
   source validation;
4. CNF clauses, literal occurrences, weight variables and rows are counted
   **separately**.

Requirement 2 (the local-vs-certified distinction) has no surface here by
construction: the lowering is exact in projection, this path introduces no local
NLP arm, and a certified solve of the lowered model is a certificate for the
source. The test ``test_a_simplex_solve_is_certified_and_agrees_with_big_m``
is what holds that claim to a measurement.

The discriminator for overlap/boundary is #1124's, reused verbatim:

    min (x - 1/2)^2   s.t.   [x <= 1] or [x >= 0],   x in [-2, 3]

whose optimum is 0 at x = 1/2 under SELECT_ONE, a point lying in **both**
disjuncts — which is exactly the case a "recovered Boolean assignment" gets
wrong.

Per CLAUDE.md §6 the module counts its executed source-residual comparisons and
a ``teardown_module`` hook fails the run if that count is zero. It is a teardown
hook and not a test because this suite runs under ``pytest-randomly``: as a test
it would be position-dependent, and a shuffle that ran it first would make the
guard against measuring nothing itself measure nothing.
"""

import math

import discopt.modeling as dm
import numpy as np
import pytest
from discopt._relax.gdp_reformulate import (
    HullPerspectiveOriginError,
    reformulate_gdp,
)
from discopt._relax.simplex_lowering import (
    MAX_CNF_CLAUSES,
    SIMPLEX_WEIGHT_PREFIX,
    DisjunctionResidualReport,
    SimplexLoweringRefused,
    disjunction_residuals,
    lower_disjunction_simplex,
    selected_disjuncts,
)
from discopt.modeling.core import (
    DisjunctionSemantics,
    VarType,
    _DisjunctiveConstraint,
)

_COMPARISONS = [0]


def _residuals(model, point):
    """``disjunction_residuals`` plus the executed-comparison bookkeeping."""
    report = disjunction_residuals(model, point)
    _COMPARISONS[0] += report.comparisons
    return report


def teardown_module(module):  # noqa: ARG001
    assert _COMPARISONS[0] > 0, (
        "no source-predicate comparison was executed; a suite that measures "
        "nothing reports 0 violations and reads as a pass (CLAUDE.md §6)"
    )
    print(f"\nexecuted source-predicate comparisons: {_COMPARISONS[0]}")


# ── fixtures reused from the semantic-contract work (#1124) ──────────────────


def _overlap_model():
    """min (x - 1/2)^2 s.t. [x <= 1] or [x >= 0]; optimum 0 at x = 1/2, in BOTH."""
    m = dm.Model("overlap")
    x = m.continuous("x", lb=-2.0, ub=3.0)
    m.minimize((x - 0.5) * (x - 0.5))
    m.either_or([[x <= 1.0], [x >= 0.0]], name="ov")
    return m, x


def _boundary_model():
    """min x s.t. [x >= 1] or [x >= 2]; optimum 1 sits exactly ON the predicate."""
    m = dm.Model("boundary")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    m.minimize(x)
    m.either_or([[x >= 1.0], [x >= 2.0]], name="bd")
    return m, x


def _disjunction(model):
    return next(c for c in model._constraints if isinstance(c, _DisjunctiveConstraint))


def _weights(model):
    return [v for v in model._variables if v.name.startswith(SIMPLEX_WEIGHT_PREFIX)]


# ── requirement 4: the four size quantities, counted separately ──────────────


@pytest.mark.unit
def test_size_quantities_are_reported_separately_not_summed():
    """A J-way disjunction of EQUALITIES is 2**J clauses, not J rows.

    Each ``h == 0`` is two predicates, and CNF distribution is multiplicative
    over the disjuncts, so the clause count is what explodes while the *declared*
    row count does not move. Requirement 4 exists because summing these hides
    exactly that.
    """
    m = dm.Model("grid")
    a = m.continuous("a", lb=-1.0, ub=1.0)
    m.minimize(a)
    m.either_or([[a == v] for v in (-0.5, 0.0, 0.5)], name="grid")

    seen = []
    record, rows = lower_disjunction_simplex(
        _disjunction(m), lambda size: _fake_weight(m, size, seen)
    )
    assert record.sizes.cnf_clauses == 2**3 == 8
    assert record.sizes.literal_occurrences == 8 * 3 == 24
    assert record.sizes.weight_variables == 8 * 3 == 24
    assert record.sizes.rows == 2 * 8 == 16
    assert len(rows) == 16
    # Four distinct numbers, deliberately not one "model size".
    assert len({record.sizes.cnf_clauses, record.sizes.literal_occurrences, record.sizes.rows}) == 3


def _fake_weight(model, size, seen):
    from discopt.modeling.core import Variable

    v = Variable(
        f"{SIMPLEX_WEIGHT_PREFIX}t{len(seen)}", VarType.CONTINUOUS, (size,), 0.0, 1.0, model
    )
    model._variables.append(v)
    seen.append(v)
    return v


@pytest.mark.unit
def test_single_predicate_disjuncts_do_not_blow_up():
    """The paper's optimal-control shape: one clause, J weights, 2 rows."""
    m = dm.Model("single")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    m.minimize(x * x)
    m.either_or([[x <= -1.0], [x >= 1.0]], name="gap")
    seen = []
    record, rows = lower_disjunction_simplex(
        _disjunction(m), lambda size: _fake_weight(m, size, seen)
    )
    assert (
        record.sizes.cnf_clauses,
        record.sizes.literal_occurrences,
        record.sizes.weight_variables,
        record.sizes.rows,
    ) == (1, 2, 2, 2)
    assert len(rows) == 2


@pytest.mark.unit
def test_clause_budget_refuses_loudly_and_names_the_cost():
    """The blowup is refused, never silently expanded (CLAUDE.md §3)."""
    m = dm.Model("blowup")
    a = m.continuous("a", 12, lb=-1.0, ub=1.0)
    m.minimize(a[0])
    # 12 disjuncts of one equality each => 2**12 = 4096 clauses > MAX_CNF_CLAUSES.
    m.either_or([[a[j] == 0.0] for j in range(12)], name="big")
    seen = []
    with pytest.raises(SimplexLoweringRefused) as exc:
        lower_disjunction_simplex(_disjunction(m), lambda size: _fake_weight(m, size, seen))
    message = str(exc.value)
    assert "4096" in message and str(MAX_CNF_CLAUSES) in message
    assert "big-m" in message  # names the alternative rather than just failing


@pytest.mark.unit
def test_the_budget_is_a_parameter_not_a_hidden_constant():
    m = dm.Model("blowup2")
    a = m.continuous("a", 12, lb=-1.0, ub=1.0)
    m.minimize(a[0])
    m.either_or([[a[j] == 0.0] for j in range(12)], name="big")
    seen = []
    record, _rows = lower_disjunction_simplex(
        _disjunction(m), lambda size: _fake_weight(m, size, seen), max_clauses=4096
    )
    assert record.sizes.cnf_clauses == 4096


# ── the refusals (contract boundaries) ───────────────────────────────────────


@pytest.mark.unit
def test_reified_semantics_is_refused_not_approximated():
    """§3.1's strict negation needs an existential exponential lift.

    Serving EXACTLY_ONE_TRUE with these rows would silently return the *union*,
    which is a different feasible set (#1124's discriminator makes the gap a
    whole interval), so the lowering refuses.
    """
    m = dm.Model("reified")
    x = m.continuous("x", lb=-2.0, ub=3.0)
    m.minimize(x)
    dc = _DisjunctiveConstraint(
        disjuncts=[[x <= 1.0], [x >= 0.0]],
        name="tv",
        semantics=DisjunctionSemantics.EXACTLY_ONE_TRUE,
    )
    with pytest.raises(SimplexLoweringRefused) as exc:
        lower_disjunction_simplex(dc, lambda size: _fake_weight(m, size, []))
    assert "EXACTLY_ONE_TRUE" in str(exc.value)
    assert "REIFIED" in str(exc.value)


@pytest.mark.unit
def test_a_nested_disjunction_is_refused():
    m = dm.Model("nested")
    x = m.continuous("x", lb=-2.0, ub=3.0)
    m.minimize(x)
    inner = _DisjunctiveConstraint(disjuncts=[[x <= 0.0], [x >= 2.0]], name="inner")
    outer = _DisjunctiveConstraint(disjuncts=[[inner], [x >= 1.0]], name="outer")
    with pytest.raises(SimplexLoweringRefused, match="nested disjunction"):
        lower_disjunction_simplex(outer, lambda size: _fake_weight(m, size, []))


@pytest.mark.unit
def test_an_empty_disjunct_is_refused_rather_than_read_as_always_true():
    m = dm.Model("empty")
    x = m.continuous("x", lb=-2.0, ub=3.0)
    m.minimize(x)
    dc = _DisjunctiveConstraint(disjuncts=[[x <= 0.0], []], name="e")
    with pytest.raises(SimplexLoweringRefused, match="declares no rows"):
        lower_disjunction_simplex(dc, lambda size: _fake_weight(m, size, []))


@pytest.mark.unit
def test_a_vector_disjunct_row_is_refused_with_the_shape_named():
    m = dm.Model("vec")
    x = m.continuous("x", 3, lb=-2.0, ub=3.0)
    m.minimize(x[0])
    dc = _DisjunctiveConstraint(disjuncts=[[x <= 1.0], [x >= 2.0]], name="v")
    with pytest.raises(SimplexLoweringRefused) as exc:
        lower_disjunction_simplex(dc, lambda size: _fake_weight(m, size, []))
    assert "(3,)" in str(exc.value)


@pytest.mark.unit
def test_mip_nlp_plus_simplex_is_refused_as_contradictory():
    """No selector binary is emitted, so there is nothing for MIP-NLP to branch on."""
    m, _x = _overlap_model()
    with pytest.raises(ValueError, match="no selector binary"):
        m.solve(time_limit=5, solver="mip-nlp", gdp_method="simplex")


# ── requirement 1: the weights are witnesses, not selectors ──────────────────


@pytest.mark.unit
def test_the_weights_are_continuous_so_a_fraction_is_not_failed_integrality():
    m, _x = _overlap_model()
    lowered = reformulate_gdp(m, method="simplex")
    weights = _weights(lowered)
    assert weights, "the lowering emitted no weight variable"
    for w in weights:
        assert w.var_type is VarType.CONTINUOUS
        assert float(np.max(w.lb)) == 0.0 and float(np.max(w.ub)) == 1.0
    # and no selector binary was emitted at all
    assert not [v for v in lowered._variables if v.var_type is VarType.BINARY]


@pytest.mark.unit
def test_the_lowering_record_exposes_no_boolean_assignment():
    """Requirement 1: a witness must not be exposed as a named Boolean value."""
    m, _x = _overlap_model()
    lowered = reformulate_gdp(m, method="simplex")
    (record,) = lowered._simplex_lowerings
    for forbidden in ("selectors", "booleans", "assignment", "y", "binary"):
        assert not hasattr(record, forbidden), (
            f"SimplexLoweringRecord.{forbidden} would invite reading a Boolean "
            "assignment off an existential witness"
        )
    assert all(n.startswith(SIMPLEX_WEIGHT_PREFIX) for n in record.weight_names)


@pytest.mark.unit
def test_a_fractional_witness_satisfies_the_weighted_row_while_a_disjunct_fails():
    """The control that makes the weighted row unusable as a source report.

    At ``x = 1.4`` in the #1124 fixture, disjunct 0 (``x <= 1``) is violated by
    ``0.4`` and disjunct 1 (``x >= 0``) holds. A *fractional* witness still
    satisfies the weighted row, so a caller reading truth off that row would
    conclude both literals hold. The source report says otherwise.
    """
    m, _x = _overlap_model()
    point = {"x": np.array([1.4])}

    report = _residuals(m, point)
    (res,) = report.residuals
    assert res.per_disjunct[0] == pytest.approx(0.4)  # x - 1 = 0.4 > 0, fails
    assert res.per_disjunct[1] == pytest.approx(-1.4)  # -x <= 0 holds
    assert res.violation == pytest.approx(-1.4)  # the disjunction holds

    # A fractional witness that satisfies the weighted row at the same point.
    lam = np.array([0.5, 0.5])
    weighted = lam[0] * res.per_disjunct[0] + lam[1] * res.per_disjunct[1]
    assert weighted <= 0.0
    # ... yet disjunct 0 is genuinely violated, and the source report says so.
    assert selected_disjuncts(m, point) == {"ov": [1]}
    _COMPARISONS[0] += 2


@pytest.mark.unit
def test_overlapping_disjuncts_report_every_disjunct_that_holds():
    """Requirement 3's overlap fixture: the answer is a SET, not a choice."""
    m, _x = _overlap_model()
    point = {"x": np.array([0.5])}
    assert selected_disjuncts(m, point) == {"ov": [0, 1]}
    report = _residuals(m, point)
    assert report.max_violation < 0.0


@pytest.mark.unit
def test_a_point_exactly_on_the_boundary_is_inside_the_closed_predicate():
    """Requirement 3's strict-boundary fixture: residual 0 is satisfied, not violated."""
    m, _x = _boundary_model()
    point = {"x": np.array([1.0])}
    report = _residuals(m, point)
    (res,) = report.residuals
    assert res.per_disjunct[0] == pytest.approx(0.0)
    assert res.violation == pytest.approx(0.0)
    assert selected_disjuncts(m, point) == {"bd": [0]}


# ── requirement 3: no stale report, no empty report ──────────────────────────


@pytest.mark.unit
def test_a_report_from_another_point_cannot_stand_in_for_source_validation():
    """The report is a pure function of the point handed in; there is no cache."""
    m, _x = _overlap_model()
    good = _residuals(m, {"x": np.array([0.5])})
    bad = _residuals(m, {"x": np.array([-3.0])})  # violates x >= 0 only... see below
    # x = -3 satisfies x <= 1, so the disjunction still holds; use a point that
    # violates BOTH disjuncts of a disjoint fixture instead.
    assert good.max_violation < 0.0
    assert bad.max_violation < 0.0

    m2 = dm.Model("disjoint")
    y = m2.continuous("y", lb=-5.0, ub=5.0)
    m2.minimize(y)
    m2.either_or([[y <= -1.0], [y >= 1.0]], name="gap")
    feasible = _residuals(m2, {"y": np.array([2.0])})
    infeasible = _residuals(m2, {"y": np.array([0.0])})
    assert feasible.max_violation == pytest.approx(-1.0)
    assert infeasible.max_violation == pytest.approx(1.0)
    assert selected_disjuncts(m2, {"y": np.array([0.0])}) == {"gap": []}


@pytest.mark.unit
def test_a_report_that_measured_nothing_refuses_to_read_as_a_pass():
    m = dm.Model("nodisj")
    x = m.continuous("x", lb=0.0, ub=1.0)
    m.minimize(x)
    with pytest.raises(ValueError, match="declares no disjunction"):
        disjunction_residuals(m, {"x": np.array([0.0])})
    with pytest.raises(ValueError, match="no disjunction was measured"):
        DisjunctionResidualReport().max_violation


@pytest.mark.unit
def test_a_missing_source_variable_is_refused_not_defaulted():
    m, _x = _overlap_model()
    with pytest.raises(KeyError, match="absent from the point"):
        disjunction_residuals(m, {})


# ── exactness, against the exact GDP references ──────────────────────────────


@pytest.mark.smoke
def test_a_simplex_solve_is_certified_and_agrees_with_big_m():
    """Exact in projection: the certified optimum is the big-M optimum.

    This is also the whole of requirement 2's surface on this path — the result
    is a genuine certificate, so no local/certified distinction arises.
    """
    m_ref, _ = _overlap_model()
    reference = m_ref.solve(time_limit=30, gdp_method="big-m")
    m_sx, _ = _overlap_model()
    result = m_sx.solve(time_limit=30, gdp_method="simplex")

    assert reference.status == "optimal" and result.status == "optimal"
    assert result.gap_certified is True
    assert result.objective == pytest.approx(reference.objective, abs=1e-5)
    assert result.objective == pytest.approx(0.0, abs=1e-5)

    source, _ = _overlap_model()
    assert _residuals(source, result.x).max_violation <= 1e-6


@pytest.mark.smoke
def test_the_disjunction_is_enforced_not_merely_relaxed_away():
    """A disjoint gap the optimum would otherwise fall into stays excluded."""
    m = dm.Model("enforced")
    x = m.continuous("x", lb=-5.0, ub=5.0)
    m.minimize(x * x)  # unconstrained optimum x = 0, inside the gap
    m.either_or([[x <= -1.0], [x >= 1.0]], name="gap")
    result = m.solve(time_limit=30, gdp_method="simplex")
    assert result.status == "optimal"
    assert result.objective == pytest.approx(1.0, rel=1e-4)

    source = dm.Model("enforced")
    xs = source.continuous("x", lb=-5.0, ub=5.0)
    source.minimize(xs * xs)
    source.either_or([[xs <= -1.0], [xs >= 1.0]], name="gap")
    assert _residuals(source, result.x).max_violation <= 1e-6


# ── the capability this lowering exists for (#1182 entry condition) ──────────


def _both_refuse_model():
    """``1/x <= 1`` on a box straddling 0: unbounded enclosure AND undefined at 0.

    big-M cannot bound the body; the Furman-Sawaya-Grossmann perspective needs
    ``g(0)``, which divides by zero. This is the shape of the 18 ``stranded_gas``
    disjunct rows the #1182 corpus scan found (``log`` of a capacity sum whose
    box includes 0), reduced to its smallest reproducer.
    """
    m = dm.Model("both_refuse")
    x = m.continuous("x", lb=-10.0, ub=10.0)
    m.minimize((x - 5.0) * (x - 5.0))
    m.either_or([[1.0 / x <= 1.0], [x >= 3.0]], name="recip")
    return m


@pytest.mark.smoke
def test_big_m_and_hull_both_refuse_the_row_the_simplex_lowering_certifies():
    with pytest.raises(ValueError, match="cannot bound the body"):
        reformulate_gdp(_both_refuse_model(), method="big-m")
    with pytest.raises(HullPerspectiveOriginError):
        reformulate_gdp(_both_refuse_model(), method="hull")

    lowered = reformulate_gdp(_both_refuse_model(), method="simplex")
    assert lowered._simplex_lowerings[0].sizes.cnf_clauses == 1

    result = _both_refuse_model().solve(time_limit=60, gdp_method="simplex")
    assert result.status == "optimal"
    assert result.objective == pytest.approx(0.0, abs=1e-6)
    source = _both_refuse_model()
    assert _residuals(source, result.x).max_violation <= 1e-6
    # At x = 5 BOTH disjuncts hold (1/5 <= 1 and 5 >= 3) — the overlap case a
    # recovered Boolean assignment would have to pick one of, wrongly.
    assert selected_disjuncts(source, result.x) == {"recip": [0, 1]}


@pytest.mark.unit
def test_the_lowered_model_carries_its_size_accounting():
    """``auto`` never routes to simplex, and every method leaves the list present."""
    for method in ("big-m", "hull", "auto"):
        lowered = reformulate_gdp(_overlap_model()[0], method=method)
        assert lowered._simplex_lowerings == []
    lowered = reformulate_gdp(_overlap_model()[0], method="simplex")
    assert len(lowered._simplex_lowerings) == 1
    assert lowered._simplex_lowerings[0].sizes.disjunctions == 1
    assert not math.isnan(float(lowered._simplex_lowerings[0].sizes.cnf_clauses))


@pytest.mark.unit
def test_the_helpers_are_reachable_without_importing_a_private_module():
    """``docs/disjunction_semantics.md`` §7 points users at these two names."""
    import discopt

    assert discopt.disjunction_residuals is disjunction_residuals
    assert discopt.selected_disjuncts is selected_disjuncts


@pytest.mark.relaxation
def test_the_lifted_row_is_exact_in_projection_over_a_sampled_box():
    """Theorem 1's exactness, asserted as the identity it actually is.

    For a fixed ``z`` the best simplex weight makes the weighted row equal
    ``min_j p_j(z)`` — a linear program over the simplex, whose optimum is at a
    vertex. So "some ``lambda`` in the simplex satisfies the clause row" holds
    exactly when the source disjunction holds, with no gap in either direction:
    no feasible point is cut, and no infeasible point is admitted.

    Sampled over both fixtures rather than argued, and the executed comparison
    count is asserted so a sampler that generated nothing cannot read as a pass.
    """
    fixtures = []

    m1, _ = _overlap_model()
    fixtures.append((m1, "x", np.linspace(-2.0, 3.0, 101)))

    m2 = dm.Model("disjoint")
    y = m2.continuous("y", lb=-5.0, ub=5.0)
    m2.minimize(y)
    m2.either_or([[y <= -1.0], [y >= 1.0]], name="gap")
    fixtures.append((m2, "y", np.linspace(-5.0, 5.0, 101)))

    checked = 0
    for model, name, grid in fixtures:
        for value in grid:
            report = _residuals(model, {name: np.array([value])})
            (res,) = report.residuals
            predicates = np.asarray(res.per_disjunct, dtype=float)
            # The simplex minimum of a linear function is attained at a vertex.
            best_weighted = float(predicates.min())
            assert best_weighted == pytest.approx(res.violation, abs=1e-12)
            # Both directions of "exact in projection", stated separately.
            holds = bool((predicates <= 1e-12).any())
            assert holds == (best_weighted <= 1e-12)
            checked += 1

    assert checked == 202, f"the sampler covered {checked} points, expected 202"


@pytest.mark.unit
def test_jacobian_nonzeros_is_the_fourth_size_quantity_and_refuses_a_gdp_row():
    """Requirement 4's fourth quantity, measured on the lowered model.

    It is a whole-model number, so it lives beside ``LoweringSizes`` rather than
    on it, and it refuses a model still carrying a disjunction — counting that
    row as zero would understate the pattern it exists to compare.
    """
    from discopt._relax.simplex_lowering import structural_jacobian_nonzeros

    def build():
        m = dm.Model("nnz")
        x = m.continuous("x", lb=-5.0, ub=5.0)
        m.minimize(x * x)
        m.either_or([[x <= -1.0], [x >= 1.0]], name="g")
        return m

    counts = {
        method: structural_jacobian_nonzeros(reformulate_gdp(build(), method=method))
        for method in ("big-m", "hull", "simplex")
    }
    # Three different sparsity patterns for the same disjunction — which is the
    # whole point of measuring this separately from clause and row counts.
    assert len(set(counts.values())) == 3
    assert counts["simplex"] < counts["big-m"] < counts["hull"]

    with pytest.raises(ValueError, match="unlowered disjunction"):
        structural_jacobian_nonzeros(build())


@pytest.mark.unit
def test_a_weight_name_cannot_collide_with_a_user_variable():
    """The prefix lives inside the reserved ``GDP_AUX_PREFIX`` namespace.

    ``docs/disjunction_semantics.md`` §4 makes that namespace the enforced
    boundary between user Boolean identities and existential compiler
    auxiliaries. A weight is the latter, so it belongs inside it — and
    ``Model._check_name`` then keeps a user variable out of the namespace
    rather than leaving the guarantee to a naming convention.
    """
    from discopt.modeling.core import GDP_AUX_PREFIX

    assert SIMPLEX_WEIGHT_PREFIX.startswith(GDP_AUX_PREFIX)
    m, _x = _overlap_model()
    with pytest.raises(ValueError, match="reserved"):
        m.continuous(f"{SIMPLEX_WEIGHT_PREFIX}0", lb=0.0, ub=1.0)

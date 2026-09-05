"""#1148 — source complementarity residuals and the local-vs-certified contract.

Slice 2 of the MCP/MPEC RFC (#1123), on top of #1147's durable source
provenance. Each test below pins one acceptance criterion of the issue:

* ``solve_mpec`` returns **one type** for ``scholtes``, ``sos1`` and ``gdp``;
* a Scholtes solve reports a source complementarity residual computed on the
  **source operands**, and it differs from the lowered row's residual on a model
  where the two genuinely differ;
* a local solve returns the distinct status, the harness maps it to something
  both ``incorrect_count`` and ``proved_optimal_count`` skip, and
  ``gap_certified`` is False;
* a local result is **never** usable as a dual bound;
* a stalled/failed continuation does not report ``infeasible``;
* every residual reported carries its definition.

Measurement discipline (CLAUDE.md §6): this module counts every residual
comparison it actually executes and :func:`test_zz_probe_actually_fired` fails
when that count is zero. A residual probe that traverses nothing prints
"0 violations" and reads as a pass — the exact failure this slice exists to
prevent, so the probe checks itself.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import pathlib  # noqa: E402

import discopt.modeling.core as dm  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from discopt import status as dstatus  # noqa: E402
from discopt.mpec import (  # noqa: E402
    box_mcp,
    complementarity,
    generated_rows,
    solve_mpec,
    source_residuals,
)
from discopt.mpec_report import (  # noqa: E402
    ComplementarityKind,
    ContinuationStage,
    ContinuationTrace,
    Residual,
    accept_local_incumbent,
    admitted_residual_scale,
    evaluate_at_point,
    max_source_complementarity,
    point_from_flat,
    relation_residuals,
    source_residual_report,
)

pytest.importorskip("pounce")

pytestmark = pytest.mark.smoke

#: Every residual comparison this module actually executed. Incremented at the
#: point of comparison, never at the top of a loop that might not run.
ASSERTIONS_EXECUTED = 0

#: Path to the module under test, for the source-level contract checks below.
discopt_mpec_file = __import__("discopt.mpec", fromlist=["x"]).__file__


def _checked(condition: bool, message: str) -> None:
    """Assert, and count the assertion as executed."""
    global ASSERTIONS_EXECUTED
    ASSERTIONS_EXECUTED += 1
    assert condition, message


# ─────────────────────────────── fixtures ───────────────────────────────


def _distance_model():
    """``min (x-1)^2 + (y-1)^2  s.t.  0 <= x ⊥ y >= 0`` — optimum 1 at (1, 0)."""
    m = dm.Model("mpec_distance")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize((x - 1) ** 2 + (y - 1) ** 2)
    return m, x, y


def _balanced_model():
    """The model on which the source and lowered residuals genuinely differ.

    ``max x + y`` under ``x == y`` and ``0 <= x ⊥ y >= 0``. The true MPEC forces
    ``x = y = 0``; the Scholtes relaxation only asks ``x·y <= t``, so its local
    optimum sits at ``x = y ≈ sqrt(t)``. There, **every generated row is
    satisfied** while the source complementarity ``min(x, y)`` is ``sqrt(t)`` —
    four orders of magnitude apart at ``t = 1e-8``. A probe that could not tell
    the two apart would report the lowered ~1e-8 and be believed.
    """
    m = dm.Model("mpec_balanced")
    x = m.continuous("x", lb=0, ub=1)
    y = m.continuous("y", lb=0, ub=1)
    m.subject_to(x == y, name="sym")
    m.minimize(-(x + y))
    return m, x, y


# ──────────────── D. one return type for all three methods ────────────────


def test_solve_mpec_returns_one_type_for_every_method():
    results = {}
    for method in ("scholtes", "sos1", "gdp"):
        m, x, y = _distance_model()
        results[method] = solve_mpec(m, [complementarity(x, y, name="c")], method=method)

    types = {method: type(res) for method, res in results.items()}
    _checked(
        len(set(types.values())) == 1,
        f"solve_mpec must return one type for every method, got {types}",
    )
    _checked(
        next(iter(types.values())) is dm.SolveResult,
        f"the common type must be modeling.core.SolveResult, got {types}",
    )
    for method, res in results.items():
        _checked(
            isinstance(res.status, str),
            f"{method}: status must be a plain string, got {type(res.status).__name__}",
        )
        _checked(
            hasattr(res, "gap_certified"),
            f"{method}: every arm must carry gap_certified",
        )
        _checked(
            res.mpec_report is not None,
            f"{method}: every arm must attach a source-residual report",
        )
        _checked(
            abs(float(res.objective) - 1.0) < 1e-3,
            f"{method}: objective {res.objective} is not the MPEC optimum 1.0",
        )


# ──────── A. the source residual, and that it is NOT the lowered one ────────


def test_scholtes_source_residual_differs_from_the_lowered_row_residual():
    """The acceptance probe: on a model where they genuinely differ, they differ.

    The lowered rows are satisfied to NLP tolerance while the *source* condition
    is violated by ~sqrt(t). If this test ever reports the two as equal, the
    residual is being read off the lowered rows and the whole slice is a no-op.
    """
    m, x, y = _balanced_model()
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes", t_min=1e-8)
    report = res.mpec_report

    _checked(report is not None, "a Scholtes solve must attach a source-residual report")
    _checked(
        report.n_scalar_relations == 1,
        f"expected 1 measured scalar relation, got {report.n_scalar_relations}",
    )
    src = float(report.complementarity.value)
    low = float(report.lowered_row_residual.value)
    _checked(
        src > 1e-5,
        f"the source complementarity residual should be near sqrt(t_min)=1e-4, got {src:.3e}",
    )
    _checked(
        low < 1e-6,
        f"the generated rows should be satisfied at the returned point, got {low:.3e}",
    )
    _checked(
        src > 100.0 * max(low, 1e-30),
        f"source ({src:.3e}) and lowered ({low:.3e}) residuals must differ by orders of "
        "magnitude on this model; a probe that cannot tell them apart is the failure "
        "this slice exists to prevent",
    )
    # And the point really is the one described: x = y ~ sqrt(t).
    xv = float(np.asarray(res.x["x"]))
    yv = float(np.asarray(res.x["y"]))
    _checked(
        abs(xv - yv) < 1e-6 and xv > 1e-5,
        f"expected a balanced point x = y ~ sqrt(t), got x={xv:.3e} y={yv:.3e}",
    )


def test_source_residual_reads_the_source_operands_not_a_rebuilt_row():
    """A hand-placed point: the residual must be ``min(f(x), g(x))`` of the SOURCE tree.

    The operands here are nonlinear expressions, which a GDP lowering lifts into
    auxiliary variables. Evaluating the *relation* must walk the declared
    expressions, so the number is predictable from the model alone.
    """
    m = dm.Model("nonlinear_operands")
    a = m.continuous("a", lb=0.0, ub=4.0)
    b = m.continuous("b", lb=0.0, ub=4.0)
    m.minimize(a + b)
    pair = complementarity(a * a, dm.exp(b) - 1.0, name="nl")
    m._complementarities.append(pair)

    point = point_from_flat(m, np.array([1.5, 0.25]))
    rows = relation_residuals(m, [pair], point)
    _checked(len(rows) == 1, f"expected one scalar relation, got {len(rows)}")
    row = rows[0]
    _checked(
        abs(row.f_value - 2.25) < 1e-9,
        f"f = a^2 at a=1.5 must be 2.25, got {row.f_value!r}",
    )
    _checked(
        abs(row.g_value - (np.exp(0.25) - 1.0)) < 1e-9,
        f"g = exp(b)-1 at b=0.25 must be {np.exp(0.25) - 1.0!r}, got {row.g_value!r}",
    )
    _checked(
        abs(row.complementarity.value - (np.exp(0.25) - 1.0)) < 1e-9,
        "min(f, g) must be the smaller operand",
    )


def test_nonnegativity_and_box_residuals_are_reported_per_operand():
    m = dm.Model("bounds")
    u = m.continuous("u", lb=-5.0, ub=5.0)
    v = m.continuous("v", lb=-5.0, ub=5.0)
    m.minimize(u + v)
    pair = complementarity(u, v, name="uv")
    point = point_from_flat(m, np.array([-0.75, 2.0]))
    row = relation_residuals(m, [pair], point)[0]
    _checked(
        abs(row.f_bound.value - 0.75) < 1e-12,
        f"f = -0.75 violates 0 <= f by 0.75, got {row.f_bound.value!r}",
    )
    _checked(row.g_bound.value == 0.0, f"g = 2.0 satisfies 0 <= g, got {row.g_bound.value!r}")
    _checked(
        "lb_f" in row.f_bound.definition and "lb_g" in row.g_bound.definition,
        "each bound residual must record its own definition",
    )


def test_box_mcp_uses_the_normal_map_not_min():
    """A box relation's residual is the MCP normal map, chosen from declared bounds."""
    m = dm.Model("boxed")
    z = m.continuous("z", lb=-1.0, ub=1.0)
    m.minimize(z)
    pair = box_mcp(z + 1.0, z, name="bz")  # F(z) = z + 1 on z in [-1, 1]
    m._complementarities.append(pair)

    # z = 1 (the upper bound) requires F <= 0; F = 2, so the relation is violated.
    at_ub = relation_residuals(m, [pair], point_from_flat(m, np.array([1.0])))[0]
    _checked(
        at_ub.complementarity.definition
        == ComplementarityKind.formula(ComplementarityKind.NATURAL_MAP),
        f"a box relation must use the normal map, got {at_ub.complementarity.definition!r}",
    )
    _checked(
        at_ub.complementarity.value > 1.0,
        f"z=u with F=+2 violates the box MCP; got residual {at_ub.complementarity.value!r}",
    )
    # z = -1 (the lower bound) requires F >= 0; F = 0, which holds.
    at_lb = relation_residuals(m, [pair], point_from_flat(m, np.array([-1.0])))[0]
    _checked(
        at_lb.complementarity.value < 1e-12,
        f"z=l with F=0 satisfies the box MCP; got {at_lb.complementarity.value!r}",
    )


def test_vector_relation_residuals_stay_attributable_to_the_declared_relation():
    m = dm.Model("vec")
    p = m.continuous("p", shape=2, lb=0, ub=10)
    q = m.continuous("q", shape=2, lb=0, ub=10)
    m.minimize(p[0] + q[1])
    pair = complementarity(p, q, name="pq")
    m._complementarities.append(pair)

    point = point_from_flat(m, np.array([0.0, 3.0, 5.0, 0.0]))
    rows = relation_residuals(m, [pair], point)
    _checked(len(rows) == 2, f"a 2-vector relation must expand to 2 scalar rows, got {len(rows)}")
    _checked(
        all(r.source_name == "pq" for r in rows),
        f"every element must name its declared relation, got {[r.source_name for r in rows]}",
    )
    _checked(
        [r.index for r in rows] == [(0,), (1,)],
        f"every element must record its index, got {[r.index for r in rows]}",
    )
    _checked(
        max(r.complementarity.value for r in rows) < 1e-12,
        "p=(0,3), q=(5,0) is elementwise complementary",
    )


def test_scale_makes_the_tolerance_meaningful():
    """A declared scale divides the residual; the raw value is kept beside it."""
    m = dm.Model("scaled")
    f = m.continuous("f", lb=0, ub=1e4)
    g = m.continuous("g", lb=0, ub=1e4)
    m.minimize(f + g)
    pair = complementarity(f, g, name="fg", scale=1e3)
    point = point_from_flat(m, np.array([2.0, 1e3]))
    row = relation_residuals(m, [pair], point)[0]
    _checked(row.complementarity.value == 2.0, f"min(2, 1e3) = 2, got {row.complementarity.value}")
    _checked(
        abs(row.complementarity.scaled_value - 2e-3) < 1e-12,
        f"the scaled residual must be 2/1e3, got {row.complementarity.scaled_value!r}",
    )


def test_every_reported_residual_records_its_definition():
    m, x, y = _distance_model()
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes")
    report = res.mpec_report
    _checked(len(report.residuals) >= 3, "the report must carry at least three residuals")
    for residual in report.residuals:
        _checked(
            isinstance(residual, Residual) and bool(residual.definition.strip()),
            f"residual {residual.name!r} has no recorded definition",
        )
    for row in report.relations:
        for residual in (row.complementarity, row.f_bound, row.g_bound):
            _checked(
                bool(residual.definition.strip()),
                f"per-relation residual {residual.name!r} has no recorded definition",
            )
    _checked(
        set(report.definitions) == {r.name for r in report.residuals},
        "report.definitions must cover exactly the reported residuals",
    )


def test_continuation_trace_is_kept_and_carries_the_admitted_residual_scale():
    m, x, y = _distance_model()
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes", t_min=1e-8)
    trace = res.mpec_report.continuation
    _checked(trace is not None, "a Scholtes solve must report its continuation")
    _checked(len(trace.stages) > 1, f"expected several homotopy stages, got {len(trace.stages)}")
    _checked(
        all(isinstance(s, ContinuationStage) for s in trace.stages),
        "stages must be ContinuationStage records",
    )
    _checked(
        trace.termination_reason == "t_min_reached" and trace.converged,
        f"expected the schedule to reach t_min, got {trace.termination_reason!r}",
    )
    _checked(
        abs(float(trace.final_t) - 1e-8) < 1e-18,
        f"final t must be t_min, got {trace.final_t!r}",
    )
    _checked(
        abs(float(trace.admitted_residual_scale) - 1e-4) < 1e-12,
        f"the min-form admitted scale sqrt(t) at t=1e-8 is 1e-4, got "
        f"{trace.admitted_residual_scale!r}",
    )
    _checked(
        any(s.source_complementarity is not None for s in trace.stages),
        "at least one stage must record its achieved SOURCE residual",
    )
    # The achieved residual is reported independently of the final t.
    _checked(
        res.mpec_report.complementarity.value != trace.final_t,
        "the achieved source residual must not be the homotopy parameter",
    )


def test_admitted_residual_scale_is_per_definition_and_absent_without_a_t():
    """Nonblocking 1: the scale is an admitted MAXIMUM, and it differs by formula.

    ``sqrt(t)`` was reported for every residual definition, including the product
    and Fischer-Burmeister forms whose formulas admit different worst cases. At
    ``t = 1e-8`` that compared a product residual against ``1e-4`` instead of
    ``1e-8`` — four orders of magnitude of slack handed to the definition that
    needs it least (#1158 review 3, nonblocking 1).
    """
    from discopt.mpec_report import (  # noqa: PLC0415
        ComplementarityKind,
        admitted_residual_scale_definition,
    )

    _checked(
        abs(admitted_residual_scale(1e-8, ComplementarityKind.MIN) - 1e-4) < 1e-15,
        "min: sqrt(1e-8) = 1e-4",
    )
    _checked(
        admitted_residual_scale(1e-8, ComplementarityKind.PRODUCT) == 1e-8,
        "product: the row bounds the product itself, so the scale IS t",
    )
    _checked(
        abs(
            admitted_residual_scale(1e-8, ComplementarityKind.FISCHER_BURMEISTER)
            - 5.857864376269049e-05
        )
        < 1e-18,
        "fischer_burmeister: (2 - sqrt(2))*sqrt(t)",
    )
    _checked(
        admitted_residual_scale(1e-8, ComplementarityKind.NATURAL_MAP) == pytest.approx(1e-4),
        "natural_map reduces to min on the nonnegative pair",
    )
    _checked(
        admitted_residual_scale(1e-8, ComplementarityKind.AUTO) is None,
        "auto names no formula, so it derives no scale rather than borrowing one",
    )
    _checked(admitted_residual_scale(None) is None, "no t means no admitted scale")
    _checked(admitted_residual_scale(0.0) is None, "t=0 admits nothing to scale against")
    # The formula is recorded beside the number, per this module's own rule.
    _checked(
        "sqrt(t)" in admitted_residual_scale_definition(ComplementarityKind.MIN)
        and admitted_residual_scale_definition(ComplementarityKind.AUTO) is None,
        "each definition records its own formula",
    )


def test_admitted_scale_is_an_upper_bound_not_a_floor():
    """Nonblocking 1: a point far BELOW the admitted scale is ordinary, not suspect.

    ``(f, g) = (0, 1)`` satisfies ``f*g <= t`` with a residual of exactly ``0`` at
    every positive ``t``, so wording that called ``sqrt(t)`` a limit on attainable
    accuracy asserted something the relaxation does not say.
    """
    from discopt.mpec_report import ComplementarityKind, Residual  # noqa: PLC0415

    scale = admitted_residual_scale(1e-8, ComplementarityKind.MIN)
    exact = Residual(
        name="source_complementarity",
        value=0.0,
        definition=ComplementarityKind.formula(ComplementarityKind.MIN),
        admitted_scale=scale,
    )
    _checked(
        exact.within_admitted_scale,
        "an exactly complementary point is within what the relaxation admits",
    )
    _checked(
        float(exact.value) < float(scale),
        f"and it is strictly BELOW the admitted scale {scale!r} — the scale is a "
        "maximum over the relaxed set, not a bound the solver cannot beat",
    )
    outside = Residual(
        name="source_complementarity",
        value=10.0 * scale,
        definition=ComplementarityKind.formula(ComplementarityKind.MIN),
        admitted_scale=scale,
    )
    _checked(
        not outside.within_admitted_scale,
        "a residual the relaxation does NOT admit is reported as outside it",
    )
    _checked(
        outside.as_dict()["within_admitted_scale"] is False
        and "admitted_scale" in outside.as_dict(),
        "and both survive serialization under their own names",
    )


def test_selector_integrality_residual_is_reported_when_selectors_exist():
    m = dm.Model("selectors")
    w = m.continuous("w", lb=0, ub=1)
    s = m.binary("s")
    m.minimize(w)
    pair = complementarity(w, s, name="ws")
    m._complementarities.append(pair)
    report = source_residual_report(m, [pair], x_flat=np.array([0.5, 0.3]))
    _checked(report.integrality is not None, "a model with binaries must report integrality")
    _checked(
        abs(report.integrality.value - 0.21) < 1e-12,
        f"y(1-y) at y=0.3 is 0.21, got {report.integrality.value!r}",
    )
    _checked(
        "0 <= y_i <= 1" in report.integrality.definition,
        "the integrality definition must record the box check it performs first",
    )


def test_selector_outside_its_box_is_not_hidden_by_the_product():
    """``y(1-y)`` is negative outside [0, 1]; the report must not read that as better."""
    m = dm.Model("bad_selector")
    w = m.continuous("w", lb=0, ub=1)
    s = m.binary("s")
    m.minimize(w)
    pair = complementarity(w, s, name="ws")
    m._complementarities.append(pair)
    report = source_residual_report(m, [pair], x_flat=np.array([0.5, 2.0]))
    _checked(
        report.integrality.value == float("inf"),
        f"a selector at y=2 is out of its box; got {report.integrality.value!r}",
    )


# ──────────── B. the distinct terminal status, and what it may claim ────────────


def test_scholtes_returns_the_local_status_and_no_certification():
    m, x, y = _distance_model()
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes")
    _checked(
        res.status == dstatus.LOCAL_OPTIMAL,
        f"a Scholtes solve must report the local status, got {res.status!r}",
    )
    _checked(res.gap_certified is False, "a local result is never certified")
    _checked(res.bound is None, "a local result carries no dual bound")
    _checked(res.gap is None, "a local result carries no gap")
    _checked(
        dstatus.is_local_status(res.status) and not dstatus.is_certified_status(res.status),
        "the status vocabulary must classify it as local and not certifying",
    )


def test_global_methods_keep_their_certified_status():
    for method in ("gdp", "sos1"):
        m, x, y = _distance_model()
        res = solve_mpec(m, [complementarity(x, y, name="c")], method=method)
        _checked(
            res.status == "optimal" and res.gap_certified,
            f"{method} is a global method and must keep its certificate, got "
            f"{res.status!r}/{res.gap_certified}",
        )
        _checked(
            not dstatus.is_local_status(res.status),
            f"{method} must not report a local status",
        )


def test_local_result_can_never_carry_a_dual_bound():
    """§C, enforced structurally: the constructor refuses, it does not silently drop."""
    for status in sorted(dstatus.LOCAL_STATUSES):
        with pytest.raises(ValueError, match="may never contribute a dual bound"):
            dm.SolveResult(status=status, objective=1.0, bound=0.0)
        with pytest.raises(ValueError, match="may never contribute a dual bound"):
            dm.SolveResult(status=status, objective=1.0, root_bound=0.0)
        _checked(True, f"{status!r}: both bound and root_bound are guarded")
    # And an honest local result is accepted, decertified.
    ok = dm.SolveResult(status=dstatus.LOCAL_OPTIMAL, objective=1.0, gap_certified=True)
    _checked(
        ok.gap_certified is False and ok.bound is None,
        "a local result is decertified rather than left claiming a gap",
    )


def test_a_local_result_from_solve_mpec_is_not_used_as_a_bound():
    m, x, y = _balanced_model()
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes")
    # The local optimum here is BELOW the true MPEC optimum (0): if this value
    # were ever taken as a dual bound it would fathom the true optimum away.
    _checked(
        float(res.objective) < -1e-6,
        f"the local point must be a strictly better-than-true objective here, got "
        f"{res.objective!r}",
    )
    _checked(res.bound is None, "and it must not be reported as a dual bound")
    _checked(res.gap_certified is False, "nor certified")


def test_local_incumbent_needs_independent_verification():
    """A local point may seed an incumbent only after an independent check."""
    m, x, y = _distance_model()
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes")
    verified = accept_local_incumbent(m, res)
    _checked(
        verified is not None and abs(verified - 1.0) < 1e-3,
        f"the Scholtes point is feasible and must verify, got {verified!r}",
    )

    class _Bogus:
        x = {"x": np.array(9.0), "y": np.array(9.0)}  # violates x*y <= t grossly

    _checked(
        accept_local_incumbent(m, _Bogus()) is None,
        "a point that fails verification must not be accepted as an incumbent",
    )


def test_a_stalled_continuation_never_reports_infeasible():
    """A homotopy that converges nowhere reports ``local_infeasible``, not ``infeasible``."""
    m, x, y = _distance_model()
    # One iteration at a large t: the schedule cannot reach t_min, so the result
    # must not claim anything global either way.
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes", max_iter=1, t0=1.0)
    _checked(
        res.status != "infeasible",
        f"a truncated continuation must never report infeasible, got {res.status!r}",
    )
    _checked(
        dstatus.is_local_status(res.status),
        f"it must report a local status, got {res.status!r}",
    )
    trace = res.mpec_report.continuation
    _checked(
        trace.termination_reason != "t_min_reached" and not trace.converged,
        f"the trace must record that it did not converge, got {trace.termination_reason!r}",
    )


def test_local_infeasible_is_not_a_certified_status():
    r = dm.SolveResult(status=dstatus.LOCAL_INFEASIBLE)
    _checked(
        not dstatus.is_certified_status(r.status),
        "a failed local search certifies nothing",
    )
    _checked(
        dstatus.is_certified_status("infeasible"),
        "a certified infeasibility is still a certificate — the two must stay distinct",
    )
    _checked(r.gap_certified is False, "and it is never gap-certified")


# ──────────── the harness fails closed on the local status ────────────


def test_benchmark_harness_skips_a_local_result():
    """Both release-gate counters must skip a local row, and neither may score it."""
    import sys
    from pathlib import Path

    bench = Path(__file__).resolve().parents[2] / "discopt_benchmarks"
    if str(bench) not in sys.path:
        sys.path.insert(0, str(bench))
    from benchmarks.metrics import (  # noqa: PLC0415
        DISCOPT_STATUS_MAP,
        incorrect_count,
        proved_optimal_count,
    )
    from benchmarks.metrics import (
        SolveResult as BenchResult,
    )
    from benchmarks.metrics import (
        SolveStatus as BenchStatus,
    )

    for local in sorted(dstatus.LOCAL_STATUSES):
        mapped = DISCOPT_STATUS_MAP.get(local, BenchStatus.UNKNOWN)
        _checked(
            mapped is not BenchStatus.OPTIMAL,
            f"{local!r} must never map to OPTIMAL, got {mapped}",
        )
        row = BenchResult(instance="probe", solver="discopt", status=mapped, objective=-1e9)
        _checked(not row.is_solved, f"{local!r} must not count as solved")
        _checked(
            incorrect_count([row], {"probe": 1.0}) == 0,
            f"incorrect_count must skip a {local!r} row (it makes no optimality claim)",
        )
        _checked(
            proved_optimal_count([row]) == 0,
            f"proved_optimal_count must skip a {local!r} row",
        )
    # Control: the same row with OPTIMAL is counted by both, so the skip above is
    # a property of the status and not of an inert probe.
    import warnings  # noqa: PLC0415

    scored = BenchResult(
        instance="probe", solver="discopt", status=BenchStatus.OPTIMAL, objective=-1e9
    )
    with warnings.catch_warnings():
        # incorrect_count warns on the row it counts; that warning IS the signal
        # here, so it is expected rather than a defect to surface.
        warnings.simplefilter("ignore", UserWarning)
        counted_wrong = incorrect_count([scored], {"probe": 1.0})
    _checked(
        counted_wrong == 1 and proved_optimal_count([scored]) == 1,
        "control: an OPTIMAL row with a wrong objective IS counted — otherwise the "
        "skip above proves nothing",
    )


# ──────────── the instrument refuses to measure nothing ────────────


def test_report_refuses_when_it_measured_no_relation():
    m = dm.Model("empty")
    z = m.continuous("z", lb=0, ub=1)
    m.minimize(z)

    class _NoElements:
        """A relation that expands to nothing — the silent-no-op shape."""

        def elements(self, model):
            return []

    with pytest.raises(ValueError, match="none"):
        relation_residuals(m, [_NoElements()], point_from_flat(m, np.array([0.5])))
    _checked(True, "a relation set that measures nothing is refused, not reported as 0.0")


def test_point_evaluation_refuses_an_unsupported_atom():
    """An expression the interval evaluator cannot walk must raise, not read as 0."""
    m = dm.Model("opaque")
    z = m.continuous("z", lb=0.1, ub=1.0)
    m.minimize(z)
    opaque = dm.custom(lambda v: v * 2.0)(z)
    with pytest.raises(ValueError, match="did not stay degenerate|not finite"):
        evaluate_at_point(m, opaque, point_from_flat(m, np.array([0.5])))
    _checked(True, "an unevaluable operand raises rather than reporting a midpoint")


def test_lowered_rows_are_tracked_per_model_and_never_carried_blindly():
    m, x, y = _distance_model()
    pair = complementarity(x, y, name="c")
    from discopt.mpec import reformulate_gdp  # noqa: PLC0415

    reformulate_gdp(m, [pair])
    rows = generated_rows(m, [pair])
    _checked(rows is not None and len(rows) > 0, "a lowering must record the rows it emitted")
    _checked(
        pair.rows_in(m) is not None,
        "the rows are recorded against the model that carries them",
    )

    other = dm.Model("other")
    _checked(
        pair.rows_in(other) is None,
        "a model that never lowered the relation knows no rows for it",
    )
    _checked(
        generated_rows(other, [pair]) is None,
        "an untracked relation yields None, not an empty (and falsely complete) list",
    )


def test_source_residuals_helper_works_on_a_plain_model_solve():
    m, x, y = _distance_model()
    m.complementarity(x, y, name="c")
    res = m.solve()
    report = source_residuals(m, res)
    _checked(report is not None, "a solved model with relations must produce a report")
    _checked(report.n_scalar_relations == 1, "one relation was measured")
    _checked(
        any("CURRENT rows" in n for n in report.notes),
        "without a pre-lowering snapshot the report must say so rather than imply a "
        "clean source reading",
    )


def test_stationarity_is_never_claimed():
    """C-/M-/S-stationarity is not checked, so it is not reported (#1148 §C)."""
    m, x, y = _distance_model()
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes")
    _checked(
        res.mpec_report.stationarity is None,
        "discopt does not check stationarity conditions and must not claim one",
    )


def test_report_serializes_with_every_definition_intact():
    m, x, y = _distance_model()
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes")
    payload = res.mpec_report.as_dict()
    _checked(
        payload["n_scalar_relations"] == 1,
        "the serialized schema must carry the measurement count",
    )
    for key in ("complementarity", "bound_violation", "primal_feasibility"):
        _checked(
            bool(payload[key]["definition"]),
            f"the serialized {key} must carry its definition",
        )
    _checked(
        payload["continuation"]["stages"] and payload["continuation"]["admitted_residual_scale"],
        "the serialized continuation must carry its stages and floor",
    )
    import json  # noqa: PLC0415

    _checked(bool(json.dumps(payload)), "the schema must be JSON-serializable")


def test_max_source_complementarity_matches_the_full_report():
    m, x, y = _balanced_model()
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes")
    point = point_from_flat(
        m, np.concatenate([np.atleast_1d(np.asarray(res.x[v.name])).ravel() for v in m._variables])
    )
    quick = max_source_complementarity(m, list(m._complementarities), point)
    _checked(
        abs(quick - res.mpec_report.complementarity.value) < 1e-15,
        "the per-stage shortcut must agree with the full report at the same point",
    )


def test_continuation_trace_serializes_a_stall_distinguishably():
    stalled = ContinuationTrace(
        parameter="t",
        stages=(ContinuationStage(0, 1e-2, "iteration_limit", False, "did not converge"),),
        final_t=1e-2,
        termination_reason="subsolver_failure",
        admitted_residual_scale=admitted_residual_scale(1e-2),
    )
    converged = ContinuationTrace(
        parameter="t",
        stages=(
            ContinuationStage(0, 1e-8, "optimal", True, "subsolver converged", certified=True),
        ),
        final_t=1e-8,
        termination_reason="t_min_reached",
        admitted_residual_scale=admitted_residual_scale(1e-8),
        reported_point_certified=True,
    )
    _checked(
        not stalled.converged and converged.converged,
        "a stall and a converged schedule must be distinguishable from the trace alone",
    )
    _checked(
        stalled.as_dict()["converged"] is False,
        "and the distinction must survive serialization",
    )


def teardown_module(module):  # noqa: ARG001 - pytest hook signature
    """CLAUDE.md §6: fail when this module's residual comparisons never executed.

    A module-teardown hook rather than a test, deliberately. As a test it was
    position-dependent — it passed only if some counted test happened to run
    before it, and this suite runs under ``pytest-randomly``, so a shuffle that
    put it first would have made the guard-against-measuring-nothing itself
    measure nothing. ``teardown_module`` runs after every test in the file, in
    any order, and an assertion here fails the run.
    """
    print(f"\n#1148 residual probe: {ASSERTIONS_EXECUTED} comparisons executed")
    assert ASSERTIONS_EXECUTED > 0, (
        "the #1148 residual probes executed ZERO comparisons. A probe that traverses "
        "nothing prints a clean pass; this counter exists so that cannot happen."
    )


# ════════════════════════════════════════════════════════════════════════════
# Regressions for the #1158 code review. Each fails on the pre-review commit.
# ════════════════════════════════════════════════════════════════════════════


def test_review_1_sum_row_residual_is_reduced():
    """HIGH 1: ``evaluate_interval`` did not reduce ``SumExpression``.

    ``sum(x)`` came back as the operand's ELEMENTWISE enclosure, so the degenerate
    -width and finiteness guards never fired and a row violated by 2.0 reported a
    residual of 0.0 — the instrument reading clean on an infeasible point, for one
    of the most common expression shapes there is.
    """
    from discopt._relax.convexity.interval_eval import evaluate_interval  # noqa: PLC0415

    m = dm.Model("sum_row")
    x = m.continuous("x", shape=2, lb=0, ub=10)
    m.minimize(x[0])
    m.subject_to(dm.sum(x) <= 10, name="budget")

    # The enclosure over the whole box must CONTAIN the value; [0, 10] did not.
    box = evaluate_interval(dm.sum(x), m)
    # CONTAINMENT, not equality: the reduction rounds outward (see the review-2
    # regression below), so the endpoint is 20 or a hair beyond it. What must never
    # happen again is an endpoint BELOW 20 — an enclosure that excludes the value.
    _checked(
        float(np.asarray(box.hi)) >= 20.0,
        f"sum(x) over x in [0,10]^2 must enclose 20, got hi={box.hi!r}",
    )
    _checked(
        float(np.asarray(box.lo)) <= 0.0,
        f"and must enclose 0, got lo={box.lo!r}",
    )

    point = point_from_flat(m, np.array([6.0, 6.0]))
    _checked(
        abs(float(evaluate_at_point(m, dm.sum(x), point)[0]) - 12.0) < 1e-12,
        "sum(x) at (6, 6) is 12",
    )
    from discopt.mpec_report import _model_rows, _primal_feasibility  # noqa: PLC0415

    viol = _primal_feasibility(m, _model_rows(m), None, point).value
    _checked(
        abs(viol - 2.0) < 1e-12,
        f"sum(x) <= 10 at (6,6) is violated by 2.0; the residual reported {viol!r}",
    )

    # And an axis reduction keeps its axis.
    a = m.continuous("a", shape=(2, 3), lb=0, ub=1)
    rowwise = evaluate_interval(dm.sum(a, axis=1), m)
    _checked(
        np.asarray(rowwise.hi).shape == (2,) and float(np.max(rowwise.hi)) >= 3.0,
        f"sum(a, axis=1) reduces to shape (2,) enclosing 3, got {rowwise.hi!r}",
    )


def test_review_1_scalar_sum_relation_is_measurable():
    """HIGH 1, second half: ``sum(x) ⊥ y`` is a scalar relation and must measure."""
    m = dm.Model("sum_relation")
    x = m.continuous("x", shape=2, lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(y)
    pair = complementarity(dm.sum(x), y, name="sy")
    row = relation_residuals(m, [pair], point_from_flat(m, np.array([1.0, 2.0, 4.0])))[0]
    _checked(abs(row.f_value - 3.0) < 1e-12, f"f = sum(x) at (1,2) is 3, got {row.f_value!r}")
    _checked(
        abs(row.complementarity.value - 3.0) < 1e-12,
        f"min(3, 4) is 3, got {row.complementarity.value!r}",
    )


def test_review_2_local_incumbent_is_gated_on_the_source_residual():
    """HIGH 2: verification ran against the LOWERED model, which is a relaxation.

    On the Scholtes arm the model holds ``f·g <= t``, not ``f·g == 0``, so
    ``verify_point`` alone vouched for a point whose source complementarity was
    1.4e-4 — and returned an objective strictly better than the true global
    optimum, which as a cutoff fathoms the optimum away.
    """
    m, x, y = _balanced_model()
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes")
    report = res.mpec_report
    _checked(
        not report.source_satisfied,
        f"the balanced point violates the source relation ({report.complementarity.value:.3e})",
    )
    _checked(
        float(res.objective) < 0.0,
        f"and its objective {res.objective!r} beats the true optimum 0 — the danger",
    )
    _checked(
        accept_local_incumbent(m, res) is None,
        "so it must NOT be vouched for as an incumbent",
    )
    # Control: a genuinely complementary local point is still accepted, so the
    # refusal above is a property of the residual and not a blanket rejection.
    m2, a, b = _distance_model()
    ok = solve_mpec(m2, [complementarity(a, b, name="c")], method="scholtes")
    _checked(ok.mpec_report.source_satisfied, "control: the distance point IS complementary")
    verified = accept_local_incumbent(m2, ok)
    _checked(
        verified is not None and abs(verified - 1.0) < 1e-3,
        f"control: it must still be accepted, got {verified!r}",
    )


def test_review_3_a_failed_report_does_not_destroy_the_solve():
    """HIGH 3: an operand the interval evaluator cannot walk killed a done solve.

    The refusal is right for the *residual* — but the row need not have anything
    to do with the relations (source primal feasibility walks every source row),
    and ``Model.solve`` had already returned a certified result.
    """
    import warnings  # noqa: PLC0415

    m = dm.Model("unwalkable")
    u = m.continuous("u", lb=1, ub=3)
    v = m.continuous("v", lb=1, ub=3)
    x = m.continuous("x", lb=0, ub=5)
    y = m.continuous("y", lb=0, ub=5)
    m.subject_to(u**v <= 8, name="pow")  # variable exponent: not interval-walkable
    m.minimize((x - 1) ** 2 + (y - 1) ** 2 + u + v)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = solve_mpec(m, [complementarity(x, y, name="c")], method="gdp", time_limit=20)

    _checked(res.status == "optimal", f"the solve must survive, got status {res.status!r}")
    _checked(res.objective is not None, "and keep its objective")
    _checked(res.mpec_report is None, "the report is None — NOT MEASURED, not 'clean'")
    _checked(
        any(
            issubclass(w.category, RuntimeWarning) and "could not be measured" in str(w.message)
            for w in caught
        ),
        "and the refusal is loud: a warning naming the failure (CLAUDE.md §7)",
    )


def test_review_4_a_scholtes_relaxation_is_refused_by_the_solve_guard():
    """MEDIUM 4: a relaxation marked 'lowered' let the guard vouch for it.

    ``require_all_relations_lowered`` exists to catch "a declared relation solved
    as if absent". Scholtes leaves ``f·g <= t`` on the model, which is weaker than
    the relation, so a following ``Model.solve`` certified the RELAXATION —
    measured as ``optimal``/``gap_certified=True`` at an objective better than the
    true optimum.
    """
    from discopt.mpec import RELAXING_METHODS, relaxed_relations  # noqa: PLC0415

    m, x, y = _balanced_model()
    pair = complementarity(x, y, name="c")
    solve_mpec(m, [pair], method="scholtes")

    _checked("scholtes" in RELAXING_METHODS, "scholtes is a relaxation, not an encoding")
    _checked(relaxed_relations(m) == [pair], "the model carries it as relaxed")
    with pytest.raises(NotImplementedError, match="RELAXATION"):
        m.solve(time_limit=20)
    _checked(True, "and a global solve over it is refused rather than certified")

    # Control: an EXACT lowering is not refused.
    m2, a, b = _distance_model()
    solve_mpec(m2, [complementarity(a, b, name="c")], method="gdp")
    _checked(relaxed_relations(m2) == [], "gdp is exact, so nothing is flagged")
    again = m2.solve(time_limit=20)
    _checked(
        again.status == "optimal",
        f"control: an exactly-lowered model still solves, got {again.status!r}",
    )


def test_review_5_binary_roundoff_is_not_reported_as_out_of_box():
    """MEDIUM 5: ``> 0.0`` on solver output made -1e-15 an infinite residual."""
    m = dm.Model("roundoff")
    w = m.continuous("w", lb=0, ub=1)
    s = m.binary("s")
    m.minimize(w)
    pair = complementarity(w, s, name="ws")
    m._complementarities.append(pair)

    report = source_residual_report(m, [pair], x_flat=np.array([0.5, -1e-15]))
    _checked(
        np.isfinite(report.integrality.value),
        f"routine roundoff must not read as out-of-box, got {report.integrality.value!r}",
    )
    _checked(
        report.integrality.value < 1e-12,
        f"y = -1e-15 is integral to tolerance, got {report.integrality.value!r}",
    )
    # Control: a genuinely out-of-box selector is still caught.
    bad = source_residual_report(m, [pair], x_flat=np.array([0.5, 2.0]))
    _checked(bad.integrality.value == float("inf"), "control: y = 2 is still refused")


def test_review_6_rows_are_recorded_per_relation_not_per_call():
    """MEDIUM 6: every pair got the aggregate row list, so rows_in lied and
    generated_rows went quadratic (3N² rows for N scalar relations)."""
    from discopt.mpec import reformulate_gdp  # noqa: PLC0415

    m = dm.Model("per_pair")
    x = m.continuous("x", shape=2, lb=0, ub=10)
    y = m.continuous("y", shape=2, lb=0, ub=10)
    m.minimize(x[0] + y[1])
    p1 = complementarity(x, y, name="a")
    p2 = complementarity(x[0], y[1], name="b")
    reformulate_gdp(m, [p1, p2])

    r1 = [c.name for c in p1.rows_in(m)]
    r2 = [c.name for c in p2.rows_in(m)]
    _checked(
        all(n.startswith("a") for n in r1),
        f"relation 'a' must own only its own rows, got {r1}",
    )
    _checked(
        all(n.startswith("b") for n in r2),
        f"relation 'b' must own only its own rows, got {r2}",
    )
    _checked(
        set(r1).isdisjoint(r2),
        "and the two relations' row sets must not overlap",
    )
    total = generated_rows(m, [p1, p2])
    _checked(
        len(total) == len(m._constraints),
        f"generated_rows must not double-count: {len(total)} vs {len(m._constraints)} rows",
    )


def test_review_8_stage_objectives_are_in_model_units_on_a_maximize():
    """LOW 8: stage objectives were raw subsolver values, sign-flipped vs the result."""
    m = dm.Model("maximize_mpec")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.maximize(-((x - 1) ** 2) - (y - 1) ** 2)
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes")
    stages = [s for s in res.mpec_report.continuation.stages if s.accepted]
    _checked(bool(stages), "at least one stage converged")
    _checked(
        abs(float(stages[-1].objective) - float(res.objective)) < 1e-6,
        f"the last accepted stage ({stages[-1].objective!r}) must agree with the "
        f"result ({res.objective!r}) — same units, same sign",
    )
    _checked(
        float(res.objective) < 0.0,
        f"and the maximize objective is reported in model units, got {res.objective!r}",
    )


def test_review_9_final_t_is_a_t_that_was_actually_solved():
    """LOW 9: final_t was one schedule step past the last solve on a max_iter exit."""
    m, x, y = _distance_model()
    res = solve_mpec(
        m, [complementarity(x, y, name="c")], method="scholtes", t0=1.0, sigma=0.1, max_iter=3
    )
    trace = res.mpec_report.continuation
    solved_ts = [s.t for s in trace.stages]
    _checked(
        float(trace.final_t) in [float(t) for t in solved_ts],
        f"final_t={trace.final_t!r} must be one of the t values solved {solved_ts}",
    )
    _checked(
        trace.termination_reason == "max_iter" and not trace.converged,
        f"a truncated schedule is not converged, got {trace.termination_reason!r}",
    )


def test_review_9_converged_requires_an_accepted_stage():
    """LOW 9b: reaching t_min with every stage failing is not convergence."""
    stalled = ContinuationTrace(
        parameter="t",
        stages=(ContinuationStage(0, 1e-8, "error", False, "no stage converged"),),
        final_t=1e-8,
        termination_reason="t_min_reached",
        admitted_residual_scale=admitted_residual_scale(1e-8),
        any_stage_accepted=False,
    )
    _checked(
        not stalled.converged,
        "reaching the end of the schedule with nothing accepted is not converged",
    )
    _checked(
        stalled.as_dict()["any_stage_accepted"] is False,
        "and the distinction is serialized",
    )


# ════════════════════════════════════════════════════════════════════════════
# Regressions for the SECOND #1158 review pass. Three are inside the fixes above.
# ════════════════════════════════════════════════════════════════════════════


def test_review2_1_a_violated_relation_is_not_hidden_by_another_relations_scale():
    """HIGH 1: the aggregate mixed one relation's value with its own scale.

    ``effective_scale`` is per-relation, so ranking by RAW residual and then
    reporting that relation's scale said nothing about the others. A box MCP on
    ``z in [0, 1e3]`` violated by 1e-3 (scaled 1e-6, inside tolerance) outranked
    an NCP pair violated by 1e-4 (scaled 1e-4, a hundred times over) — and the
    report read ``source_satisfied=True`` with the second relation badly violated.
    Since ``accept_local_incumbent`` gates on exactly this, it was a hole in the
    HIGH-2 fix with the same shape as the original bug.
    """
    m = dm.Model("mixed_scales")
    zf = m.continuous("zf", lb=0, ub=1e3)
    zg = m.continuous("zg", lb=0, ub=1e3)
    nf = m.continuous("nf", lb=0, ub=10)
    ng = m.continuous("ng", lb=0, ub=10)
    m.minimize(zf + nf)

    big = complementarity(zf, zg, name="big", scale=1e3)
    small = complementarity(nf, ng, name="small")  # effective_scale 1.0
    for rel in (big, small):
        m._complementarities.append(rel)

    # big: min(1e-3, 5) = 1e-3 -> scaled 1e-6 (at tolerance)
    # small: min(1e-4, 5) = 1e-4 -> scaled 1e-4 (100x over tolerance)
    report = source_residual_report(m, [big, small], x_flat=np.array([1e-3, 5.0, 1e-4, 5.0]))

    by_name = {r.name: r for r in report.relations}
    _checked(
        abs(by_name["big"].complementarity.scaled_value - 1e-6) < 1e-12,
        f"big scales to 1e-6, got {by_name['big'].complementarity.scaled_value!r}",
    )
    _checked(
        abs(by_name["small"].complementarity.scaled_value - 1e-4) < 1e-12,
        f"small scales to 1e-4, got {by_name['small'].complementarity.scaled_value!r}",
    )
    _checked(
        report.complementarity.value == pytest.approx(1e-4),
        f"the aggregate must rank by SCALED value, so it reports 'small' "
        f"({1e-4}), got {report.complementarity.value!r} from {report.complementarity.where!r}",
    )
    _checked(
        not report.source_satisfied,
        "and source_satisfied must be False — one relation is 100x over tolerance",
    )
    # Control: with both relations inside tolerance it still reads satisfied, so
    # the False above is a property of the violation and not a broken predicate.
    ok = source_residual_report(m, [big, small], x_flat=np.array([1e-3, 5.0, 1e-9, 5.0]))
    _checked(ok.source_satisfied, "control: both within tolerance still reads satisfied")


def test_review2_2_the_sum_reduction_rounds_outward():
    """MEDIUM 2: the new reduction dropped the module's outward-rounding invariant.

    ``evaluate_interval`` is on the solve path (nonlinear bound tightening,
    uniform/OA relaxation, the g-convex injection), where an enclosure narrower
    than the true image is the FBBT failure that cuts the optimum out of the box.
    Summing n terms accumulates ~n ULP, so the endpoints must be pushed outward —
    as the sibling reduction ``_eval_matmul`` already did.
    """
    from discopt._relax.convexity.interval import Interval  # noqa: PLC0415
    from discopt._relax.convexity.interval_eval import evaluate_interval  # noqa: PLC0415

    m = dm.Model("rounding")
    x = m.continuous("x", shape=3, lb=0.1, ub=0.1)
    m.minimize(x[0])
    enc = evaluate_interval(dm.sum(x), m)
    exact = 0.1 + 0.1 + 0.1
    _checked(
        float(enc.lo) < exact < float(enc.hi),
        f"the enclosure must strictly straddle the float sum {exact!r}, got "
        f"[{float(enc.lo)!r}, {float(enc.hi)!r}]",
    )
    # And on a degenerate point box it stays tight enough for the residual probe
    # to accept it (the two requirements pull in opposite directions; both hold).
    point = point_from_flat(m, np.array([0.1, 0.1, 0.1]))
    _checked(
        abs(float(evaluate_at_point(m, dm.sum(x), point)[0]) - exact) < 1e-12,
        "and a point evaluation still resolves, so outward rounding did not "
        "widen it past the degeneracy guard",
    )
    _checked(isinstance(enc, Interval), "the reduction returns an Interval")


def test_review2_3_a_stage_residual_refusal_does_not_abort_the_homotopy():
    """MEDIUM 3: the per-stage residual call was bare, so it aborted the solve.

    ``_report_for`` guards the final report, but the in-loop
    ``max_source_complementarity`` did not — so a relation containing an operand
    the interval evaluator cannot walk raised on the first iteration, which is
    strictly worse than the pre-#1148 behaviour.
    """
    import warnings  # noqa: PLC0415

    m = dm.Model("stage_refusal")
    u = m.continuous("u", lb=1.0, ub=2.0)
    v = m.continuous("v", lb=1.0, ub=2.0)
    y = m.continuous("y", lb=0, ub=5)
    m.minimize((y - 1) ** 2 + u + v)
    # The RELATION itself carries the unwalkable atom, so the per-stage call --
    # not just the final report -- must survive it.
    pair = complementarity(u**v - 1.0, y, name="nl")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = solve_mpec(m, [pair], method="scholtes", max_iter=3)

    _checked(
        dstatus.is_local_status(res.status),
        f"the homotopy must still return a local result, got {res.status!r}",
    )
    _checked(
        any(issubclass(w.category, RuntimeWarning) for w in caught),
        "and the unmeasured residual is reported, not swallowed",
    )
    trace = res.mpec_report.continuation if res.mpec_report else None
    if trace is not None:
        _checked(
            all(st.source_complementarity is None for st in trace.stages),
            "stages record source_complementarity=None — NOT MEASURED, not 0.0",
        )
    else:
        _checked(True, "the final report also refused, which is equally acceptable")


def test_review2_4_a_partial_result_x_does_not_abort_a_finished_solve():
    """MEDIUM 4: ``_result_point`` ran as an argument expression, outside the guard.

    Its refusal (a result missing one of the model's variables) therefore bypassed
    ``_report_for``'s ``except`` and turned a certified ``Model.solve`` into an
    exception out of ``solve_mpec`` — the exact HIGH-3 class the guard closed.
    """
    from discopt.mpec import _report_for  # noqa: PLC0415

    m, x, y = _distance_model()
    pair = complementarity(x, y, name="c")
    m._complementarities.append(pair)

    def raising_point():
        raise ValueError("solve result does not carry variable 'y'")

    import warnings  # noqa: PLC0415

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = _report_for(m, [pair], (None, None), raising_point, kind="auto")
    _checked(got is None, "a point that cannot be resolved yields no report, not an exception")
    _checked(
        any(issubclass(w.category, RuntimeWarning) for w in caught),
        "and it is reported loudly",
    )


def test_review2_5_a_stalled_but_usable_stage_is_taken_forward():
    """MEDIUM 5: only OPTIMAL stages were accepted, discarding good late iterates.

    Scholtes subproblems become degenerate as ``t -> 0`` (MFCQ fails in the limit),
    so Ipopt code 3 -> ``ITERATION_LIMIT`` at small ``t`` is the expected case. The
    old behaviour reported the last *certified* stage, which on such a run is a
    large-``t`` point with a residual orders of magnitude worse.
    """
    from discopt.solvers import SolveStatus  # noqa: PLC0415

    m, x, y = _distance_model()
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes")
    trace = res.mpec_report.continuation
    accepted = [st for st in trace.stages if st.accepted]
    _checked(bool(accepted), "at least one stage is accepted on a healthy run")
    _checked(
        float(trace.final_t) == pytest.approx(min(st.t for st in accepted)),
        f"the reported point comes from the smallest accepted t, got {trace.final_t!r}",
    )

    # The status set is the contract: a stalled-with-a-point stage counts, an
    # INFEASIBLE or ERROR one does not — its "point" is wherever restoration
    # stopped and stands for nothing.
    from discopt.mpec import _solve_scholtes  # noqa: F401, PLC0415

    src = pathlib.Path(discopt_mpec_file).read_text()
    _checked(
        "SolveStatus.OPTIMAL, SolveStatus.ITERATION_LIMIT" in src,
        "usable stages are OPTIMAL and ITERATION_LIMIT",
    )
    _checked(
        "SolveStatus.INFEASIBLE" not in src.split("_USABLE_STAGE_STATUSES")[1][:200],
        "and INFEASIBLE is deliberately excluded",
    )
    _checked(SolveStatus.ITERATION_LIMIT is not SolveStatus.OPTIMAL, "the two stay distinct")


def test_review2_5_converged_is_false_when_the_point_came_from_a_larger_t():
    """MEDIUM 5b: reaching t_min while reporting an earlier point is not convergence."""
    trace = ContinuationTrace(
        parameter="t",
        stages=(ContinuationStage(0, 1e-1, "optimal", True, "converged"),),
        final_t=1e-1,
        termination_reason="t_min_reached_but_best_point_is_from_a_larger_t",
        admitted_residual_scale=admitted_residual_scale(1e-1),
        any_stage_accepted=True,
    )
    _checked(
        not trace.converged,
        "the schedule reached t_min but the reported point did not — not converged",
    )


def test_review3_1_a_carried_report_cannot_vouch_for_a_different_point():
    """BLOCKING 1: gating on ``result.mpec_report`` authorized an unmeasured point.

    A :class:`SourceResidualReport` is a measurement of one ``(model, x)`` pair
    and carries no record of which point it was taken at. ``accept_local_incumbent``
    used to accept the report carried on ``result`` and then verify a *different*
    ``x_flat``, so a result whose report was clean at the true solution vouched
    for a violating point nearby — here a report taken at ``(0, 0)`` (exactly
    complementary) admitting ``(1e-4, 1e-4)``, whose source residual is ``1e-4``
    and whose objective ``-2e-4`` beats the true optimum of ``0``. Fed to a global
    solve as a cutoff, that fathoms the optimum away, which is the #1148 §C
    hazard the function exists to block.

    The fix recomputes at the boundary, so the gate is a statement about the
    point and the relations actually in front of it. The control below is what
    shows this is not a blanket refusal.
    """
    m = dm.Model("stale_report")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(-(x + y))
    m.complementarity(x, y, name="c")

    clean = source_residual_report(m, x_flat=np.array([0.0, 0.0]))
    _checked(clean.source_satisfied, "the report is taken at a genuinely complementary point")

    violating = np.array([1e-4, 1e-4])
    fresh = source_residual_report(m, x_flat=violating)
    _checked(
        not fresh.source_satisfied,
        f"and the candidate genuinely violates the relation "
        f"({fresh.complementarity.value:.3e}) — otherwise this probe measures nothing",
    )

    class _Result:
        x = {"x": 0.0, "y": 0.0}
        mpec_report = clean
        objective = 0.0

    _checked(
        accept_local_incumbent(m, _Result(), x_flat=violating) is None,
        "a report taken elsewhere must not authorize this point",
    )
    _checked(
        accept_local_incumbent(m, _Result()) is not None,
        "control: the point the report WAS taken at is still accepted",
    )


def test_review3_1_a_stale_report_cannot_vouch_across_a_model_change():
    """BLOCKING 1, second face: the report is not tied to the model either.

    Relations are added, rebuilt and re-scaled between a solve and the moment an
    incumbent is offered — #1147 exists precisely because models get rebuilt. A
    report taken before a relation was declared says nothing about the model that
    now carries it, so binding the report to the *point* alone would still leave
    this hole open; recomputing closes both.
    """
    m = dm.Model("late_relation")
    x = m.continuous("x", lb=0, ub=10)
    y = m.continuous("y", lb=0, ub=10)
    m.minimize(-(x + y))

    point = np.array([1.0, 1.0])
    before = source_residual_report(m, pairs=(), x_flat=point)
    _checked(
        before.n_scalar_relations == 0,
        "the report predates the relation, so it measured nothing about it",
    )

    m.complementarity(x, y, name="late")

    class _Result:
        x = {"x": 1.0, "y": 1.0}
        mpec_report = before
        objective = -2.0

    _checked(
        accept_local_incumbent(m, _Result(), x_flat=point) is None,
        "a report from before the relation existed must not vouch for (1, 1)",
    )


def test_review3_3_a_zero_iteration_subsolver_does_not_report_local_optimal():
    """BLOCKING 3: an iteration-limited iterate was promoted to a stationary point.

    ``local_optimal`` is defined by :mod:`discopt.status` as a *local stationary
    point*. With the subsolver allowed zero iterations it does no optimization at
    all and hands back the starting point under ``ITERATION_LIMIT``; the wrapper
    published that as ``local_optimal`` — here the starting point ``(5, 5)``, at
    which the generated product row is violated by ``24``. The merge base kept
    ``ITERATION_LIMIT``, so this was a regression introduced by the local-status
    work, in the one direction the vocabulary exists to prevent.

    The point is still reported: it is a usable warm start and carries its
    residuals (that is #1158 review 2, MEDIUM 5, and the assertions below hold
    it). What changes is the label — ``local_limit``, which is in
    ``LOCAL_STATUSES``, so the no-dual-bound chokepoint still applies and no
    consumer reads it as stationary.
    """
    from discopt.status import LOCAL_LIMIT, LOCAL_OPTIMAL, LOCAL_STATUSES  # noqa: PLC0415

    m, x, y = _distance_model()
    res = solve_mpec(
        m,
        [complementarity(x, y, name="c")],
        method="scholtes",
        max_iter=1,
        x0=np.array([5.0, 5.0]),
        nlp_options={"max_iter": 0},
    )
    _checked(
        res.status != LOCAL_OPTIMAL,
        f"a subsolver that did zero iterations did not find a stationary point, "
        f"got status {res.status!r}",
    )
    _checked(res.status == LOCAL_LIMIT, f"it is a limit termination, got {res.status!r}")
    _checked(
        res.status in LOCAL_STATUSES,
        "and it stays inside LOCAL_STATUSES, so the no-dual-bound guard still applies",
    )
    _checked(
        res.bound is None and not res.gap_certified,
        f"no bound, no certification: bound={res.bound!r} gap_certified={res.gap_certified!r}",
    )
    # The iterate is RETAINED — the fix is about the claim, not about the point.
    _checked(res.x is not None, "the warm start is still reported")
    trace = res.mpec_report.continuation
    _checked(
        not trace.converged and not trace.reported_point_certified,
        "and the trace says plainly that the reported point did not converge",
    )
    _checked(
        trace.any_stage_accepted and not trace.any_stage_certified,
        "accepting an iterate and converging are now distinct in the trace",
    )
    _checked(
        trace.as_dict()["reported_point_certified"] is False,
        "and the distinction survives serialization",
    )


def test_review3_3_a_converged_stage_still_reports_local_optimal():
    """Control for BLOCKING 3: the demotion is driven by evidence, not blanket.

    A healthy homotopy whose reported point came from a converged stage must
    still say ``local_optimal`` — otherwise the fix would have removed the
    status the local mode exists to publish rather than reserved it.
    """
    from discopt.status import LOCAL_OPTIMAL  # noqa: PLC0415

    m, x, y = _distance_model()
    res = solve_mpec(m, [complementarity(x, y, name="c")], method="scholtes")
    trace = res.mpec_report.continuation
    _checked(
        trace.any_stage_certified,
        "the control run must actually have a converged stage, or it measures nothing",
    )
    if trace.reported_point_certified:
        _checked(
            res.status == LOCAL_OPTIMAL,
            f"a converged reported point is local_optimal, got {res.status!r}",
        )
    else:
        # A late-stage stall is the expected Scholtes outcome; then the honest
        # label is the weaker one, and the point is still reported.
        _checked(
            res.status == "local_limit" and res.x is not None,
            f"a stalled reported point is local_limit with the point kept, got {res.status!r}",
        )

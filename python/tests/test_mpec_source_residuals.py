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
    complementarity_accuracy_floor,
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


def test_continuation_trace_is_kept_and_carries_the_accuracy_floor():
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
        abs(float(trace.accuracy_floor) - 1e-4) < 1e-12,
        f"the sqrt(t) accuracy floor at t=1e-8 is 1e-4, got {trace.accuracy_floor!r}",
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


def test_accuracy_floor_is_sqrt_t_and_absent_without_one():
    _checked(
        abs(complementarity_accuracy_floor(1e-8) - 1e-4) < 1e-15,
        "sqrt(1e-8) must be 1e-4",
    )
    _checked(complementarity_accuracy_floor(None) is None, "no t means no floor")
    _checked(complementarity_accuracy_floor(0.0) is None, "t=0 imposes no floor")


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
        payload["continuation"]["stages"] and payload["continuation"]["accuracy_floor"],
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
        accuracy_floor=complementarity_accuracy_floor(1e-2),
    )
    converged = ContinuationTrace(
        parameter="t",
        stages=(ContinuationStage(0, 1e-8, "optimal", True, "subsolver converged"),),
        final_t=1e-8,
        termination_reason="t_min_reached",
        accuracy_floor=complementarity_accuracy_floor(1e-8),
    )
    _checked(
        not stalled.converged and converged.converged,
        "a stall and a converged schedule must be distinguishable from the trace alone",
    )
    _checked(
        stalled.as_dict()["converged"] is False,
        "and the distinction must survive serialization",
    )


def test_zz_probe_actually_fired():
    """CLAUDE.md §6: fail when this module's residual comparisons never executed."""
    assert ASSERTIONS_EXECUTED > 0, (
        "the #1148 residual probes executed ZERO comparisons. A probe that traverses "
        "nothing prints a clean pass; this counter exists so that cannot happen."
    )
    print(f"\n#1148 residual probe: {ASSERTIONS_EXECUTED} comparisons executed")

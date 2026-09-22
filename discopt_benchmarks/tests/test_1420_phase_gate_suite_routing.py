"""The phase gate evaluator must measure what benchmarks.toml says it measures (#1420).

Three defects, all of the same family — an instrument that reports a verdict it did
not earn:

  A. Each criterion declares the ``suite`` it is measured on. The evaluator never
     read that key, so every criterion was evaluated against whichever single results
     file the CLI had loaded. Two criteria declared against two different suites
     reported identical numbers.
  B. Ten ``*_ratio_vs_<solver>`` criteria were guarded on a ``reference_solvers``
     argument the CLI never passed, so they sat at NaN and rendered as ordinary
     failures.
  C. ``max_residual_vs_suitesparse`` has no implementing branch and did the same.

The fix is not "make them pass" — most of them still cannot pass, because the panels
do not exist. The fix is that a criterion nobody measured is reported as
``not_measured`` with a reason, *while still blocking the gate*. These tests pin both
halves: the routing/derivation that makes measurable criteria measurable, and the
refusal that keeps an unmeasurable one from being mistaken for either a pass or a
solver failure.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BENCH_ROOT = Path(__file__).resolve().parents[1]
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from benchmarks.metrics import (  # noqa: E402
    BenchmarkResults,
    SolveResult,
    SolveStatus,
    evaluate_phase_gate,
    reference_solver_columns,
)


def _res(instance: str, solver: str, *, wall: float, obj: float | None) -> SolveResult:
    return SolveResult(
        instance=instance,
        solver=solver,
        status=SolveStatus.OPTIMAL if obj is not None else SolveStatus.TIME_LIMIT,
        objective=obj,
        bound=obj,
        wall_time=wall,
        node_count=5,
    )


def _bench(suite: str, rows: list[SolveResult]) -> BenchmarkResults:
    b = BenchmarkResults(suite=suite, timestamp="2026-09-22T00:00:00")
    for r in rows:
        b.add_result(r)
    return b


@pytest.fixture
def two_suites() -> dict[str, BenchmarkResults]:
    """Two panels with deliberately DIFFERENT discopt outcomes.

    ``alpha`` solves both instances; ``beta`` solves one of two. A criterion routed
    to the wrong panel therefore reports the wrong number, not a coincidentally
    equal one — which is what makes defect A detectable at all.
    """
    return {
        "alpha": _bench("alpha", [
            _res("i1", "discopt", wall=1.0, obj=1.0),
            _res("i2", "discopt", wall=1.0, obj=2.0),
        ]),
        "beta": _bench("beta", [
            _res("j1", "discopt", wall=1.0, obj=1.0),
            _res("j2", "discopt", wall=1.0, obj=None),
        ]),
    }


def test_each_criterion_is_measured_on_the_suite_it_declares(two_suites):
    """Defect A: two criteria, same metric, different declared suites, different values."""
    gate = {"criteria": {
        "on_alpha": {"min": 0, "suite": "alpha", "metric": "solved_count"},
        "on_beta": {"min": 0, "suite": "beta", "metric": "solved_count"},
    }}
    _, crits = evaluate_phase_gate(
        "g", two_suites["alpha"], gate, suite_results=two_suites
    )
    by_name = {c.name: c for c in crits}
    assert by_name["on_alpha"].actual == 2, by_name["on_alpha"]
    assert by_name["on_beta"].actual == 1, by_name["on_beta"]
    # The pre-fix behaviour: both evaluated against the single passed-in `benchmark`,
    # so both read 2. An equality here means suite routing has regressed.
    assert by_name["on_alpha"].actual != by_name["on_beta"].actual
    # And the declared suite is carried into the report, so the reader can see it.
    assert by_name["on_beta"].suite == "beta"


def test_a_declared_suite_with_no_results_is_not_measured_rather_than_failed(two_suites):
    """Blocks the gate, but is distinguishable from a measured miss."""
    gate = {"criteria": {
        "orphan": {"min": 1, "suite": "nowhere", "metric": "solved_count"},
    }}
    all_passed, crits = evaluate_phase_gate(
        "g", two_suites["alpha"], gate, suite_results=two_suites
    )
    (c,) = crits
    assert c.status == "not_measured"
    assert c.passed is False, "an unmeasured criterion must still block the gate"
    assert all_passed is False
    assert "nowhere" in c.detail


def test_a_measured_miss_is_reported_as_fail_not_not_measured(two_suites):
    """The other arm: `fail` must stay reachable, or the distinction is worthless."""
    gate = {"criteria": {
        "too_few": {"min": 99, "suite": "beta", "metric": "solved_count"},
    }}
    _, (c,) = evaluate_phase_gate("g", two_suites["alpha"], gate, suite_results=two_suites)
    assert c.status == "fail"
    assert c.actual == 1
    assert c.detail == ""


def test_reference_solver_columns_are_derived_from_the_results(two_suites):
    """Defect B: the ratio metrics get their reference column with no caller help."""
    panel = _bench("cmp", [
        _res("i1", "discopt", wall=1.0, obj=1.0),
        _res("i2", "discopt", wall=2.0, obj=2.0),
        _res("i1", "scip", wall=2.0, obj=1.0),
        _res("i2", "scip", wall=4.0, obj=2.0),
        # An ablation column of our own solver is the subject, never a reference.
        _res("i1", "discopt_cpu", wall=9.0, obj=1.0),
    ])
    assert set(reference_solver_columns(panel)) == {"scip"}

    gate = {"criteria": {
        "vs_scip": {"max": 1.0, "suite": "cmp", "metric": "geomean_ratio_vs_scip"},
    }}
    # No `reference_solvers=` argument — exactly how run_benchmarks.py calls it.
    _, (c,) = evaluate_phase_gate("g", panel, gate, suite_results={"cmp": panel})
    assert c.status == "pass", c.detail
    # discopt is uniformly 2x faster; the +1 s shift in the geometric mean pulls the
    # ratio slightly above 1/2 at these small times.
    assert c.actual == pytest.approx(0.5, rel=0.02), c.actual


def test_a_missing_reference_column_names_the_solver_it_wanted(two_suites):
    gate = {"criteria": {
        "vs_baron": {"max": 1.0, "suite": "alpha", "metric": "geomean_ratio_vs_baron"},
    }}
    _, (c,) = evaluate_phase_gate(
        "g", two_suites["alpha"], gate, suite_results=two_suites
    )
    assert c.status == "not_measured"
    assert "baron" in c.detail


def test_an_unimplemented_metric_is_not_measured_rather_than_a_silent_nan(two_suites):
    """Defect C: `max_residual_vs_suitesparse` has no branch and never had one."""
    gate = {"criteria": {
        "sparse_accuracy": {
            "max": 1e-12, "suite": "alpha", "metric": "max_residual_vs_suitesparse",
        },
    }}
    _, (c,) = evaluate_phase_gate(
        "g", two_suites["alpha"], gate, suite_results=two_suites
    )
    assert c.status == "not_measured"
    assert "unknown metric" in c.detail


def test_the_shipped_gates_declare_only_metrics_the_evaluator_implements():
    """Every `metric` in benchmarks.toml resolves to a branch, or is knowingly listed.

    A typo in a criterion's metric name used to be indistinguishable from a solver
    failure. It is now `not_measured`, but the config should still not carry one by
    accident, so this enumerates the real gates and pins the single known exception.
    """
    import tomllib

    from benchmarks.metrics import _is_known_gate_metric

    cfg = tomllib.loads((_BENCH_ROOT / "config" / "benchmarks.toml").read_text())
    gates = cfg.get("gates", {})
    assert gates, "no [gates.*] tables found — this test measured nothing"

    checked = 0
    unimplemented = []
    for gate_name, gate in gates.items():
        for crit_name, crit in gate.get("criteria", {}).items():
            checked += 1
            if not _is_known_gate_metric(crit.get("metric", "")):
                unimplemented.append(f"{gate_name}.{crit_name}={crit.get('metric')!r}")

    assert checked >= 30, f"only checked {checked} criteria — the config did not load"
    # `sparse_accuracy` is deliberately retained with no implementation: there is no
    # SuiteSparse corpus in the tree, so it reports NOT MEASURED and blocks phase1
    # rather than being deleted (which would let the gate go green unmeasured).
    assert unimplemented == ["phase1.sparse_accuracy='max_residual_vs_suitesparse'"], (
        f"unexpected unimplemented gate metric(s): {unimplemented}"
    )


def test_every_gate_criterion_declares_a_suite_that_benchmarks_toml_defines():
    """A criterion naming a suite with no `[suites.*]` table can never be measured."""
    import tomllib

    cfg = tomllib.loads((_BENCH_ROOT / "config" / "benchmarks.toml").read_text())
    suites = set(cfg.get("suites", {}))
    assert suites, "no [suites.*] tables found — this test measured nothing"

    checked = 0
    orphans = []
    for gate_name, gate in cfg.get("gates", {}).items():
        for crit_name, crit in gate.get("criteria", {}).items():
            checked += 1
            suite = crit.get("suite")
            if suite and suite not in suites:
                orphans.append(f"{gate_name}.{crit_name} -> {suite!r}")
    assert checked >= 30, f"only checked {checked} criteria — the config did not load"
    assert not orphans, f"gate criteria naming undefined suites: {orphans}"


def test_every_suite_instance_list_file_exists():
    """`[suites.comparison]` named a file that never existed in git history (#1420 §D)."""
    import tomllib

    cfg = tomllib.loads((_BENCH_ROOT / "config" / "benchmarks.toml").read_text())
    checked = 0
    missing = []
    for name, suite in cfg.get("suites", {}).items():
        path = suite.get("instance_list")
        # `instance_list` may be an inline TOML array instead of a path.
        if not path or isinstance(path, list):
            continue
        checked += 1
        if not (_BENCH_ROOT / path).exists():
            missing.append(f"{name} -> {path}")
    assert checked >= 5, f"only checked {checked} instance lists — the config did not load"
    assert not missing, f"suites naming a non-existent instance_list: {missing}"


def test_a_suite_whose_corpus_has_no_loader_is_refused_not_silently_substituted():
    """`sources` was decorative: `--suite lp_netlib` ran 122 MINLPLib instances.

    Measured 2026-09-22 on `6dc49ebe`: lp_netlib, nlp_cutest, sparse_matrices and
    lp_kennington each resolved to the same 122-instance MINLPLib panel and would
    have been saved under their own names. A mislabelled panel is worse than a
    missing one, so the runner must exit rather than run.
    """
    import run_benchmarks as rb

    declared_elsewhere = []
    for name, suite in rb._load_toml_config().get("suites", {}).items():
        srcs = suite.get("sources", []) or []
        if any(s not in rb._LOADABLE_SOURCES for s in srcs):
            declared_elsewhere.append(name)
    assert declared_elsewhere, (
        "no suite declares a non-loadable source any more — if those suites were "
        "removed, remove this test with them rather than letting it assert nothing"
    )

    for name in declared_elsewhere:
        with pytest.raises(SystemExit) as exc:
            rb._require_loadable_sources(name, rb._load_suite_config(name))
        assert exc.value.code != 0, f"{name} exited 0 despite an unloadable source"

    # And the control arm: a MINLPLib suite is not refused.
    rb._require_loadable_sources("phase1", rb._load_suite_config("phase1"))


def test_an_empty_solved_intersection_is_not_measured_not_an_infinite_ratio():
    """`geometric_mean_ratio` returns inf when no instance is solved by both.

    Rendered as a number that is a FAIL reading "catastrophically slower"; it is
    really an empty comparison. `--gate phase3` showed exactly this against the
    BARON column before the fix.
    """
    panel = _bench("cmp", [
        _res("i1", "discopt", wall=1.0, obj=1.0),
        _res("i2", "baron", wall=1.0, obj=1.0),  # disjoint instance
    ])
    gate = {"criteria": {
        "vs_baron": {"max": 2.5, "suite": "cmp", "metric": "geomean_ratio_vs_baron"},
    }}
    _, (c,) = evaluate_phase_gate("g", panel, gate, suite_results={"cmp": panel})
    assert c.status == "not_measured"
    assert "empty set" in c.detail


def test_find_latest_results_skips_files_with_a_foreign_schema(tmp_path, monkeypatch):
    """`reports/` is shared: the glob also matches gate reports and the 3-way script.

    `--gate phase3` and `--gate cert0` died with `KeyError: 'suite'` because the
    newest `phase3_*.json` was a gate report, not a results file.
    """
    import run_benchmarks as rb

    reports = tmp_path / "reports"
    reports.mkdir()
    # Newest by name: a foreign schema that BenchmarkResults.load cannot read.
    (reports / "phase3_gate_2099-01-01.json").write_text('{"criteria": [], "passed": true}')
    # Older, but the real thing.
    good = reports / "phase3_2026-01-01.json"
    good.write_text('{"suite": "phase3", "timestamp": "t", "solver_results": {}}')
    (reports / "phase3_broken_2098-01-01.json").write_text("not json at all")

    monkeypatch.chdir(tmp_path)
    found = rb._find_latest_results("phase3")
    assert found is not None, "the valid results file was skipped too"
    assert found.name == good.name, found
    # And it really does load, which is the property the gate needs.
    assert BenchmarkResults.load(found).suite == "phase3"


def test_suite_results_override_maps_a_panel_saved_under_another_name(tmp_path):
    """`reports/cert0_*.json` carries suite="cert0" but IS the global50 panel.

    Auto-discovery must not assume that (it is the silent substitution #1420 is
    about); `--suite-results global50=<path>` lets the operator state it, and the
    mapping is echoed into the run's output.
    """
    import run_benchmarks as rb

    panel = _bench("cert0", [_res("i1", "discopt", wall=1.0, obj=1.0)])
    path = tmp_path / "cert0_2026-01-01.json"
    panel.save(path)

    gate = {"criteria": {
        "c": {"min": 1, "suite": "global50", "metric": "solved_count"},
    }}
    loaded = rb._load_suite_results_for_gate(
        gate, "cert0", panel, [f"global50={path}"]
    )
    assert "global50" in loaded
    _, (c,) = evaluate_phase_gate("cert0", panel, gate, suite_results=loaded)
    assert c.status == "pass", c.detail

    # Without the override the same gate refuses rather than substituting.
    loaded_auto = rb._load_suite_results_for_gate(gate, "cert0", panel, [])
    _, (c2,) = evaluate_phase_gate("cert0", panel, gate, suite_results=loaded_auto)
    assert c2.status == "not_measured", c2


def test_suite_results_override_rejects_a_malformed_spec(tmp_path):
    import run_benchmarks as rb

    panel = _bench("cert0", [])
    for bad in ["no-equals-sign", f"global50={tmp_path / 'nope.json'}"]:
        with pytest.raises(SystemExit) as exc:
            rb._load_suite_results_for_gate({"criteria": {}}, "cert0", panel, [bad])
        assert exc.value.code != 0, bad


def test_an_inline_instance_list_resolves_instead_of_raising():
    """`instance_list` may be a TOML array (minlptests_smoke); `Path(<list>)` raised."""
    import run_benchmarks as rb

    names = rb._read_instance_list(["a", "b", "c"])
    assert names == {"a", "b", "c"}
    assert rb._instance_list_order(["a", "b"]) == ["a", "b"]

    # And the shipped suite that has one now resolves through the real loader.
    cfg = rb._load_suite_config("minlptests_smoke")
    assert isinstance(cfg.get("instance_list"), list), (
        "minlptests_smoke no longer declares an inline list — if inline lists are "
        "gone from the config, drop this arm rather than letting it assert nothing"
    )
    instances, _ = rb._load_minlplib_instances(cfg)
    assert isinstance(instances, list)


def test_single_panel_mode_is_opt_in_by_omitting_the_map(two_suites):
    """`suite_results=None` keeps the pre-#1420 single-panel semantics on purpose.

    It exists only for unit tests that drive one metric against one stub panel. The
    distinction is load-bearing, so pin both arms: omitting the map evaluates every
    criterion against `benchmark`; passing even an EMPTY map turns on strict routing.
    Production callers (the CLI, generate_report) always pass a map.
    """
    gate = {"criteria": {
        "elsewhere": {"min": 1, "suite": "nowhere", "metric": "solved_count"},
    }}
    # No map: the passed-in panel answers, so the criterion is measured.
    _, (legacy,) = evaluate_phase_gate("g", two_suites["alpha"], gate)
    assert legacy.status == "pass", legacy
    assert legacy.actual == 2

    # Empty map: strict. The declared suite is absent, so nothing is substituted.
    _, (strict,) = evaluate_phase_gate("g", two_suites["alpha"], gate, suite_results={})
    assert strict.status == "not_measured", strict


def test_generate_report_renders_not_measured_distinctly():
    """The human-facing markdown must not show an unmeasured criterion as a plain 🔴."""
    from utils.reporting import generate_report

    panel = _bench("phase1", [_res("i1", "discopt", wall=1.0, obj=1.0)])
    gate_config = {"criteria": {
        "measured": {"min": 1, "suite": "phase1", "metric": "solved_count"},
        "absent": {"min": 1, "suite": "nlp_cutest", "metric": "solved_count"},
    }}
    md = generate_report(
        panel, gate_name="phase1", gate_config=gate_config, suite_results={"phase1": panel}
    )
    assert "not measured" in md, md[md.find("## Phase Gate"):][:800]
    assert "nlp_cutest" in md, "the declared suite is not shown in the report"

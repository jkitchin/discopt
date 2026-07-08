import importlib.util
import sys
from pathlib import Path


def _load_baseline_module():
    """Load the baseline script as a module for direct helper testing."""
    path = Path(__file__).resolve().parents[2] / "scripts" / "shot_parity_baseline.py"
    spec = importlib.util.spec_from_file_location("shot_parity_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shot_parity_fixture_set_covers_issue_acceptance_classes():
    """Verify the fixture set covers the issue's accepted problem classes."""
    module = _load_baseline_module()

    fixtures = module.fixture_specs()
    classes = {fixture.problem_class for fixture in fixtures}

    assert "NLP" in classes
    assert "MIQP" in classes
    assert any(f.problem_class == "MINLP" and f.convexity == "convex" for f in fixtures)
    assert any(f.problem_class == "MINLP" and f.convexity == "nonconvex" for f in fixtures)
    assert any(f.expected_certification == "heuristic" for f in fixtures)
    assert all("time_limit" in fixture.solve_options for fixture in fixtures)

    by_key = {fixture.key: fixture for fixture in fixtures}
    assert by_key["convex_minlp"].solve_options["mip_nlp_profile"] == "shot"
    assert by_key["convex_minlp"].solve_options["direct_quadratic_routing"] == "off"
    assert by_key["miqp"].solve_options["mip_nlp_profile"] == "shot"


def test_shot_unavailable_result_records_explicit_caveat(tmp_path):
    """Verify unavailable SHOT rows preserve the diagnostic caveat."""
    module = _load_baseline_module()

    result = module.unavailable_shot_result("not built", tmp_path / "SHOT")

    assert result["backend"] == "SHOT"
    assert result["available"] is False
    assert result["status"] == "unavailable"
    assert result["bound_validity"] == "not_run"
    assert result["certification_caveat"] == "not built"


def test_collect_baseline_empty_selection_has_stable_schema(tmp_path):
    """Verify empty fixture selections still emit a stable baseline schema."""
    module = _load_baseline_module()

    baseline = module.collect_baseline(
        set(),
        include_shot=False,
        shot_root=tmp_path / "SHOT",
        workdir=tmp_path / "work",
    )

    assert baseline["schema_version"] == 1
    assert baseline["fixtures"] == []
    assert baseline["shot_executable"] is None
    assert baseline["workdir"] == str(tmp_path / "work")


def test_shot_status_does_not_treat_optimality_caveat_as_optimal():
    """Verify SHOT caveats keep feasible results uncertified."""
    module = _load_baseline_module()
    output = """
    Feasible primal solution found to convex problem.
    Can not guarantee optimality to the given termination criteria.
    Objective bound (minimization) [dual, primal]:  [-9.6875, 1.5625].
    Objective gap absolute / relative:              11.25 / 7.2.
    """

    status = module._parse_shot_status(output, 0)

    assert status == "feasible"
    assert module._parse_shot_gap_certified(output, status, 0) is False
    assert module._parse_shot_bound_pair(output) == (-9.6875, 1.5625)
    assert module._parse_shot_gap_pair(output) == (11.25, 7.2)


def test_shot_status_recognizes_global_optimality_report():
    """Verify SHOT global optimality reports are parsed as certified."""
    module = _load_baseline_module()
    output = """
    Terminated since absolute gap met requirements.
    Globally optimal primal solution found.
    Objective bound (minimization) [dual, primal]:  [0.0496452, 0.0500009].
    Objective gap absolute / relative:              0.000355636 / 0.0071126.
    Unfulfilled termination criteria:
     - solution time limit (s)                      0.446913 <= 30
    Termination.TimeLimit = 30
    """

    status = module._parse_shot_status(output, 0)

    assert status == "optimal"
    assert module._parse_shot_gap_certified(output, status, 0) is True


def test_shot_osrl_metrics_prefer_structured_bounds(tmp_path):
    """Verify structured OSrL metrics are parsed for SHOT bounds and gaps."""
    module = _load_baseline_module()
    osrl_path = tmp_path / "result.osrl"
    osrl_path.write_text(
        """<osrl xmlns="os.optimizationservices.org">
  <general>
    <otherResults>
      <other name="DualObjectiveBound" value="0.049645244896349541"/>
      <other name="PrimalObjectiveBound" value="0.050000880952378424"/>
      <other name="AbsoluteOptimalityGap" value="0.00035563605602888237"/>
      <other name="RelativeOptimalityGap" value="0.0071125957891889106"/>
    </otherResults>
  </general>
</osrl>
"""
    )

    metrics = module._parse_shot_osrl_metrics(osrl_path)

    assert metrics["DualObjectiveBound"] == 0.049645244896349541
    assert metrics["PrimalObjectiveBound"] == 0.050000880952378424
    assert metrics["AbsoluteOptimalityGap"] == 0.00035563605602888237
    assert metrics["RelativeOptimalityGap"] == 0.0071125957891889106

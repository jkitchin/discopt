import importlib.util
import sys
from pathlib import Path


def _load_baseline_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "shot_parity_baseline.py"
    spec = importlib.util.spec_from_file_location("shot_parity_baseline", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_shot_parity_fixture_set_covers_issue_acceptance_classes():
    module = _load_baseline_module()

    fixtures = module.fixture_specs()
    classes = {fixture.problem_class for fixture in fixtures}

    assert "NLP" in classes
    assert "MIQP" in classes
    assert any(f.problem_class == "MINLP" and f.convexity == "convex" for f in fixtures)
    assert any(f.problem_class == "MINLP" and f.convexity == "nonconvex" for f in fixtures)
    assert any(f.expected_certification == "heuristic" for f in fixtures)
    assert all("time_limit" in fixture.solve_options for fixture in fixtures)


def test_shot_unavailable_result_records_explicit_caveat(tmp_path):
    module = _load_baseline_module()

    result = module.unavailable_shot_result("not built", tmp_path / "SHOT")

    assert result["backend"] == "SHOT"
    assert result["available"] is False
    assert result["status"] == "unavailable"
    assert result["bound_validity"] == "not_run"
    assert result["certification_caveat"] == "not built"


def test_collect_baseline_empty_selection_has_stable_schema(tmp_path):
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

"""The two-tier NLP benchmark measures what it says it measures.

Tier 2 (MINLPLib pure NLP): the old ``nonconvex_nlp`` suite was 11 hand-labelled
names of which 10 carry integer variables (MBNLP/MINLP/IQCP/...), and
``convex_nlp`` matched nothing. The ``nlp_smoke``/``nlp`` suites are drawn from
``instancedata.csv`` by probtype instead; these tests pin that.

Tier 1 (CUTEst): the scoring in ``run_cutest_benchmarks.summarize`` decides
what counts as wrong / lost / differs, and the runner's reference arm
(``ipopt_standalone``) used to raise on every constrained problem and was
recorded as a silent ERROR row.
"""

from __future__ import annotations

import sys
import tomllib
from pathlib import Path

import pytest

_BENCH = Path(__file__).resolve().parents[1]
if str(_BENCH) not in sys.path:
    sys.path.insert(0, str(_BENCH))

from benchmarks.metrics import BenchmarkResults, SolveResult, SolveStatus  # noqa: E402
from scripts import make_suites  # noqa: E402
from scripts import run_cutest_benchmarks as rcb  # noqa: E402
from utils.minlplib_data import InstanceMeta, load_instance_data  # noqa: E402

_SUITES = _BENCH / "config" / "suites"
_INSTANCEDATA = Path.home() / ".cache/discopt/minlplib/current/instancedata.csv"
needs_index = pytest.mark.skipif(
    not _INSTANCEDATA.exists(), reason="needs the MINLPLib instancedata.csv cache"
)


def _names(path: Path) -> list[str]:
    return [ln.strip() for ln in path.read_text().splitlines() if ln.strip() and ln[0] != "#"]


def _meta(name, probtype, n_vars, proven, formats="{'gms', 'nl'}", n_int=0):
    return InstanceMeta(
        name=name,
        probtype=probtype,
        n_vars=n_vars,
        n_intvars=n_int,
        proven_optimal=proven,
        raw={"formats": formats},
    )


# ── Tier 2: MINLPLib selection ───────────────────────────────────────────────


@pytest.mark.smoke
def test_has_nl_reads_the_formats_column():
    assert _meta("a", "NLP", 3, True).has_nl is True
    assert _meta("b", "NLP", 3, True, formats="{'gms', 'osil', 'py'}").has_nl is False
    assert InstanceMeta(name="c").has_nl is None


@pytest.mark.smoke
def test_nlp_tier_takes_only_proven_pure_nlp_with_an_nl_file():
    index = {
        m.name: m
        for m in [
            _meta("keep_small", "NLP", 5, True),
            _meta("keep_mid", "NLP", 250, True),
            _meta("unproven", "NLP", 5, False),
            _meta("no_nl", "NLP", 5, True, formats="{'gms', 'osil'}"),
            _meta("too_big", "NLP", 900, True),
            _meta("mbnlp", "MBNLP", 5, True, n_int=3),
            _meta("qcp", "QCP", 5, True),
        ]
    }
    assert make_suites.make_nlp(index) == ["keep_mid", "keep_small"]
    assert make_suites.make_nlp_smoke(index) == ["keep_small"]


@pytest.mark.smoke
def test_nlp_suites_are_wired_to_the_list_files():
    with open(_BENCH / "config" / "benchmarks.toml", "rb") as f:
        suites = tomllib.load(f)["suites"]
    for name in ("nlp_smoke", "nlp"):
        assert suites[name]["instance_list"] == f"config/suites/{name}.txt"
        assert _names(_BENCH / suites[name]["instance_list"]), f"{name} list is empty"
    # The problem_class suites resolved to 0 (convex_nlp) or to integer
    # instances (nonconvex_nlp); they must not come back.
    assert "convex_nlp" not in suites and "nonconvex_nlp" not in suites


@needs_index
def test_nlp_list_files_hold_only_proven_pure_nlp():
    index = load_instance_data(_INSTANCEDATA)
    checked = 0
    for name in ("nlp_smoke", "nlp"):
        for inst in _names(_SUITES / f"{name}.txt"):
            meta = index[inst]
            assert meta.probtype == "NLP", (inst, meta.probtype)
            assert meta.n_binvars + meta.n_intvars == 0, inst
            assert meta.proven_optimal and meta.known_optimum is not None, inst
            assert meta.has_nl is True, inst
            checked += 1
    assert checked >= 100, f"only {checked} list entries checked"


@needs_index
def test_instance_classes_label_nothing_with_integers_as_nlp():
    with open(_BENCH / "config" / "instance_classes.toml", "rb") as f:
        labels = tomllib.load(f)["instances"]
    index = load_instance_data(_INSTANCEDATA)
    checked = 0
    for name, cls in labels.items():
        if not cls.endswith("_nlp") or name not in index:
            continue
        meta = index[name]
        assert meta.n_binvars + meta.n_intvars == 0, f"{name} ({meta.probtype}) labelled {cls}"
        checked += 1
    assert checked >= 1


# ── Tier 1: CUTEst scoring ───────────────────────────────────────────────────


def _results(rows):
    res = BenchmarkResults(suite="t", timestamp="now")
    for solver, inst, status, obj in rows:
        res.add_result(
            SolveResult(instance=inst, solver=solver, status=status, objective=obj, wall_time=0.01)
        )
    return res


@pytest.mark.smoke
def test_summarize_separates_wrong_lost_and_differs():
    opt, err = SolveStatus.OPTIMAL, SolveStatus.ERROR
    res = _results(
        [
            ("ipopt_standalone", "A", opt, 1.0),
            ("ipopt_standalone", "B", opt, 2.0),
            ("ipopt_standalone", "C", opt, 3.0),
            ("ipopt_standalone", "D", err, None),
            ("discopt_pounce", "A", opt, 1.0 + 1e-9),  # agrees
            ("discopt_pounce", "B", opt, 2.5),  # differs from the reference
            ("discopt_pounce", "C", SolveStatus.TIME_LIMIT, None),  # lost
            ("discopt_pounce", "D", opt, 7.0),  # wrong vs known optimum
        ]
    )
    s = rcb.summarize(res, known_optima={"D": 4.0})
    arm = s["solvers"]["discopt_pounce"]
    assert [w[0] for w in arm["wrong"]] == ["D"]
    assert [d[0] for d in arm["differs"]] == ["B"]
    assert [lo[0] for lo in arm["lost"]] == ["C"]
    assert (arm["converged"], arm["limit"], arm["error"]) == (3, 1, 0)
    assert s["solvers"]["ipopt_standalone"]["error"] == 1
    # 3 reference comparisons + 1 known-optimum comparison
    assert s["comparisons"] == 4


@pytest.mark.smoke
def test_objective_agreement_uses_a_floored_relative_tolerance():
    assert rcb.objectives_agree(0.0, 5e-5)
    assert not rcb.objectives_agree(0.0, 5e-4)
    assert rcb.objectives_agree(1e6, 1e6 + 50)
    assert not rcb.objectives_agree(1e6, 1e6 + 500)


@pytest.mark.smoke
def test_every_cutest_suite_builds_and_unknown_keys_are_refused():
    config = rcb.load_config()
    for name in config["suites"]:
        assert rcb.suite_config(config, name).name == name
    bad = {"suites": {"x": {"max_varaibles": 3}}}
    with pytest.raises(SystemExit):
        rcb.suite_config(bad, "x")


def _cutest_usable() -> bool:
    # pycutest reads PYCUTEST_CACHE once, at import; pin it before that happens.
    rcb._prepare_environment()
    try:
        import pycutest

        pycutest.find_problems(constraints="unconstrained", n=[2, 2])
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _cutest_usable(), reason="needs a working CUTEst install")
def test_cutest_smoke_runs_every_arm_on_constrained_problems(tmp_path, monkeypatch):
    """End to end on two problems; the reference arm must not error on HS71."""
    pytest.importorskip("cyipopt")
    monkeypatch.chdir(tmp_path)
    out = tmp_path / "smoke.json"
    code = rcb.main(
        [
            "--suite",
            "cutest_smoke",
            "--max-instances",
            "2",
            "--solvers",
            "discopt_pounce,ipopt_standalone",
            "--output",
            str(out),
        ]
    )
    assert code == 0
    assert out.exists() and out.with_suffix(".md").exists()

    config = rcb.load_config()
    cfg = rcb.suite_config(config, "cutest_smoke", problem_names=["HS71"])
    from benchmarks.cutest_runner import CUTEstBenchmarkRunner

    runner = CUTEstBenchmarkRunner(cfg)
    res = runner.run_all(solvers=["discopt_pounce", "ipopt_standalone"], verbose=False)
    assert runner.errors == {}
    for solver in ("discopt_pounce", "ipopt_standalone"):
        (row,) = res.get_results(solver)
        assert row.status == SolveStatus.OPTIMAL, (solver, row.status)
        assert row.objective == pytest.approx(17.0140173, rel=1e-6)

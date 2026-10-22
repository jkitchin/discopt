"""The fast lane's per-test time budget (python/tests/_time_budget.py).

A budget that silently passed everything would read as a lane under control while
the tests it exists to catch kept landing. These pin that it fails an over-budget
test, spares an allowlisted one, stays off without the option, and reports the
same verdicts under xdist, where the hook that decides and the summary that
reports run in different processes.
"""

import json
from pathlib import Path

import pytest

pytest_plugins = ["pytester"]

CONFTEST = (
    "import _time_budget\n"
    "def pytest_addoption(parser):\n"
    "    _time_budget.addoption(parser)\n"
    "def pytest_configure(config):\n"
    "    _time_budget.configure(config)\n"
)

TESTS = (
    "import time\n"
    "def test_fast(): pass\n"
    "def test_slow(): time.sleep(0.4)\n"
    "def test_slow_allowlisted(): time.sleep(0.4)\n"
    "def test_was_slow(): pass\n"
    "def test_slow_but_fails_anyway():\n"
    "    time.sleep(0.4)\n"
    "    assert False, 'its own failure'\n"
)


@pytest.fixture
def suite(pytester):
    pytester.makeconftest(CONFTEST)
    pytester.makepyfile(test_mod=TESTS)
    allow = pytester.path / "allow.json"
    allow.write_text(
        json.dumps({"test_mod.py::test_slow_allowlisted": 0.4, "test_mod.py::test_was_slow": 9.0})
    )
    return pytester, allow


def _run(pytester, allow, *extra):
    return pytester.runpytest(
        "-p",
        "no:cacheprovider",
        "--time-budget",
        "0.2",
        "--time-budget-allowlist",
        str(allow),
        *extra,
    )


def test_over_budget_test_fails_and_allowlisted_one_passes(suite):
    pytester, allow = suite
    result = _run(pytester, allow)
    result.assert_outcomes(passed=3, failed=2)
    result.stdout.fnmatch_lines(
        [
            "*test_mod.py::test_slow took 0.* s, over the fast-lane budget of 0.2 s*",
            "*1 over budget, 2 allowlisted, 1 allowlisted ran under 0.08 s*",
            "*OVER*test_mod.py::test_slow",
            "*could leave the allowlist (9.0 -> 0.0 s)*test_mod.py::test_was_slow",
        ]
    )
    # A test that failed on its own keeps its own failure; the budget does not
    # rewrite it or count it.
    result.stdout.fnmatch_lines(["*its own failure*"])
    assert "test_slow_but_fails_anyway took" not in result.stdout.str()


def test_off_without_the_option(suite):
    pytester, _ = suite
    result = pytester.runpytest("-p", "no:cacheprovider")
    result.assert_outcomes(passed=4, failed=1)
    assert "fast-lane time budget" not in result.stdout.str()


def test_verdicts_cross_the_xdist_boundary(suite, monkeypatch):
    pytest.importorskip("xdist")
    pytester, allow = suite
    # A subprocess run does not inherit this session's sys.path entry for
    # python/tests, where _time_budget lives.
    monkeypatch.setenv("PYTHONPATH", str(Path(__file__).parent))
    result = pytester.runpytest_subprocess(
        "-p",
        "no:cacheprovider",
        "-n",
        "2",
        "--time-budget",
        "0.2",
        "--time-budget-allowlist",
        str(allow),
    )
    result.assert_outcomes(passed=3, failed=2)
    result.stdout.fnmatch_lines(
        ["*1 over budget, 2 allowlisted, 1 allowlisted ran under 0.08 s*", "*OVER*test_slow"]
    )


def test_rejects_a_nonpositive_budget(suite):
    pytester, allow = suite
    result = _run(pytester, allow)  # sanity: the fixture suite runs
    assert result.ret != 0
    bad = pytester.runpytest("-p", "no:cacheprovider", "--time-budget", "0")
    assert bad.ret != 0
    bad.stderr.fnmatch_lines(["*--time-budget must be positive*"])

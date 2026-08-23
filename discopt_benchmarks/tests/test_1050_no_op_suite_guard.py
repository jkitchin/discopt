"""The no-op suite guard must actually fire (#1050).

`discopt_benchmarks/tests/` shipped 118 tests that had never executed. They did
not fail — they skipped, every time, with the reason "discopt not yet
available", and a summary line cannot tell a skip from a pass. The guard added
to `conftest.py` exists so that cannot recur silently.

A guard is itself an instrument, and CLAUDE.md §6 is explicit that an
instrument which quietly measures nothing is worse than none: it reports a pass.
So these tests drive the real hook code — the project `conftest.py` is copied
into a temporary directory and pytest is run against synthetic modules in a
subprocess — and assert both arms: that the guard fires on the shapes it exists
to catch, and that it stays silent on a legitimate dependency skip. The second
arm is the one that matters for the guard's survival; a guard with false
positives gets deleted.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

_CONFTEST = Path(__file__).resolve().parent / "conftest.py"

# A skip that is legitimate: the dependency is genuinely absent.
_HONEST_SKIP = '''
import pytest

def test_needs_a_dependency_we_do_not_have():
    pytest.importorskip("a_module_that_does_not_exist_anywhere")
    assert False, "unreachable"
'''

# The exact shape #1050 was about: the harness was never written.
_STUB_SKIP = '''
import pytest

def test_pretends_to_check_something():
    pytest.skip("discopt not yet available")
'''

# A module that declares it must run, and then does not.
_MUST_EXECUTE_ALL_SKIPPED = '''
import pytest

MUST_EXECUTE = True

def test_one():
    pytest.skip("no instance present")

def test_two():
    pytest.skip("no instance present")
'''

# The same module with one test that really runs.
_MUST_EXECUTE_ONE_RUNS = '''
import pytest

MUST_EXECUTE = True

def test_one():
    pytest.skip("no instance present")

def test_two():
    assert 2 + 2 == 4
'''


def _run(tmp_path: Path, name: str, body: str, *extra: str) -> subprocess.CompletedProcess:
    shutil.copy(_CONFTEST, tmp_path / "conftest.py")
    (tmp_path / name).write_text(body)
    return subprocess.run(
        [sys.executable, "-m", "pytest", name, "-q", "-p", "no:cacheprovider", *extra],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.unit
def test_a_placeholder_skip_reason_fails_the_session(tmp_path):
    """Guard 1: the skip reason #1050 actually shipped."""
    result = _run(tmp_path, "test_stub.py", _STUB_SKIP)
    assert result.returncode != 0, (
        "the guard did not fire on a placeholder skip reason; "
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "NO-OP SUITE GUARD (#1050)" in result.stdout, result.stdout
    assert "1 skipped" in result.stdout, result.stdout


@pytest.mark.unit
def test_an_honest_dependency_skip_is_left_alone(tmp_path):
    """The control arm. A guard that fires on this would be turned off."""
    result = _run(tmp_path, "test_honest.py", _HONEST_SKIP)
    assert result.returncode == 0, (
        "the guard fired on a legitimate missing-dependency skip; "
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert "NO-OP SUITE GUARD" not in result.stdout, result.stdout
    assert "1 skipped" in result.stdout, result.stdout


@pytest.mark.unit
def test_a_must_execute_module_that_runs_nothing_fails(tmp_path):
    """Guard 2: survives a rewording of the skip reason."""
    result = _run(tmp_path, "test_must.py", _MUST_EXECUTE_ALL_SKIPPED)
    assert result.returncode != 0, (
        f"the guard did not fire on an all-skipped MUST_EXECUTE module; stdout:\n{result.stdout}"
    )
    assert "declares MUST_EXECUTE" in result.stdout, result.stdout


@pytest.mark.unit
def test_one_executed_test_satisfies_must_execute(tmp_path):
    """The module is doing its job; the guard must be silent."""
    result = _run(tmp_path, "test_must.py", _MUST_EXECUTE_ONE_RUNS)
    assert result.returncode == 0, (
        f"the guard fired on a module that executed a test; stdout:\n{result.stdout}"
    )
    assert "NO-OP SUITE GUARD" not in result.stdout, result.stdout


@pytest.mark.unit
def test_a_narrowing_filter_suppresses_the_must_execute_arm(tmp_path):
    """`-k` legitimately leaves only environment-gated tests; do not fire."""
    result = _run(tmp_path, "test_must.py", _MUST_EXECUTE_ONE_RUNS, "-k", "test_one")
    assert result.returncode == 0, (
        f"the guard fired on a deselecting -k filter; stdout:\n{result.stdout}"
    )
    assert "NO-OP SUITE GUARD" not in result.stdout, result.stdout


@pytest.mark.unit
def test_the_shipped_correctness_module_opts_in(tmp_path):
    """The module #1050 was filed about must carry the backstop."""
    from discopt_benchmarks.tests import test_correctness

    assert getattr(test_correctness, "MUST_EXECUTE", False) is True, (
        "discopt_benchmarks/tests/test_correctness.py must declare MUST_EXECUTE = True; "
        "it is the module whose 98 tests never ran (#1050)"
    )

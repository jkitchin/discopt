"""The compiled-extension load gate (CLAUDE.md §8).

Each test reconstructs a layout that actually occurred during #993, rather than a
hypothetical one. The gate exists because a panel can assert `discopt.__file__`
and two source markers, pass all three, and still be measuring a `_rust` build
from a different tree — the source-level checks cannot see a build output.

There is deliberately no "build is out of date" test, because the gate makes no
such claim: mtime answers that question wrongly in both directions (see the
module docstring), and the exact form needs a build-time stamp.
"""

from __future__ import annotations

import pytest

from benchmarks.load_gate import inspect_extension


def _tree(root, *, so_names=("_rust.cpython-312-darwin.so",)):
    """A repo-shaped tree: a `python/discopt/` package holding the given builds."""
    pkg = root / "python" / "discopt"
    pkg.mkdir(parents=True)
    built = []
    for name in so_names:
        so = pkg / name
        so.write_bytes(b"\x00")
        built.append(so)
    return pkg, built


def test_one_extension_beside_its_own_package_is_accepted(tmp_path):
    """The common case must stay silent, or the gate gets flagged off.

    This is the case the first version of this gate got wrong: it refused a
    correct tree because a checkout had bumped a source mtime without changing
    the content, which is exactly how a gate teaches people to override it.
    """
    pkg, built = _tree(tmp_path)
    report = inspect_extension(package_path=pkg, extension_path=built[0])
    assert report.hybrid is False
    assert report.reason() is None


def test_two_extensions_side_by_side_are_refused(tmp_path):
    """The #993 layout: `_rust.cpython-312-darwin.so` and a `-313-` sibling eleven
    weeks apart, where the interpreter silently picks the winner."""
    pkg, built = _tree(
        tmp_path,
        so_names=("_rust.cpython-312-darwin.so", "_rust.cpython-313-darwin.so"),
    )
    report = inspect_extension(package_path=pkg, extension_path=built[0])
    assert report.hybrid is False, "both sit beside the package; the ambiguity is the defect"
    assert len(report.siblings) == 1
    reason = report.reason()
    assert reason is not None
    assert "which interpreter starts" in reason


def test_a_worktree_with_no_extension_is_refused(tmp_path):
    """A worktree that was never built. Passing an explicit `None` is the caller
    asserting absence; `AUTO` would have gone and asked this interpreter."""
    pkg = tmp_path / "wt" / "python" / "discopt"
    pkg.mkdir(parents=True)
    report = inspect_extension(package_path=pkg, extension_path=None)
    reason = report.reason()
    assert reason is not None
    assert "no compiled `_rust` extension" in reason
    assert "hybrid" in reason


def test_rust_from_a_different_tree_than_python_is_refused(tmp_path):
    """The exact #993 hazard, detected directly rather than by proxy: `discopt`
    imports from an unbuilt worktree while `_rust` resolves to the main tree's
    build. Both `__file__` and every source marker pass; only the extension's
    directory betrays it."""
    _, built = _tree(tmp_path / "main")
    worktree_pkg = tmp_path / "wt993c" / "python" / "discopt"
    worktree_pkg.mkdir(parents=True)

    report = inspect_extension(package_path=worktree_pkg, extension_path=built[0])
    assert report.hybrid is True
    reason = report.reason()
    assert reason is not None
    assert "NOT the directory discopt itself was imported from" in reason


@pytest.mark.smoke
def test_it_inspects_the_live_interpreter_without_arguments():
    """The zero-argument form is what the runner calls; it must not raise, and it
    must report the extension this process actually imported."""
    pytest.importorskip("discopt")
    report = inspect_extension()
    assert report.package_path.name == "discopt"
    # Whatever the verdict here, the report must be *specific* -- a gate that
    # cannot name what it looked at cannot be acted on.
    assert report.extension_path is not None or report.reason() is not None

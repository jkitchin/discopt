"""The compiled-extension load gate (CLAUDE.md §8).

Each test reconstructs a layout that actually occurred during #993, rather than a
hypothetical one. The gate exists because a panel can assert `discopt.__file__`
and two source markers, pass all three, and still be measuring a `_rust` build
from a different tree — the source-level checks cannot see a build output.
"""

from __future__ import annotations

import os

import pytest

from benchmarks.load_gate import inspect_extension


def _tree(root, *, so_names=("_rust.cpython-312-darwin.so",), so_age=0.0, src_age=100.0):
    """A repo-shaped tree: `python/discopt/` beside `crates/`.

    `so_age`/`src_age` are seconds *before now*, so a larger `src_age` means the
    sources are older than the build, i.e. the build is current.
    """
    pkg = root / "python" / "discopt"
    pkg.mkdir(parents=True)
    crates = root / "crates" / "discopt-core" / "src"
    crates.mkdir(parents=True)

    src = crates / "lib.rs"
    src.write_text("// rust")
    now = 1_700_000_000.0
    os.utime(src, (now - src_age, now - src_age))

    built = []
    for name in so_names:
        so = pkg / name
        so.write_bytes(b"\x00")
        os.utime(so, (now - so_age, now - so_age))
        built.append(so)
    return pkg, built, src


def test_a_build_newer_than_its_sources_is_accepted(tmp_path):
    """The common case must stay silent, or the gate gets flagged off."""
    pkg, built, _ = _tree(tmp_path, so_age=0.0, src_age=100.0)
    report = inspect_extension(package_path=pkg, extension_path=built[0])
    assert report.stale is False
    assert report.reason() is None


def test_sources_newer_than_the_build_are_refused(tmp_path):
    """A checkout that changed Rust and was not rebuilt."""
    pkg, built, src = _tree(tmp_path, so_age=100.0, src_age=0.0)
    report = inspect_extension(package_path=pkg, extension_path=built[0])
    assert report.stale is True
    assert report.newest_source == src
    reason = report.reason()
    assert reason is not None
    assert "OLDER than the Rust sources" in reason
    assert "maturin develop" in reason, "a refusal must say what to do about it"


def test_two_extensions_side_by_side_are_refused(tmp_path):
    """The #993 layout: `_rust.cpython-312-darwin.so` and a `-313-` sibling eleven
    weeks apart, where the interpreter silently picks the winner."""
    pkg, built, _ = _tree(
        tmp_path,
        so_names=("_rust.cpython-312-darwin.so", "_rust.cpython-313-darwin.so"),
        so_age=0.0,
        src_age=100.0,
    )
    report = inspect_extension(package_path=pkg, extension_path=built[0])
    assert report.stale is False, "neither build is stale; the ambiguity is the defect"
    assert len(report.siblings) == 1
    reason = report.reason()
    assert reason is not None
    assert "which interpreter starts" in reason


def test_a_worktree_with_no_extension_is_refused(tmp_path):
    """A worktree that was never built. Passing an explicit `None` is the caller
    asserting absence; `AUTO` would have gone and asked this interpreter."""
    pkg = tmp_path / "wt" / "python" / "discopt"
    pkg.mkdir(parents=True)
    report = inspect_extension(package_path=pkg, extension_path=None, crates_dir=None)
    reason = report.reason()
    assert reason is not None
    assert "no compiled `_rust` extension" in reason
    assert "hybrid" in reason


def test_rust_from_a_different_tree_than_python_is_refused(tmp_path):
    """The exact #993 hazard, detected directly rather than by proxy: `discopt`
    imports from an unbuilt worktree while `_rust` resolves to the main tree's
    build. Both `__file__` and every source marker pass; only the extension's
    directory betrays it."""
    _, built, _ = _tree(tmp_path / "main", so_age=0.0, src_age=100.0)
    worktree_pkg = tmp_path / "wt993c" / "python" / "discopt"
    worktree_pkg.mkdir(parents=True)

    report = inspect_extension(package_path=worktree_pkg, extension_path=built[0], crates_dir=None)
    assert report.hybrid is True
    reason = report.reason()
    assert reason is not None
    assert "NOT the directory discopt itself was imported from" in reason


def test_documentation_churn_does_not_trip_the_gate(tmp_path):
    """A gate that fires on a touched README trains its reader to pass the flag."""
    pkg, built, _ = _tree(tmp_path, so_age=50.0, src_age=100.0)
    readme = tmp_path / "crates" / "discopt-core" / "README.md"
    readme.write_text("docs")
    os.utime(readme, (1_700_000_000.0, 1_700_000_000.0))  # newer than the build
    report = inspect_extension(package_path=pkg, extension_path=built[0])
    assert report.stale is False
    assert report.reason() is None


def test_run_suite_refuses_rather_than_measuring_a_hybrid(monkeypatch, tmp_path):
    """The gate must sit on the measurement, not on the CLI.

    The #993 panels called ``run_suite`` directly, so a check that only ran in
    ``main()`` would have missed exactly the callers that hit this.
    """
    gr = pytest.importorskip("benchmarks.gdplib_runner")
    from benchmarks.load_gate import ExtensionReport, StaleExtensionError

    hybrid = ExtensionReport(
        extension_path=tmp_path / "main" / "python" / "discopt" / "_rust.so",
        package_path=tmp_path / "wt" / "python" / "discopt",
        crates_dir=None,
        newest_source=None,
        stale=False,
        hybrid=True,
        siblings=(),
    )
    monkeypatch.setattr(gr, "is_available", lambda: True)
    monkeypatch.setattr(gr, "inspect_extension", lambda: hybrid)

    def _must_not_run(*a, **k):
        raise AssertionError("run_suite reached discover_models despite a hybrid build")

    monkeypatch.setattr(gr, "discover_models", _must_not_run)

    with pytest.raises(StaleExtensionError, match="NOT the directory"):
        gr.run_suite(gr.GDPLibSuiteConfig(include=["jobshop"]))

    # And the deliberate override still gets through, with the warning.
    monkeypatch.setattr(gr, "discover_models", lambda **k: [])
    gr.run_suite(gr.GDPLibSuiteConfig(include=["jobshop"], allow_stale_extension=True))


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

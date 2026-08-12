"""Assert that a benchmark run measures the code it claims to measure.

CLAUDE.md §8 asks every measurement to assert `module.__file__` and a marker
string unique to the version under test. A panel can satisfy both and still
measure something else, because `discopt` is two artifacts, not one: the Python
package, and a compiled `_rust` extension that is a *build output* and therefore
invisible to every source-level check. Switching a checkout, or running a panel
from a worktree that was never built, leaves a Python-new / Rust-old hybrid that
imports cleanly and reports numbers with nothing flagging them.

This was found during #993 rather than hypothesized. Two GDP panels ran with
`discopt` imported from a worktree and `_rust` silently resolving to a *different*
tree's `.so`, because the worktree had no compiled extension at all; and that tree
carried two extensions built eleven weeks apart — `_rust.cpython-312-darwin.so`
and `_rust.cpython-313-darwin.so` — so which stale artifact loaded was decided by
which interpreter happened to start.

It was first written up as harmless on the grounds that no Rust had changed. That
was wrong, and writing this gate is what found it: the loaded `.so` was built at
09:03, and commit `5dc804b9` (#928, expired MILP budgets) landed at 09:09 in the
panels' own base, changing `lp_bindings.rs` and `solver.py` *in the same commit*.
Both panels ran the new Python against the old Rust. The build was never stale by
much, and never stale in a way anything could see — which is the argument for
checking it mechanically rather than by recollection.

The check here is deliberately *mtime against sources*, not mtime against a
branch-switch time or a git hash. A checkout only rewrites files whose content
changed, so a switch across commits that touch no Rust legitimately leaves the
build current, and a hash comparison would cry stale on every such switch until
people learned to ignore it. An alarm that is usually wrong is worse than none:
it trains its reader to pass the flag.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

#: Source patterns whose edit invalidates a compiled extension. Kept narrow on
#: purpose: a stale `.rs` file changes solver behaviour, whereas a stale README
#: in `crates/` does not, and a gate that fires on documentation gets disabled.
_RUST_SOURCE_GLOBS = ("**/*.rs", "**/Cargo.toml", "**/build.rs")


class _Auto:
    """Distinguishes "work it out from this interpreter" from "there is none".

    `None` cannot carry both meanings, and conflating them makes the
    missing-extension arm unreachable except by running an interpreter that
    genuinely lacks the extension — i.e. untestable exactly where it matters.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<auto>"


AUTO = _Auto()


class StaleExtensionError(RuntimeError):
    """Raised instead of measuring a build the caller did not intend to measure.

    This is a refusal, not a warning (CLAUDE.md §3): the numbers a hybrid produces
    look entirely ordinary, so anything short of stopping gets scrolled past.
    """


@dataclass(frozen=True)
class ExtensionReport:
    """What was actually loaded, and whether it is current."""

    extension_path: Path | None
    package_path: Path
    crates_dir: Path | None
    newest_source: Path | None
    stale: bool
    hybrid: bool
    siblings: tuple[Path, ...]

    def reason(self) -> str | None:
        """A loud, actionable message, or None when the build is trustworthy."""
        if self.extension_path is None:
            return (
                f"discopt is imported from {self.package_path} but no compiled `_rust` "
                "extension was found next to it. If this run resolved `_rust` from a "
                "different tree, the panel is a Python-from-here / Rust-from-there "
                "hybrid and its numbers describe neither. Build in this tree "
                "(`maturin develop --release`) or run from the tree that was built."
            )
        problems = []
        if self.hybrid:
            problems.append(
                f"the loaded `_rust` extension lives in {self.extension_path.parent}, which "
                f"is NOT the directory discopt itself was imported from ({self.package_path}). "
                "This run is a Python-from-one-tree / Rust-from-another hybrid: it will "
                "import cleanly and report numbers that describe neither checkout. Build in "
                "the tree you are measuring (`maturin develop --release`)."
            )
        if self.stale and self.newest_source is not None:
            problems.append(
                f"the loaded extension {self.extension_path.name} is OLDER than the Rust "
                f"sources it was built from — {self.newest_source} was modified after it. "
                "Every number this run produces would describe a build that predates the "
                "checkout. Run `maturin develop --release`."
            )
        if self.siblings:
            names = ", ".join(sorted(p.name for p in self.siblings))
            problems.append(
                f"more than one compiled extension sits beside the package ({names} "
                f"alongside {self.extension_path.name}); which one loads is decided by "
                "which interpreter starts, not by anything this run asserts. Remove the "
                "ones you are not measuring."
            )
        return " Also: ".join(problems) if problems else None


def _newest_source(crates_dir: Path) -> Path | None:
    newest: Path | None = None
    newest_mtime = -1.0
    for pattern in _RUST_SOURCE_GLOBS:
        for path in crates_dir.glob(pattern):
            if not path.is_file():
                continue
            mtime = path.stat().st_mtime
            if mtime > newest_mtime:
                newest, newest_mtime = path, mtime
    return newest


def inspect_extension(
    package_path: Path | _Auto = AUTO,
    extension_path: Path | None | _Auto = AUTO,
    crates_dir: Path | None | _Auto = AUTO,
) -> ExtensionReport:
    """Describe the loaded `discopt` build.

    Every argument defaults to `AUTO`, meaning "work it out from this
    interpreter". Passing an explicit `None` asserts the thing is genuinely
    absent, which is how the missing-extension and no-crates layouts are tested.
    """
    if isinstance(package_path, _Auto):
        import discopt

        package_path = Path(discopt.__file__).resolve().parent
    if isinstance(extension_path, _Auto):
        rust = sys.modules.get("discopt._rust")
        if rust is None:
            try:
                import discopt._rust as rust  # type: ignore[no-redef]
            except ImportError:
                rust = None
        raw = getattr(rust, "__file__", None)
        extension_path = Path(raw).resolve() if raw else None
    if isinstance(crates_dir, _Auto):
        # `python/discopt/` -> repo root -> `crates/`. Anchored on the extension
        # when there is one, since that is the tree the build came from, which is
        # not necessarily the tree the Python came from — the whole failure mode.
        crates_dir = None
        anchor = extension_path or package_path
        for parent in anchor.parents:
            candidate = parent / "crates"
            if candidate.is_dir():
                crates_dir = candidate
                break

    siblings: tuple[Path, ...] = ()
    hybrid = False
    if extension_path is not None:
        siblings = tuple(
            p
            for p in sorted(extension_path.parent.glob("_rust*.so"))
            if p.resolve() != extension_path
        )
        hybrid = extension_path.parent != package_path

    newest = _newest_source(crates_dir) if crates_dir is not None else None
    stale = (
        extension_path is not None
        and newest is not None
        and newest.stat().st_mtime > extension_path.stat().st_mtime
    )
    return ExtensionReport(
        extension_path=extension_path,
        package_path=package_path,
        crates_dir=crates_dir,
        newest_source=newest,
        stale=stale,
        hybrid=hybrid,
        siblings=siblings,
    )

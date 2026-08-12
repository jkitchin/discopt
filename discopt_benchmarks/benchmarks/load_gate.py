"""Assert that a benchmark run measures the code it claims to measure.

CLAUDE.md §8 asks every measurement to assert `module.__file__` and a marker
string unique to the version under test. A panel can satisfy both and still
measure something else, because `discopt` is two artifacts, not one: the Python
package, and a compiled `_rust` extension that is a *build output* and therefore
invisible to every source-level check. Running a panel from a worktree that was
never built leaves the Python coming from one tree and the Rust from another; it
imports cleanly and reports numbers with nothing flagging them.

This was found during #993 rather than hypothesized. Two GDP panels ran with
`discopt` imported from a worktree and `_rust` silently resolving to a *different*
tree's `.so`, because the worktree had no compiled extension at all; and that tree
carried two extensions built eleven weeks apart — `_rust.cpython-312-darwin.so`
and `_rust.cpython-313-darwin.so` — so which stale artifact loaded was decided by
which interpreter happened to start.

Deliberately *not* checked here: whether the build is current with respect to its
sources. The obvious proxy is mtime, and mtime is wrong in both directions. A
`git checkout` rewrites every file that differs between the two commits, so a
switch that lands on identical content still bumps the mtime — the main tree
today has `lp_bindings.rs` at 09:39 and a 09:03 extension whose content matches
it exactly. And a build legitimately *predates* the commit it validates, because
the order of work is edit, build, test, then commit; reading that as staleness
inverts it. This module made both mistakes before it made none, and a gate that
refuses on a correct tree teaches its reader to pass `--allow-stale-extension`,
after which it protects nothing.

The durable answer is a build stamp: hash the Rust sources at build time, embed
the digest in the extension, and compare it against a hash of the tree's current
sources. That is exact, immune to checkout churn and to build-before-commit
ordering, and it answers the real question — *was this binary built from this
source* — rather than a proxy for it. It needs `build.rs` cooperation, so it is
tracked separately; the two structural checks below need no build support and are
correct as they stand.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


class _Auto:
    """Distinguishes "work it out from this interpreter" from "there is none".

    `None` cannot carry both meanings, and conflating them makes the
    missing-extension arm unreachable except by running an interpreter that
    genuinely lacks the extension — i.e. untestable exactly where it matters.
    """

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "<auto>"


AUTO = _Auto()


class ExtensionMismatchError(RuntimeError):
    """Raised instead of measuring an extension the caller did not intend to load.

    This is a refusal, not a warning (CLAUDE.md §3): the numbers a mismatched
    build produces look entirely ordinary, so anything short of stopping gets
    scrolled past.
    """


@dataclass(frozen=True)
class ExtensionReport:
    """Which `discopt` Python and which compiled `_rust` this process actually got."""

    extension_path: Path | None
    package_path: Path
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
        if self.siblings:
            names = ", ".join(sorted(p.name for p in self.siblings))
            problems.append(
                f"more than one compiled extension sits beside the package ({names} "
                f"alongside {self.extension_path.name}); which one loads is decided by "
                "which interpreter starts, not by anything this run asserts. Remove the "
                "ones you are not measuring."
            )
        return " Also: ".join(problems) if problems else None


def inspect_extension(
    package_path: Path | _Auto = AUTO,
    extension_path: Path | None | _Auto = AUTO,
) -> ExtensionReport:
    """Describe the loaded `discopt` build.

    Every argument defaults to `AUTO`, meaning "work it out from this
    interpreter". Passing an explicit `None` asserts the thing is genuinely
    absent, which is how the missing-extension layout is tested.
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
    siblings: tuple[Path, ...] = ()
    hybrid = False
    if extension_path is not None:
        siblings = tuple(
            p
            for p in sorted(extension_path.parent.glob("_rust*.so"))
            if p.resolve() != extension_path
        )
        hybrid = extension_path.parent != package_path

    return ExtensionReport(
        extension_path=extension_path,
        package_path=package_path,
        hybrid=hybrid,
        siblings=siblings,
    )

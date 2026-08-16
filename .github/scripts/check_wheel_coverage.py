#!/usr/bin/env python3
"""Assert the built distribution actually covers every platform we promise.

Run against the collected ``dist/`` directory *before* uploading to PyPI.

v0.8.0 shipped with ``requires-python = ">=3.10"`` while macOS and Windows had
wheels for 3.12 only (#1055): the release workflow's macOS/Windows jobs never
passed an interpreter list, so maturin built against the runner's ``setup-python``
version alone. Every job was green and the upload succeeded -- the gap was
invisible because nothing checked *what* got published, only that publishing
worked. A Mac user on 3.11 fell through to the sdist and hit a Rust toolchain
error that reads as a broken package.

The fix was ``abi3``: one wheel per platform, valid for the whole supported
range. This script is the guard that keeps it that way. It fails the release if
a platform is missing, if a wheel is not ``abi3``, or if the ``abi3`` floor has
drifted away from ``requires-python`` -- that last one is the silent failure the
next person would otherwise hit, because bumping ``requires-python`` without
bumping the ``abi3-pyXY`` feature in ``Cargo.toml`` publishes wheels that claim
support for versions they were never built against.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

# Platform tag -> a human name for the error message. Every one of these must be
# represented by at least one wheel in dist/.
REQUIRED_PLATFORMS: dict[str, str] = {
    r"manylinux.*_x86_64": "Linux x86_64",
    r"manylinux.*_aarch64": "Linux aarch64",
    r"macosx_.*_x86_64": "macOS x86_64",
    r"macosx_.*_arm64": "macOS arm64",
    r"win_amd64": "Windows x64",
}


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    dist = Path(sys.argv[1] if len(sys.argv) > 1 else "dist")

    with (repo / "pyproject.toml").open("rb") as fh:
        pyproject = tomllib.load(fh)
    requires_python = pyproject["project"]["requires-python"]
    m = re.fullmatch(r">=\s*3\.(\d+)", requires_python.strip())
    if not m:
        print(
            f"FAIL: cannot parse requires-python {requires_python!r}. This script "
            "understands '>=3.N' only; teach it the new form rather than deleting "
            "the check.",
            file=sys.stderr,
        )
        return 1
    floor_minor = int(m.group(1))
    expected_abi_tag = f"cp3{floor_minor}-abi3"

    wheels = sorted(p.name for p in dist.glob("*.whl"))
    sdists = sorted(p.name for p in dist.glob("*.tar.gz"))

    failures: list[str] = []
    checks = 0

    # 1. Something was actually built. Without this, an empty dist/ would sail
    #    through every loop below and report a clean pass.
    checks += 1
    if not wheels:
        failures.append(f"no wheels found in {dist}/")

    # 2. Exactly one sdist.
    checks += 1
    if len(sdists) != 1:
        failures.append(f"expected exactly 1 sdist, found {len(sdists)}: {sdists}")

    # 3. Every promised platform has a wheel.
    for pattern, label in REQUIRED_PLATFORMS.items():
        checks += 1
        if not any(re.search(pattern, w) for w in wheels):
            failures.append(f"no wheel for {label} (no filename matches /{pattern}/)")

    # 4. Every wheel is abi3 at exactly the requires-python floor.
    for w in wheels:
        checks += 1
        if expected_abi_tag not in w:
            failures.append(
                f"{w} is not tagged {expected_abi_tag}. requires-python is "
                f"{requires_python}, so the pyo3 'abi3-py3{floor_minor}' feature in "
                "Cargo.toml must match it."
            )

    print(f"dist:            {dist}")
    print(f"requires-python: {requires_python}  ->  expecting {expected_abi_tag}")
    print(f"sdists:          {len(sdists)}")
    print(f"wheels:          {len(wheels)}")
    for w in wheels:
        print(f"  {w}")
    print(f"EXECUTED CHECKS: {checks}")

    if checks == 0:
        print("FAIL: PROBE NEVER FIRED -- zero checks executed", file=sys.stderr)
        return 1
    if failures:
        print(f"\nFAIL: {len(failures)} coverage problem(s):", file=sys.stderr)
        for f in failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    print("\nOK: wheel coverage complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

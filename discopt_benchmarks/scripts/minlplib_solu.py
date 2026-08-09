"""MINLPLib ``minlplib.solu`` oracle reader.

``python/tests/_optima.py`` records optima for 27 curated instances. Scoring a
corpus panel against it silently degrades to "no oracle" for everything else --
the #966 coupled panel had oracle coverage on **1 of 19** instances and still
printed ``unsound: []``, which reads as a soundness result and was not one.

``minlplib.solu`` (from the benchmark snapshot named in CLAUDE.md) covers the
whole library. Entries:

    =opt=      name  v   proven optimum
    =best=     name  v   best known primal  (>= optimum for min)
    =bestdual= name  v   best known dual    (<= optimum for min)
    =inf=      name      proven infeasible
    =unkn=     name      nothing known

For a MINIMIZATION instance a valid dual bound must satisfy ``bound <= optimum``,
and ``optimum <= best``, so ``bound > best`` is unsound. ``=opt=`` is used when
present, ``=best=`` otherwise; instances with neither yield ``None`` and MUST be
reported as uncovered rather than counted as clean.
"""

from __future__ import annotations

import os
from pathlib import Path

_DEFAULT = Path(
    os.environ.get(
        "DISCOPT_MINLPLIB_SOLU",
        Path.home() / "Dropbox/projects/discopt-minlp-benchmark/minlplib.solu",
    )
)


def load(path: Path | None = None) -> dict[str, dict[str, float | str]]:
    """Parse the .solu file into ``{name: {tag: value}}``. Raises if unreadable."""
    p = Path(path) if path is not None else _DEFAULT
    if not p.exists():
        raise FileNotFoundError(
            f"minlplib.solu not found at {p}; set DISCOPT_MINLPLIB_SOLU to override"
        )
    out: dict[str, dict[str, float | str]] = {}
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) < 2 or not parts[0].startswith("="):
            continue
        tag, name = parts[0].strip("="), parts[1]
        rec = out.setdefault(name, {})
        rec[tag] = float(parts[2]) if len(parts) >= 3 else True
    return out


def primal_ceiling(name: str, table: dict) -> float | None:
    """Largest value a valid MIN dual bound may take, or ``None`` if unknown.

    ``=opt=`` when proven; otherwise ``=best=``, the best known primal, which is
    an upper bound on the true optimum and therefore still a valid ceiling.
    """
    rec = table.get(name)
    if not rec:
        return None
    for tag in ("opt", "best"):
        v = rec.get(tag)
        if isinstance(v, float):
            return v
    return None

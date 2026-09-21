#!/usr/bin/env python
"""#1401: no solve path may report a sentinel-derived value as a dual bound.

The LP layer's "no bound" is the sentinel ``1e20``, not ``inf``. It survives
arithmetic as an ordinary finite number, so a gate written as ``np.isfinite(v)``
accepts it and reports "the relaxation proved nothing" as a proved bound. The
spatial B&B exit took ``stats["global_lower_bound"]`` behind ``np.isfinite``
alone, bypassing ``_finalize_reported_bound``'s ``1e19`` refusal.

Measured on ``QPLIB_2967`` (MAXIMIZE, ``time_limit=20``) before the fix:
``bound=4.4495549999999977e+21``, ``gap=4.07e20``, ``gap_certified=False`` --
445x past the 1e19 refusal point and two orders past the LP layer's own
``INF``.

This probe is fail-closed and counts its assertions (CLAUDE.md §6): a row it
cannot solve is a FAILURE, not a skip, because "we never reached the check" is
exactly how this class of instrument degrades to a no-op that prints "0
violations".

Run BOTH arms. The fix arm must exit 0, the baseline arm must exit 1; a probe
that passes on both is measuring nothing.

    PYTHONPATH=<worktree>/python python -u scripts/audit_1401_sentinel_bound_reported.py
    PYTHONPATH=<main>/python     python -u scripts/audit_1401_sentinel_bound_reported.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import discopt
import numpy as np
from discopt import modeling as dm
from discopt.interfaces.qplib import from_qplib

#: One order below ``CONSTRAINT_INF`` (1e20). NOT ``SENTINEL_THRESHOLD`` (1e29),
#: which answers "is this an infeasibility marker?" -- see solver._EFF_INF_BOUND.
EFF_INF = 1e19

QPLIB_DIR = Path.home() / "Dropbox/projects/discopt-minlp-benchmark/qplib/qplib"


def _load_gate() -> bool:
    """§8: say which code is loaded and whether the fix marker is present."""
    import discopt.solver as sv

    print(f"discopt     : {discopt.__file__}")
    print(f"solver      : {sv.__file__}")
    marker = hasattr(sv, "_bound_is_usable") and getattr(sv, "_EFF_INF_BOUND", None) == EFF_INF
    print(f"fix marker  : {'PRESENT' if marker else 'ABSENT'} (_bound_is_usable + _EFF_INF_BOUND)")
    try:
        print(f"_rust       : {discopt._rust.__file__}")
    except Exception as e:  # noqa: BLE001 -- reported, never swallowed
        print(f"_rust       : UNAVAILABLE ({type(e).__name__}: {e})")
    return marker


def _sentinel_box_model(bound: float, maximize: bool):
    """A bilinear term over a box declared out to *bound*.

    The synthetic route to a sentinel-magnitude relaxation value: the McCormick
    envelope of ``x*y`` over ``[0, bound]^2`` has corner value ``bound**2``, so
    for ``bound=1e10`` the relaxation carries ``1e20`` with no user-visible
    infinity anywhere in the model.
    """
    m = dm.Model(f"sentinel_box_{'max' if maximize else 'min'}_{bound:g}")
    x = m.continuous("x", lb=0.0, ub=bound)
    y = m.continuous("y", lb=0.0, ub=bound)
    z = m.continuous("z", lb=0.0, ub=bound)
    m.subject_to(x * y - z <= 1.0)
    m.subject_to(x + y >= 1.0)
    if maximize:
        m.maximize(z - x)
    else:
        m.minimize(x - z)
    return m


def _rows():
    rows = []
    for b in (1e8, 1e10, 1e12):
        for mx in (True, False):
            tag = f"sentinel_box[{b:g},{'max' if mx else 'min'}]"
            rows.append((tag, _sentinel_box_model, (b, mx)))
    rows.append(("QPLIB_2967", None, None))
    return rows


def main() -> int:
    marker = _load_gate()
    print()

    checked = 0
    bad: list[str] = []

    for name, factory, args in _rows():
        if factory is None:
            path = QPLIB_DIR / f"{name}.qplib"
            if not path.exists():
                bad.append(f"{name}: corpus file missing at {path} -- row NOT checked")
                continue
            model = from_qplib(str(path))
        else:
            model = factory(*args)

        res = model.solve(time_limit=20.0)
        b = res.bound
        cert = bool(getattr(res, "gap_certified", False))
        finite = b is not None and bool(np.isfinite(b))
        offending = finite and abs(float(b)) >= EFF_INF
        checked += 1
        flag = "SENTINEL-AS-BOUND" if offending else "ok"
        print(
            f"  {name:<34} bound={b!r:>26} certified={cert!s:<5} obj={res.objective!r:>22}  {flag}"
        )
        if offending:
            bad.append(
                f"{name}: reported bound={b!r} -- finite but |bound| >= {EFF_INF:g}, "
                f"{abs(float(b)) / EFF_INF:.1f}x past the effective-infinity refusal "
                f"(gap_certified={cert})"
            )

    print()
    print(f"EXECUTED ASSERTIONS: {checked}")
    if checked == 0:
        print("FAIL: the probe checked nothing -- it proves nothing about #1401")
        return 1
    if bad:
        print(f"FAIL: {len(bad)} of {checked} rows reported a sentinel-derived bound as finite")
        for b in bad:
            print(f"  - {b}")
        return 1
    print(f"PASS: {checked}/{checked} rows reported either no bound or a real one")
    if not marker:
        print(
            "WARNING: the fix marker was ABSENT yet every row passed -- on the baseline "
            "arm this probe is expected to FAIL, so it is not discriminating"
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

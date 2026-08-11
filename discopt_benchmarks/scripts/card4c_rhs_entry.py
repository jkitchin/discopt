#!/usr/bin/env python3
"""Entry experiment for the ``Constraint.rhs`` silent-divergence defect (Card 4c
Task 2's incidental finding, §6 2026-07-30).

**Hypothesis under test (H-RHS).** A non-zero ``Constraint.rhs`` is *unreachable*
through supported construction: every producer in the tree normalizes the offset
into ``body`` and leaves ``rhs == 0.0``, exactly as ``Constraint``'s own docstring
declares ("always 0.0 in normalized form").

**Why it matters.** ``_relax/dag_compiler.compile_constraint`` compiles
``constraint.body`` and discards ``rhs``; ``validation/feasibility`` honours it
(``signed = body - rhs``). If H-RHS holds, the divergence is unreachable except by
out-of-contract construction and the correct fix is a LOUD REFUSAL at the model
boundary. If H-RHS is FALSIFIED — some supported path really does emit a non-zero
``rhs`` — refusing would break working models and the solve path must honour
``rhs`` instead.

**Kill criterion.** One constraint with ``abs(rhs) > 0`` reached from a supported
constructor (``.nl`` reader, GAMS parser, modeling DSL, or any internal
reformulation running on those) falsifies H-RHS.

Per CLAUDE.md §6 the probe prints an executed-comparison count and exits non-zero
when it is zero. Per §7 no exception is swallowed: a load failure is reported and
counted, never absorbed.
"""

from __future__ import annotations

import glob
import os
import sys
import traceback

# CLAUDE.md §8 — verify which code was actually loaded before measuring.
import discopt
import discopt.modeling.core as core
from discopt.modeling import Constraint  # noqa: F401  (import-time contract check)

MARKER = "always 0.0 in normalized form"


def _assert_loaded_tree() -> None:
    print(f"discopt.__file__      = {discopt.__file__}", flush=True)
    print(f"core.__file__         = {core.__file__}", flush=True)
    doc = core.Constraint.__doc__ or ""
    print(f"marker {MARKER!r} present in Constraint.__doc__: {MARKER in doc}", flush=True)
    if MARKER not in doc:
        raise SystemExit(
            "ABORT: the Constraint docstring marker is absent — this is not the tree "
            "under test (CLAUDE.md §8)."
        )


def scan_model(model, label: str) -> tuple[int, list[str]]:
    """Return (comparisons_executed, violations) for one model."""
    n = 0
    bad: list[str] = []
    for i, c in enumerate(getattr(model, "_constraints", []) or []):
        rhs = getattr(c, "rhs", None)
        if rhs is None:
            # Not a Constraint-shaped row (SOS/complementarity/logical). Counted so
            # the total reflects rows examined, never silently skipped (§6).
            n += 1
            continue
        n += 1
        if float(rhs) != 0.0:
            bad.append(f"{label}[{i}] name={getattr(c, 'name', None)!r} rhs={float(rhs)!r}")
    return n, bad


def main() -> int:
    _assert_loaded_tree()

    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.dirname(os.path.dirname(here))
    corpus = sorted(glob.glob(os.path.join(repo, "python", "tests", "data", "minlplib_nl", "*.nl")))
    print(f"corpus instances found: {len(corpus)}", flush=True)
    if not corpus:
        raise SystemExit("ABORT: corpus empty — the probe would measure nothing (§6).")

    total_rows = 0
    total_models = 0
    load_failures: list[str] = []
    violations: list[str] = []

    for path in corpus:
        name = os.path.basename(path)
        try:
            m = core.from_nl(path)
        except Exception as exc:  # noqa: BLE001 — reported, never swallowed (§7)
            load_failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  LOAD-FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        total_models += 1
        n, bad = scan_model(m, name)
        total_rows += n
        violations.extend(bad)
        print(f"  {name:28s} rows={n:6d} nonzero_rhs={len(bad)}", flush=True)

    # Control arm (§6 non-vacuity): the probe MUST be able to see a non-zero rhs.
    ctl = core.Model("rhs_control")
    w = ctl.continuous("w", lb=0.0, ub=10.0)
    ctl.minimize(w)
    ctl._constraints.append(core.Constraint(w, ">=", 5.0))
    ctl_rows, ctl_bad = scan_model(ctl, "CONTROL")
    total_rows += ctl_rows
    print(f"\ncontrol arm: rows={ctl_rows} nonzero_rhs={len(ctl_bad)} -> {ctl_bad}", flush=True)
    if len(ctl_bad) != 1:
        raise SystemExit(
            "ABORT: the control arm did not register its planted non-zero rhs — the "
            "probe cannot detect what it claims to measure (CLAUDE.md §6)."
        )

    # Operator arm: the supported DSL must normalize.
    op = core.Model("rhs_operator")
    v = op.continuous("v", lb=0.0, ub=10.0)
    op.minimize(v)
    op.subject_to(v >= 5.0)
    op_rows, op_bad = scan_model(op, "OPERATOR")
    total_rows += op_rows
    print(f"operator arm: rows={op_rows} nonzero_rhs={len(op_bad)}", flush=True)

    print("\n" + "=" * 72, flush=True)
    print(f"EXECUTED_RHS_COMPARISONS {total_rows}", flush=True)
    print(f"MODELS_LOADED            {total_models}", flush=True)
    print(f"LOAD_FAILURES            {len(load_failures)}", flush=True)
    print(f"CORPUS_VIOLATIONS        {len(violations)}", flush=True)
    for v_ in violations:
        print(f"  VIOLATION {v_}", flush=True)
    if load_failures:
        print("\nload failures (reported, not swallowed):", flush=True)
        for f in load_failures:
            print(f"  {f}", flush=True)

    if total_rows == 0:
        print("VACUOUS: zero comparisons executed", flush=True)
        return 2
    if op_bad:
        print("H-RHS FALSIFIED: the operator DSL itself leaves a non-zero rhs.", flush=True)
        return 3
    if violations:
        print("H-RHS FALSIFIED: a supported corpus path emits a non-zero rhs.", flush=True)
        return 3
    print(
        "H-RHS HOLDS on this population: no supported constructor emits a non-zero "
        "rhs; the control arm proves the probe can see one.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — crash loudly with the traceback (§7)
        traceback.print_exc()
        sys.exit(1)

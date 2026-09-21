#!/usr/bin/env python
"""#1409 — CLAUDE.md §5 differential panel for the reduced-cost-fixing divisor bound.

``bnb::milp_driver::reduced_cost_fix`` divides the gap by ``|d_j|`` where
``d_j = c_j − A_jᵀy`` is a *cancelling* difference. Over-stating ``|d_j|`` shrinks
``⌊gap/|d_j|⌋`` and can land it a whole integer low, writing an upper bound that
excludes an improving point. The fix deflates the divisor toward zero by
``gamma(nnz+2) · S_j`` (``S_j = |c_j| + Σ|a_ij y_i|``) before dividing, and skips a
column whose sign does not survive that deflation.

This is **bound-changing on a default-ON path**, so the regime is §5's differential
panel, not bound-neutral parity: node counts are expected to move. The fix can only
make fixing *weaker*, so the prior is "same or slightly more nodes, never a worse
bound".

**Why two builds rather than a flag.** The change is an unconditional correctness
fix with no env switch — deliberately, since CLAUDE.md §5's flag-retirement clause
says a soundness fix on a default-ON path is not a graduation candidate. The arms
are therefore two *builds*: this repo's branch (marker present) and the baseline
commit (marker absent), each asserted by ``--arm`` at startup (§8).

**What this panel does and does not claim.** It claims the deterministic quantities
— status, bound, incumbent, certification, node count — which ``deterministic=True``
makes reproducible (verify with ``--reps``). It does **not** claim wall-clock: the
arms are separate processes and cannot be interleaved within an instance, which is
exactly the control §9 requires for a timing claim. Wall is recorded for context and
must not be quoted as a speed result.

Usage::

    python -u .../issue1409_rc_fix_divisor_panel.py --arm fix  --out fix.json
    python -u .../issue1409_rc_fix_divisor_panel.py --arm base --out base.json
    python -u .../issue1409_rc_fix_divisor_panel.py --compare fix.json base.json
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import discopt  # noqa: E402

REPO = pathlib.Path(__file__).resolve().parents[2]
CORPUS = REPO / "python" / "tests" / "data" / "minlplib_nl"
ORACLE = REPO / "python" / "tests" / "data" / "known_optima.toml"

REL_TOL = 1e-4
ABS_TOL = 1e-6


def _load_oracle() -> dict[str, float]:
    import tomllib

    data = tomllib.loads(ORACLE.read_text())
    return {
        k: float(v["optimum"]) for k, v in data.items() if isinstance(v, dict) and "optimum" in v
    }


def _sense_is_min(model) -> bool:
    """True for a MINIMIZE objective. Read from the model, never assumed.

    The bound/incumbent invariant flips with the sense, so guessing it turns the
    soundness check into noise.
    """
    obj = model._objective
    return "MINIMIZE" in str(obj.sense).upper()


def _assert_arm(arm: str) -> None:
    """§8: assert *which* build is loaded, by a marker unique to the fix.

    The marker is the ``reduced_cost_errors`` key the #1409 binding requires. It
    lives in the same extension module as the ``reduced_cost_fix`` change, so its
    presence pins the Rust build and not merely the Python tree. A baseline run
    asserts the marker **absent**, which is the half that catches "I measured the
    same build twice".
    """
    print(f"marker: discopt.__file__ = {discopt.__file__}", flush=True)
    import discopt._rust as _rust

    print(f"marker: discopt._rust.__file__ = {_rust.__file__}", flush=True)
    doc = _rust.PyModelRepr.presolve.__doc__ or ""
    present = "reduced_cost_errors" in doc
    if arm == "fix" and not present:
        raise SystemExit(
            "MARKER ABSENT on the 'fix' arm: this extension predates #1409. "
            "Rebuild the worktree before measuring."
        )
    if arm == "base" and present:
        raise SystemExit(
            "MARKER PRESENT on the 'base' arm: this is the #1409 build, not the "
            "baseline. Both arms would be the same experiment."
        )
    print(f"marker: #1409 divisor bound {'PRESENT' if present else 'ABSENT'} (arm={arm})")


def _solve(path: pathlib.Path, time_limit: float) -> dict:
    """One instance. An exception is recorded as a cell, never swallowed (§7)."""
    from discopt.modeling.core import from_nl

    model = from_nl(str(path))
    is_min = _sense_is_min(model)
    t0 = time.perf_counter()
    try:
        r = model.solve(time_limit=time_limit, deterministic=True)
        cell = {
            "status": str(r.status),
            "objective": r.objective,
            "bound": r.bound,
            "gap_certified": bool(getattr(r, "gap_certified", False)),
            "node_count": getattr(r, "node_count", None),
        }
    except Exception as exc:  # noqa: BLE001 -- recorded, never hidden (§7)
        cell = {
            "status": "ERROR",
            "error": f"{type(exc).__name__}: {exc}"[:200],
            "objective": None,
            "bound": None,
            "gap_certified": False,
            "node_count": None,
        }
    cell["wall"] = time.perf_counter() - t0
    cell["is_min"] = is_min
    return cell


def _invariant_violations(cell: dict, optimum: float | None) -> list[str]:
    """Sense-correct soundness checks on a single arm."""
    bad: list[str] = []
    obj, bnd, is_min = cell["objective"], cell["bound"], cell["is_min"]
    slack = ABS_TOL + REL_TOL * max(1.0, abs(obj) if obj is not None else 1.0)
    if obj is not None and bnd is not None:
        if is_min and bnd > obj + slack:
            bad.append(f"bound {bnd!r} above incumbent {obj!r} (min)")
        if not is_min and bnd < obj - slack:
            bad.append(f"bound {bnd!r} below incumbent {obj!r} (max)")
    if bnd is not None and optimum is not None:
        oslack = ABS_TOL + REL_TOL * max(1.0, abs(optimum))
        if is_min and bnd > optimum + oslack:
            bad.append(f"bound {bnd!r} above reference optimum {optimum!r} (min)")
        if not is_min and bnd < optimum - oslack:
            bad.append(f"bound {bnd!r} below reference optimum {optimum!r} (max)")
    return bad


def run_arm(args: argparse.Namespace) -> int:
    _assert_arm(args.arm)
    oracle = _load_oracle()
    names = sorted(p.stem for p in CORPUS.glob("*.nl"))
    if args.only:
        wanted = {n.strip() for n in args.only.split(",") if n.strip()}
        names = [n for n in names if n in wanted]
    if not names:
        print("no instances selected", file=sys.stderr)
        return 2
    print(
        f"corpus: {len(names)} instances, time_limit={args.time_limit}s, "
        f"reps={args.reps}, oracle-backed: {sum(n in oracle for n in names)}",
        flush=True,
    )

    solved = 0
    rows: dict[str, dict] = {}
    for idx, name in enumerate(names, 1):
        reps = [_solve(CORPUS / f"{name}.nl", args.time_limit) for _ in range(args.reps)]
        cell = reps[0]
        # Determinism check: with deterministic=True the deterministic quantities
        # must repeat. A cell that does not is flagged so the comparison can refuse
        # to read a difference on it as a result.
        keys = ("status", "bound", "node_count", "gap_certified")
        want = tuple(cell[k] for k in keys)
        cell["stable"] = all(tuple(r[k] for k in keys) == want for r in reps)
        rows[name] = cell
        solved += 1
        # §10: per-item progress, so a long run is never mistaken for a dead one.
        print(
            f"  [{idx}/{len(names)}] {name:<20} status={cell['status']:<12} "
            f"bound={cell['bound']!r} nodes={cell['node_count']!r} "
            f"cert={cell['gap_certified']} stable={cell['stable']} "
            f"wall={cell['wall']:.2f}s",
            flush=True,
        )

    # §6: prove the probe fired.
    print(f"\nEXECUTED SOLVES: {solved}")
    if solved == 0:
        print("PROBE FIRED ZERO TIMES -- no result", file=sys.stderr)
        return 3

    payload = {
        "arm": args.arm,
        "time_limit": args.time_limit,
        "reps": args.reps,
        "discopt": discopt.__file__,
        "rows": rows,
    }
    if args.out:
        pathlib.Path(args.out).write_text(json.dumps(payload, indent=2))
        print(f"wrote {args.out}")
    return 0


def compare(fix_path: str, base_path: str) -> int:
    fix = json.loads(pathlib.Path(fix_path).read_text())
    base = json.loads(pathlib.Path(base_path).read_text())
    if fix["arm"] != "fix" or base["arm"] != "base":
        print(f"arm mismatch: {fix['arm']!r} / {base['arm']!r}", file=sys.stderr)
        return 2
    oracle = _load_oracle()
    names = sorted(set(fix["rows"]) & set(base["rows"]))

    comparisons = 0
    incorrect: list[str] = []
    cert_regressions: list[str] = []
    obj_drift: list[str] = []
    unstable: list[str] = []
    node_delta: list[tuple[str, int, int]] = []
    bound_worse: list[str] = []

    for name in names:
        f, b = fix["rows"][name], base["rows"][name]
        opt = oracle.get(name)
        comparisons += 1

        for arm_name, cell in (("fix", f), ("base", b)):
            incorrect += [f"{name} [{arm_name}]: {m}" for m in _invariant_violations(cell, opt)]

        if not (f.get("stable", True) and b.get("stable", True)):
            unstable.append(name)
            continue

        # Bar 1 -- cert-clean.
        if b["gap_certified"] and not f["gap_certified"]:
            cert_regressions.append(name)
        fo, bo = f["objective"], b["objective"]
        drifted = (
            fo is not None
            and bo is not None
            and abs(fo - bo) > ABS_TOL + REL_TOL * max(1.0, abs(bo))
        )
        if drifted:
            obj_drift.append(f"{name}: fix={fo!r} base={bo!r}")
        # The fix can only weaken fixing, so a *worse* (looser) bound is expected
        # and allowed; a bound that moved the unsound way is not.
        fb, bb, is_min = f["bound"], b["bound"], f["is_min"]
        if fb is not None and bb is not None and opt is not None:
            oslack = ABS_TOL + REL_TOL * max(1.0, abs(opt))
            if is_min and fb > opt + oslack:
                bound_worse.append(f"{name}: fix bound {fb!r} past optimum {opt!r}")
            if not is_min and fb < opt - oslack:
                bound_worse.append(f"{name}: fix bound {fb!r} past optimum {opt!r}")

        fn_, bn_ = f["node_count"], b["node_count"]
        if fn_ is not None and bn_ is not None and fn_ != bn_:
            node_delta.append((name, bn_, fn_))

    print(f"\ncompared instances: {comparisons}")
    if comparisons == 0:
        print("PROBE FIRED ZERO TIMES -- no result", file=sys.stderr)
        return 3

    print(f"unstable (excluded from the differential): {len(unstable)} {unstable}")
    print(f"incorrect_count: {len(incorrect)}")
    for m in incorrect:
        print(f"  ! {m}")
    print(f"certification regressions: {len(cert_regressions)} {cert_regressions}")
    print(f"bound moved past a reference optimum: {len(bound_worse)}")
    for m in bound_worse:
        print(f"  ! {m}")
    print(f"objective drift beyond rel=1e-4: {len(obj_drift)}")
    for m in obj_drift:
        print(f"  ! {m}")

    print(f"\nnode-count differences: {len(node_delta)}")
    for name, bn, fn in node_delta:
        print(f"  {name:<20} base={bn:<8} fix={fn:<8} delta={fn - bn:+}")
    if node_delta:
        tot_b = sum(bn for _, bn, _ in node_delta)
        tot_f = sum(fn for _, _, fn in node_delta)
        print(
            f"  totals over differing instances: base={tot_b} fix={tot_f} delta={tot_f - tot_b:+}"
        )

    clean = not (incorrect or cert_regressions or obj_drift or bound_worse)
    print(f"\nBAR 1 cert-clean: {'PASS' if clean else 'FAIL'}")
    print(
        "BAR 2 net-positive: not applicable -- this is an unconditional soundness "
        "fix, not a flag graduation. The fix can only weaken reduced-cost fixing, so "
        "the bar it must clear is 'cert-clean and not broadly harmful', reported as "
        "the node-count table above."
    )
    return 0 if clean else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=("fix", "base"), default=None)
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--compare", nargs=2, metavar=("FIX_JSON", "BASE_JSON"), default=None)
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if not args.arm:
        ap.error("one of --arm or --compare is required")
    return run_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())

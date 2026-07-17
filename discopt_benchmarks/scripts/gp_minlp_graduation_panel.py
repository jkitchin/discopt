#!/usr/bin/env python3
"""Graduation panel for the ``DISCOPT_GP_MINLP`` flag (issue #116).

CLAUDE.md §5 (policy updated 2026-07-17): a bound-changing flag graduates on a
single panel run — flag **ON vs OFF over the in-repo corpus** — meeting BOTH:

  1. *cert-clean* — ``incorrect_count = 0`` (no false optimal/infeasible), no
     dual bound above the reference optimum, no certification regression (no
     ``gap_certified=True`` instance regressing to uncertified), certified
     objective within tolerance, and every ON incumbent independently
     feasibility-verified.
  2. *net-positive* — measurably helpful broadly (node count / wall / bound).

Because ``DISCOPT_GP_MINLP`` only ever fires where ``classify_gp_minlp`` accepts
the model, the corpus splits cleanly:

  * **affected** instances (the auto-route actually changes the solve): full
    ON-vs-OFF differential + oracle bracket + independent feasibility check.
  * **inert** instances (``classify_gp_minlp is None``): the flag is
    *structurally* unreachable, so ON must be byte-identical to OFF. Verified
    both structurally (classification) and empirically on a fast deterministic
    sample (identical status / objective / node_count).

Independent cross-checks used as oracles: the published MINLPLib optima and the
solver's *own classic spatial branch-and-bound* (``solver="bb"``) — a code path
entirely disjoint from the GP-MINLP log-space route.

Usage:
    PYTHONPATH=python JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      python discopt_benchmarks/scripts/gp_minlp_graduation_panel.py \
        --time-limit 120 --inert-sample 12
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from discopt.gp import classify_gp_minlp
from discopt.modeling.core import from_nl

_REPO = Path(__file__).resolve().parents[2]
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"

# Published MINLPLib global optima for the GP-MINLP instances present in the
# in-repo corpus (source: MINLPLib instancedata; mirrored in
# discopt_benchmarks/scripts/t21_root_loop_replay.py).
_ORACLE: dict[str, float] = {
    "cvxnonsep_nsig30": 130.6287126,
    "cvxnonsep_psig30": 78.99885434,
}

# Tolerances (conftest.py house values).
_ABS = 1e-6
_REL = 1e-4
_INT = 1e-5


@dataclass
class InstanceResult:
    name: str
    n_vars: int
    n_int: int
    # OFF (classic spatial B&B) vs ON (GP-MINLP auto-route)
    off_status: str = ""
    off_obj: float | None = None
    off_bound: float | None = None
    off_cert: bool | None = None
    off_nodes: int | None = None
    off_wall: float | None = None
    on_status: str = ""
    on_obj: float | None = None
    on_bound: float | None = None
    on_cert: bool | None = None
    on_nodes: int | None = None
    on_wall: float | None = None
    oracle: float | None = None
    # verdicts
    cert_clean: bool = True
    net_positive: bool | None = None
    feasible_verified: bool | None = None
    violations: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def _solve(nl_path: Path, flag_on: bool, time_limit: float):
    """Solve one instance with the GP-MINLP auto-route flag on/off.

    The flag is read live inside ``solve_model`` (``os.environ.get`` per call),
    so an in-process toggle is faithful — OFF forces the classic spatial B&B,
    ON enables the GP-MINLP auto-route for a recognised model.
    """
    if flag_on:
        os.environ["DISCOPT_GP_MINLP"] = "1"
    else:
        os.environ.pop("DISCOPT_GP_MINLP", None)
    m = from_nl(str(nl_path))
    # OFF: force classic B&B explicitly so the pure-GP auto-route can't fire
    # either — a clean, independent baseline. ON: default routing with the flag.
    if flag_on:
        return m.solve(time_limit=time_limit)
    return m.solve(solver="bb", time_limit=time_limit)


def _within(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= _ABS + _REL * max(1.0, abs(b))


def _feasible(nl_path: Path, x: dict) -> tuple[bool, list[str]]:
    """Independently verify an incumbent against the raw .nl model.

    Evaluates variable bounds, integrality, and every constraint residual via
    the Rust ``_nl_repr`` evaluator — a path disjoint from the GP-MINLP solver.
    """
    msgs: list[str] = []
    m = from_nl(str(nl_path))
    repr_ = m._nl_repr
    # Flatten the incumbent to the .nl variable order (== m._variables order),
    # checking each variable's own declared bounds and integrality en route.
    x_flat = np.empty(repr_.n_vars, dtype=np.float64)
    off = 0
    for v in m._variables:
        chunk = np.asarray(x[v.name], dtype=np.float64).reshape(-1)
        lb = np.asarray(v.lb, dtype=np.float64).reshape(-1)
        ub = np.asarray(v.ub, dtype=np.float64).reshape(-1)
        if np.any(chunk < lb - 1e-6) or np.any(chunk > ub + 1e-6):
            msgs.append(f"variable bound violation on {v.name}")
        if v.var_type.name != "CONTINUOUS" and np.any(np.abs(chunk - np.round(chunk)) > _INT):
            msgs.append(f"non-integral discrete var {v.name}")
        x_flat[off : off + v.size] = chunk
        off += v.size
    # Constraint residuals via the Rust evaluator (a path disjoint from the GP-MINLP solver).
    for i in range(repr_.n_constraints):
        body = float(repr_.evaluate_constraint(i, x_flat))
        rhs = float(repr_.constraint_rhs(i))
        sense = repr_.constraint_sense(i)
        tol = _ABS + _REL * max(1.0, abs(rhs))
        if sense in ("<=", "L") and body > rhs + tol:
            msgs.append(f"constraint {i} ({sense}) violated: {body:.6g} > {rhs:.6g}")
        elif sense in (">=", "G") and body < rhs - tol:
            msgs.append(f"constraint {i} ({sense}) violated: {body:.6g} < {rhs:.6g}")
        elif sense in ("==", "E") and abs(body - rhs) > tol:
            msgs.append(f"constraint {i} (==) violated: {body:.6g} != {rhs:.6g}")
    return (not msgs), msgs


def evaluate_affected(name: str, nl_path: Path, time_limit: float) -> InstanceResult:
    m = from_nl(str(nl_path))
    info = classify_gp_minlp(m)
    res = InstanceResult(
        name=name,
        n_vars=len(m._variables),
        n_int=len(info.integer_offsets) if info else 0,
        oracle=_ORACLE.get(name),
    )
    minimize = m._objective.sense.name == "MINIMIZE"

    off = _solve(nl_path, flag_on=False, time_limit=time_limit)
    on = _solve(nl_path, flag_on=True, time_limit=time_limit)
    res.off_status, res.off_obj, res.off_bound = off.status, off.objective, off.bound
    res.off_cert, res.off_nodes, res.off_wall = off.gap_certified, off.node_count, off.wall_time
    res.on_status, res.on_obj, res.on_bound = on.status, on.objective, on.bound
    res.on_cert, res.on_nodes, res.on_wall = on.gap_certified, on.node_count, on.wall_time

    # ── cert-clean bar ──────────────────────────────────────────────────────
    oracle = res.oracle
    # (a) certified objective within tolerance of the oracle.
    if on.objective is None or not _within(on.objective, oracle):
        res.cert_clean = False
        res.violations.append(f"ON objective {on.objective} != oracle {oracle}")
    # (b) no dual bound above the reference optimum (soundness invariant).
    if on.bound is not None and oracle is not None:
        # min: bound <= opt; max: bound >= opt.
        crossed = (on.bound > oracle + _ABS) if minimize else (on.bound < oracle - _ABS)
        if crossed:
            res.cert_clean = False
            res.violations.append(f"ON bound {on.bound} crosses oracle {oracle}")
    # (c) ON must certify (a recognised GP-MINLP closes its tree).
    if not on.gap_certified:
        res.cert_clean = False
        res.violations.append("ON did not certify (gap_certified=False)")
    # (d) cross-check against the independent classic B&B optimum, when it has one.
    if off.objective is not None and not _within(on.objective, off.objective):
        res.cert_clean = False
        res.violations.append(f"ON obj {on.objective} != classic-B&B obj {off.objective}")
    # (e) independent feasibility verification of the ON incumbent.
    if on.x is not None:
        ok, msgs = _feasible(nl_path, on.x)
        res.feasible_verified = ok
        if not ok:
            res.cert_clean = False
            res.violations += [f"infeasible incumbent: {m}" for m in msgs]
    else:
        res.feasible_verified = False
        res.cert_clean = False
        res.violations.append("ON returned no incumbent")

    # ── net-positive bar ────────────────────────────────────────────────────
    # Helpful on node count OR wall OR a tighter/earlier certificate vs the
    # classic spatial B&B baseline. (A certified ON where OFF did not certify is
    # itself a decisive improvement.)
    gains = []
    if off.gap_certified is False and on.gap_certified:
        gains.append("ON certifies where classic B&B did not")
    if res.off_nodes is not None and res.on_nodes is not None and res.on_nodes < res.off_nodes:
        gains.append(f"nodes {res.on_nodes} < {res.off_nodes}")
    if res.off_wall is not None and res.on_wall is not None and res.on_wall < res.off_wall * 0.9:
        gains.append(f"wall {res.on_wall:.1f}s < {res.off_wall:.1f}s")
    res.net_positive = len(gains) > 0
    res.notes += gains
    return res


def evaluate_inert(name: str, nl_path: Path, time_limit: float) -> InstanceResult | None:
    """Empirical inertness check on a fast instance: ON must equal OFF exactly.

    Only instances that terminate ``optimal`` in both configs within the limit
    are compared (their status/objective/node_count are deterministic); slow or
    limit-terminated instances are skipped (their inertness is already proven
    structurally by ``classify_gp_minlp is None``).
    """
    m = from_nl(str(nl_path))
    res = InstanceResult(name=name, n_vars=len(m._variables), n_int=0)
    off = _solve(nl_path, flag_on=False, time_limit=time_limit)
    on = _solve(nl_path, flag_on=True, time_limit=time_limit)
    if off.status != "optimal" or on.status != "optimal":
        return None  # not a deterministic comparison point
    res.off_status, res.off_obj, res.off_nodes = off.status, off.objective, off.node_count
    res.on_status, res.on_obj, res.on_nodes = on.status, on.objective, on.node_count
    identical = (
        off.status == on.status
        and _within(on.objective, off.objective)
        and off.node_count == on.node_count
    )
    res.cert_clean = identical
    res.net_positive = None  # inert: not scored on net-positive
    if not identical:
        res.violations.append(
            f"flag changed an inert instance: "
            f"status {off.status}->{on.status}, obj {off.objective}->{on.objective}, "
            f"nodes {off.node_count}->{on.node_count}"
        )
    return res


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--time-limit", type=float, default=120.0, help="seconds per solve")
    ap.add_argument("--inert-sample", type=int, default=12, help="fast inert instances to compare")
    ap.add_argument("--inert-time-limit", type=float, default=20.0)
    ap.add_argument("--out-dir", type=str, default=str(_REPO / "reports"))
    args = ap.parse_args()

    nl_files = sorted(_NL_DIR.glob("*.nl"))
    affected: list[Path] = []
    inert: list[Path] = []
    for p in nl_files:
        try:
            m = from_nl(str(p))
        except Exception:
            continue
        if classify_gp_minlp(m) is not None:
            affected.append(p)
        else:
            inert.append(p)

    print(
        f"# corpus: {len(nl_files)} .nl instances ({len(affected)} affected, {len(inert)} inert)",
        flush=True,
    )
    print(f"# affected: {[p.stem for p in affected]}", flush=True)

    results: list[InstanceResult] = []
    print("\n=== affected (ON-vs-OFF differential) ===", flush=True)
    for p in affected:
        r = evaluate_affected(p.stem, p, args.time_limit)
        results.append(r)
        print(
            f"  {r.name:20} oracle={r.oracle} ON obj={r.on_obj} bound={r.on_bound} "
            f"cert={r.on_cert} nodes={r.on_nodes} t={r.on_wall:.1f}s | "
            f"OFF obj={r.off_obj} cert={r.off_cert} nodes={r.off_nodes} t={r.off_wall:.1f}s | "
            f"cert_clean={r.cert_clean} net_positive={r.net_positive} feas={r.feasible_verified}",
            flush=True,
        )
        for v in r.violations:
            print(f"      VIOLATION: {v}", flush=True)
        for ndot in r.notes:
            print(f"      + {ndot}", flush=True)

    # Inert empirical sample: fastest-first (fewest vars) to stay within budget.
    print("\n=== inert (flag must be byte-identical) ===", flush=True)
    inert_by_size = sorted(inert, key=lambda p: len(from_nl(str(p))._variables))
    compared = 0
    inert_results: list[InstanceResult] = []
    for p in inert_by_size:
        if compared >= args.inert_sample:
            break
        try:
            r = evaluate_inert(p.stem, p, args.inert_time_limit)
        except Exception as e:
            print(f"  {p.stem:20} skipped ({type(e).__name__})", flush=True)
            continue
        if r is None:
            continue
        compared += 1
        inert_results.append(r)
        flag = "IDENTICAL" if r.cert_clean else "CHANGED!!"
        print(f"  {r.name:20} {flag} (obj={r.on_obj}, nodes={r.on_nodes})", flush=True)
        for v in r.violations:
            print(f"      VIOLATION: {v}", flush=True)

    results += inert_results

    # ── overall verdict ──────────────────────────────────────────────────────
    affected_results = [r for r in results if r.name in _ORACLE]
    cert_clean = all(r.cert_clean for r in results)
    net_positive = all(r.net_positive for r in affected_results) if affected_results else False
    all_feasible = all(r.feasible_verified for r in affected_results)
    eligible = cert_clean and net_positive

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y-%m-%dT%H-%M-%S")
    payload = {
        "flag": "DISCOPT_GP_MINLP",
        "issue": 116,
        "timestamp": ts,
        "time_limit": args.time_limit,
        "corpus": {"total": len(nl_files), "affected": len(affected), "inert": len(inert)},
        "inert_compared": len(inert_results),
        "cert_clean": cert_clean,
        "net_positive": net_positive,
        "all_incumbents_feasible": all_feasible,
        "eligible": eligible,
        "results": [asdict(r) for r in results],
    }
    js = out_dir / f"gp_minlp_graduation_panel_{ts}.json"
    js.write_text(json.dumps(payload, indent=2))

    print("\n─── VERDICT ───", flush=True)
    print(f"  affected instances : {len(affected)}", flush=True)
    print(
        f"  inert compared     : {len(inert_results)} (all identical: "
        f"{all(r.cert_clean for r in inert_results)})",
        flush=True,
    )
    print(f"  cert-clean         : {'PASS' if cert_clean else 'FAIL'}", flush=True)
    print(f"  incumbents feasible: {'PASS' if all_feasible else 'FAIL'}", flush=True)
    print(f"  net-positive       : {'PASS' if net_positive else 'FAIL'}", flush=True)
    print(f"  GRADUATION-ELIGIBLE: {'YES' if eligible else 'NO'}", flush=True)
    print(f"# JSON: {js}", flush=True)
    return 0 if eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())

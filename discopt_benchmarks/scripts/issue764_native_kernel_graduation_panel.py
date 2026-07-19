"""Issue #764 — Regime-2 graduation panel for the native Rust spatial B&B kernel.

Runs every ``.nl`` in ``python/tests/data/minlplib_nl/`` twice — with
``DISCOPT_NATIVE_SPATIAL_KERNEL`` OFF then ON — at a 60 s budget, one subprocess
per (instance, flag) so env / JAX / global-counter state is fully isolated. It
records per instance: status, objective, bound, node_count, wall, whether the
native kernel *engaged* (its hand-off returned a certified result), and — for every
ON solve where it engaged and reported optimal — an independent feasibility
verification of the returned incumbent against the ORIGINAL model.

It then evaluates the two CLAUDE.md Regime-2 bars over ALL instances:

  * CERT-CLEAN (hard gate, zero slack): no ON-optimal objective differing from the
    OFF-optimal objective by more than tol (abs 1e-6 / rel 1e-4); no ON dual bound
    past the reference optimum by more than tol (sense-aware); no OFF-optimal
    instance regressing to non-optimal ON; every engaged ON-optimal incumbent
    independently feasibility-verified.
  * NET-POSITIVE: the kernel engages somewhere AND measurably helps (e.g. a
    timeout->optimal), AND does not measurably harm the rest — the median wall
    delta on NON-engaged instances (the producer-probe decline overhead) is small
    (<= max(0.5 s, 5 %)).

Reference optima come from ``docs/dev/data/cert-optima.json``; instances absent
from it skip the oracle check and are SAID SO in the summary. Errored solves are
LABELED errored, never silently dropped.

Usage (parent):
    python discopt_benchmarks/scripts/issue764_native_kernel_graduation_panel.py
Child mode (internal): --solve <instance> <0|1>
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
_CORPUS = _REPO_ROOT / "python" / "tests" / "data" / "minlplib_nl"
_CERT_OPTIMA = _REPO_ROOT / "docs" / "dev" / "data" / "cert-optima.json"
_RESULTS_DIR = _BENCH_ROOT / "results"

_ABS_TOL = 1e-6
_REL_TOL = 1e-4
_TIME_LIMIT = 60.0
_CHILD_TIMEOUT = 200.0  # subprocess wall guard; solve itself is bounded to _TIME_LIMIT


def _obj_match(a, b) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= _ABS_TOL + _REL_TOL * max(abs(a), abs(b))


# --------------------------------------------------------------------------- #
# Child: solve ONE instance under ONE flag setting, print a single JSON line.  #
# --------------------------------------------------------------------------- #
def _run_child(instance: str, flag: str) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    if flag == "1":
        os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = "1"
    else:
        os.environ.pop("DISCOPT_NATIVE_SPATIAL_KERNEL", None)

    import discopt.solver as solver_mod
    import numpy as np
    from discopt.modeling.core import ObjectiveSense, from_nl

    nl = str(_CORPUS / f"{instance}.nl")
    out: dict = {"instance": instance, "flag": flag, "engaged": False}

    # Engagement detection: wrap the native hand-off; it returns non-None only on a
    # fully-certified native solve. Also stash the model so we can feasibility-verify.
    engaged = {"v": False}
    orig_fn = solver_mod._try_native_spatial_kernel

    def _wrapped(*a, **k):
        r = orig_fn(*a, **k)
        if r is not None:
            engaged["v"] = True
        return r

    solver_mod._try_native_spatial_kernel = _wrapped

    try:
        model = from_nl(nl)
        sense = "max" if model._objective.sense == ObjectiveSense.MAXIMIZE else "min"
        out["sense"] = sense
        t0 = time.perf_counter()
        r = model.solve(time_limit=_TIME_LIMIT)
        out["wall"] = time.perf_counter() - t0
        out["status"] = str(r.status)
        out["objective"] = None if r.objective is None else float(r.objective)
        out["bound"] = None if r.bound is None else float(r.bound)
        out["node_count"] = int(r.node_count)
        out["engaged"] = bool(engaged["v"])

        # Independent incumbent feasibility verification (ON + engaged + optimal).
        if flag == "1" and engaged["v"] and str(r.status) == "optimal" and r.x is not None:
            try:
                x_flat = np.array(
                    [float(np.asarray(r.x[v.name]).reshape(-1)[0]) for v in model._variables],
                    dtype=np.float64,
                )
                ok, verified_obj = solver_mod._native_kernel_verify_point(model, x_flat)
                out["incumbent_feasible"] = bool(ok)
                out["verified_obj"] = None if verified_obj is None else float(verified_obj)
            except Exception as exc:  # verification machinery failure -> record, don't pass
                out["incumbent_feasible"] = False
                out["verify_error"] = repr(exc)
    except Exception as exc:
        out["status"] = "errored"
        out["error"] = repr(exc)
    finally:
        solver_mod._try_native_spatial_kernel = orig_fn

    print("RESULT_JSON " + json.dumps(out))
    return 0


# --------------------------------------------------------------------------- #
# Parent: drive every instance, both flags, then evaluate the two bars.        #
# --------------------------------------------------------------------------- #
def _solve_one(instance: str, flag: str) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--solve", instance, flag]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_CHILD_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"instance": instance, "flag": flag, "status": "child_timeout", "engaged": False}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {
        "instance": instance,
        "flag": flag,
        "status": "child_crashed",
        "engaged": False,
        "stderr_tail": proc.stderr[-800:],
    }


def main() -> int:
    if len(sys.argv) >= 4 and sys.argv[1] == "--solve":
        return _run_child(sys.argv[2], sys.argv[3])

    optima: dict = {}
    if _CERT_OPTIMA.exists():
        optima = json.loads(_CERT_OPTIMA.read_text())

    instances = sorted(p.stem for p in _CORPUS.glob("*.nl"))
    # Optional smoke subset (validation only): PANEL_LIMIT=N runs the first N, and
    # PANEL_ONLY=a,b,c runs exactly those. Unset -> the full corpus.
    only = os.environ.get("PANEL_ONLY", "").strip()
    if only:
        wanted = {s.strip() for s in only.split(",") if s.strip()}
        instances = [i for i in instances if i in wanted]
    limit = os.environ.get("PANEL_LIMIT", "").strip()
    if limit.isdigit():
        instances = instances[: int(limit)]
    print(
        f"#764 native-kernel graduation panel: {len(instances)} instances, "
        f"budget {_TIME_LIMIT:.0f}s, OFF then ON, subprocess-isolated.\n"
        f"Reference optima: {len(optima)} entries in {_CERT_OPTIMA.name}.\n",
        flush=True,
    )

    rows: dict[str, dict] = {}
    for i, inst in enumerate(instances, 1):
        off = _solve_one(inst, "0")
        on = _solve_one(inst, "1")
        rows[inst] = {"off": off, "on": on}
        eng = "ENGAGED" if on.get("engaged") else "decline"
        print(
            f"  [{i:2d}/{len(instances)}] {inst:20s} "
            f"OFF={off.get('status', '?'):11s} ON={on.get('status', '?'):11s} {eng:8s} "
            f"objOFF={off.get('objective')!s:>12.12s} objON={on.get('objective')!s:>12.12s} "
            f"wOFF={off.get('wall', 0):6.1f} wON={on.get('wall', 0):6.1f}",
            flush=True,
        )

    verdict = _evaluate(rows, optima)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = _RESULTS_DIR / f"issue764_native_kernel_graduation_panel_{stamp}.json"
    raw_path.write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=1))
    summary = _render_summary(rows, verdict, optima, stamp)
    summary_path = _RESULTS_DIR / f"issue764_native_kernel_graduation_panel_{stamp}.txt"
    summary_path.write_text(summary)
    print("\n" + summary)
    print(f"\nRaw JSON:  {raw_path}")
    print(f"Summary :  {summary_path}")
    return 0 if verdict["graduate"] else 1


def _evaluate(rows: dict, optima: dict) -> dict:
    cert_violations: list[str] = []
    errored: list[str] = []
    engaged_insts: list[str] = []
    helped: list[str] = []  # engaged AND ON strictly better outcome than OFF
    non_engaged_wall_delta: list[float] = []
    no_oracle: list[str] = []

    _bad = ("errored", "child_crashed", "child_timeout")
    for inst, pair in sorted(rows.items()):
        off, on = pair["off"], pair["on"]
        off_status = off.get("status")
        on_status = on.get("status")
        if off_status in _bad:
            detail = str(off.get("error", off.get("stderr_tail", "")))[:120]
            errored.append(f"{inst}: OFF {off_status} ({detail})")
        if on_status in _bad:
            detail = str(on.get("error", on.get("stderr_tail", "")))[:120]
            errored.append(f"{inst}: ON {on_status} ({detail})")

        engaged = bool(on.get("engaged"))
        if engaged:
            engaged_insts.append(inst)
        else:
            # Producer-probe decline overhead: ON wall vs OFF wall on non-engaged.
            wall_ok = isinstance(off.get("wall"), (int, float)) and isinstance(
                on.get("wall"), (int, float)
            )
            if wall_ok and off_status not in _bad and on_status not in _bad:
                non_engaged_wall_delta.append(float(on["wall"]) - float(off["wall"]))

        sense = on.get("sense") or off.get("sense") or "min"

        # (1) objective agreement when BOTH optimal.
        if (
            off_status == "optimal"
            and on_status == "optimal"
            and not _obj_match(off.get("objective"), on.get("objective"))
        ):
            cert_violations.append(
                f"{inst}: ON/OFF optimal objective mismatch "
                f"OFF={off.get('objective')} ON={on.get('objective')}"
            )

        # (2) OFF-optimal must not regress to non-optimal ON.
        if off_status == "optimal" and on_status != "optimal":
            cert_violations.append(
                f"{inst}: OFF optimal but ON {on_status} (optimality regression)"
            )

        # (3) ON dual bound must not pass the reference optimum (sense-aware).
        if inst in optima and isinstance(on.get("bound"), (int, float)):
            opt = float(optima[inst])
            b = float(on["bound"])
            tol = _ABS_TOL + _REL_TOL * max(abs(opt), abs(b))
            if sense == "min" and b > opt + tol:
                cert_violations.append(
                    f"{inst}: ON lower bound {b} ABOVE reference optimum {opt} (+tol {tol:.1e})"
                )
            if sense == "max" and b < opt - tol:
                cert_violations.append(
                    f"{inst}: ON upper bound {b} BELOW reference optimum {opt} (-tol {tol:.1e})"
                )
        elif inst not in optima:
            no_oracle.append(inst)

        # (4) engaged ON-optimal incumbent must be independently feasibility-verified.
        if engaged and on_status == "optimal":
            if not on.get("incumbent_feasible", False):
                cert_violations.append(
                    f"{inst}: engaged ON optimal but incumbent NOT feasibility-verified "
                    f"({on.get('verify_error', '')})"
                )
            else:
                # And its verified true objective must match the reported objective.
                if not _obj_match(on.get("verified_obj"), on.get("objective")):
                    cert_violations.append(
                        f"{inst}: engaged ON verified_obj {on.get('verified_obj')} != "
                        f"reported {on.get('objective')}"
                    )

        # Net-positive "helped": engaged AND ON reached optimal where OFF did not.
        if engaged and on_status == "optimal" and off_status != "optimal":
            helped.append(inst)

    non_engaged_wall_delta.sort()
    median_delta = 0.0
    if non_engaged_wall_delta:
        n = len(non_engaged_wall_delta)
        median_delta = (
            non_engaged_wall_delta[n // 2]
            if n % 2
            else 0.5 * (non_engaged_wall_delta[n // 2 - 1] + non_engaged_wall_delta[n // 2])
        )

    cert_clean = len(cert_violations) == 0
    overhead_ok = median_delta <= max(0.5, 0.05 * _TIME_LIMIT)
    net_positive = (len(engaged_insts) > 0) and (len(helped) > 0) and overhead_ok

    return {
        "cert_clean": cert_clean,
        "cert_violations": cert_violations,
        "net_positive": net_positive,
        "engaged": engaged_insts,
        "helped": helped,
        "median_nonengaged_wall_delta_s": median_delta,
        "overhead_ok": overhead_ok,
        "errored": errored,
        "no_oracle_instances": no_oracle,
        "n_nonengaged_measured": len(non_engaged_wall_delta),
        "graduate": cert_clean and net_positive,
    }


def _render_summary(rows: dict, v: dict, optima: dict, stamp: str) -> str:
    lines: list[str] = []
    lines.append(f"# #764 native-kernel graduation panel — {stamp}")
    lines.append(
        f"# corpus: {len(rows)} instances, budget {_TIME_LIMIT:.0f}s, OFF vs ON, "
        f"subprocess-isolated. Reference optima: {len(optima)} entries."
    )
    lines.append("")
    lines.append("## VERDICT")
    lines.append(
        f"  cert-clean   : {'PASS' if v['cert_clean'] else 'FAIL'} "
        f"({len(v['cert_violations'])} violation(s))"
    )
    lines.append(
        f"  net-positive : {'PASS' if v['net_positive'] else 'FAIL'} "
        f"(engaged={len(v['engaged'])}, helped={len(v['helped'])}, "
        f"median non-engaged wall Δ={v['median_nonengaged_wall_delta_s']:+.3f}s over "
        f"{v['n_nonengaged_measured']} instances, overhead_ok={v['overhead_ok']})"
    )
    _grad = "YES — flip default ON" if v["graduate"] else "NO — keep opt-in"
    lines.append(f"  GRADUATE     : {_grad}")
    lines.append("")
    if v["cert_violations"]:
        lines.append("## CERT-CLEAN VIOLATIONS")
        for s in v["cert_violations"]:
            lines.append(f"  - {s}")
        lines.append("")
    _eng = ", ".join(v["engaged"]) or "(none)"
    lines.append(f"## ENGAGED instances ({len(v['engaged'])}): {_eng}")
    lines.append(
        f"## HELPED  (engaged, ON optimal where OFF was not) ({len(v['helped'])}): "
        f"{', '.join(v['helped']) or '(none)'}"
    )
    lines.append("")
    if v["errored"]:
        lines.append(f"## ERRORED ({len(v['errored'])})")
        for s in v["errored"]:
            lines.append(f"  - {s}")
        lines.append("")
    if v["no_oracle_instances"]:
        lines.append(
            f"## NO REFERENCE OPTIMUM (oracle check skipped) ({len(v['no_oracle_instances'])}):"
        )
        lines.append("  " + ", ".join(v["no_oracle_instances"]))
        lines.append("")
    lines.append("## PER-INSTANCE (engaged rows first)")
    ordered = sorted(rows.items(), key=lambda kv: (not kv[1]["on"].get("engaged"), kv[0]))
    lines.append(
        f"  {'instance':20s} {'OFF status':12s} {'ON status':12s} eng "
        f"{'objOFF':>14s} {'objON':>14s} {'boundON':>14s} {'wOFF':>7s} {'wON':>7s}"
    )
    for inst, pair in ordered:
        off, on = pair["off"], pair["on"]
        lines.append(
            f"  {inst:20s} {str(off.get('status')):12s} {str(on.get('status')):12s} "
            f"{'Y' if on.get('engaged') else '.':3s} "
            f"{str(off.get('objective'))[:14]:>14s} {str(on.get('objective'))[:14]:>14s} "
            f"{str(on.get('bound'))[:14]:>14s} "
            f"{off.get('wall', 0):7.1f} {on.get('wall', 0):7.1f}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 5.4 — the §5 differential panel for ``DISCOPT_CONVEX_KERNEL``.

Consolidation plan §0.1 Regime C: a bound-changing / routing flag graduates only on a
differential panel run **both directions** over the corpus, passing three bars —
cert-clean, quality-clean (#902), net-positive. This is that panel for the convex
kernel, structured after ``issue764_native_kernel_graduation_panel.py`` (the harness
whose defects #902 catalogued) with the same three lessons applied:

* **Both corpus directories.** ``minlplib_nl`` (68) and ``minlplib`` (81) are not
  nested; panelling one silently omits families. The union is 119.
* **Both arms set explicitly.** ``DISCOPT_CONVEX_KERNEL`` is written ``0``/``1`` in
  every child. Inferring an arm from a default the harness does not control is how a
  panel compares OFF against OFF forever.
* **Replication, not a load gate.** The decisive instances (those where the arms
  disagree, or where the kernel engaged) are re-run with the arms interleaved. A win
  must hold in EVERY replicate, a regression in a MAJORITY, and an instance whose
  replicates disagree is quarantined as *unresolved* — load can move a row to
  unresolved but can no longer make the verdict wrong.

Engagement is read from the solver, not guessed: the child records whether
``build_convex_spec`` accepted the model (eligibility) and whether the returned
``SolveResult`` came from ``try_convex_solve`` (adoption).

**A limit this panel cannot lift, stated loudly rather than buried.** The convex
family this flag exists for is ``syn*``/``rsyn*``/``clay*``/``cvxnonsep*`` — 136 `.nl`
files in the MINLPLib snapshot, of which the in-repo corpus vendors **four eligible
instances** (measured, not assumed: the eligibility sweep below accepts
``clay0303hfsg``, ``cvxnonsep_psig40r``, ``syn05hfsg``, ``syn05m`` and nothing else).
The named counter-case ``watercontamination0202`` is likewise snapshot-only. So this
panel can establish that turning the flag on does not harm the 115 instances it does
not route, and it can measure the four it does — but a graduation decision for the
family rests on a population this environment does not have. Wherever that matters the
output says ``SKIPPED — local only`` rather than passing on a set it never saw.

Usage::

    python -u discopt_benchmarks/scripts/phase5_convex_kernel_diff_panel.py
    python -u discopt_benchmarks/scripts/phase5_convex_kernel_diff_panel.py --budget 45 --reps 3
    python -u discopt_benchmarks/scripts/phase5_convex_kernel_diff_panel.py --subset 10

Internal child mode: ``--solve <instance> <0|1> <budget>``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "discopt_benchmarks") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "discopt_benchmarks"))
if str(_REPO_ROOT / "python") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "python"))

from scripts.panel_baseline import (  # noqa: E402
    _load1,
    _short_sha,
    corpus_instances,
    instance_path,
)
from utils.reference_optima import reference_oracle  # noqa: E402

_REPORTS_DIR = _REPO_ROOT / "reports"
_ABS_TOL = 1e-6
_REL_TOL = 1e-4
_CHILD_SLACK = 150.0

#: The family this flag targets, and the size of the pool that would decide it.
#: Counted in `sota-parity-analysis-2026-07-27.md` §4 P3 from the MINLPLib snapshot.
_FAMILY_POOL_SNAPSHOT = 136


def _run_child(instance: str, flag: str, budget: float) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    os.environ["DISCOPT_CONVEX_KERNEL"] = "1" if flag == "1" else "0"

    import discopt  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from discopt.modeling.core import ObjectiveSense, from_nl  # noqa: PLC0415
    from discopt.solvers import _convex_kernel  # noqa: PLC0415

    out: dict = {
        "instance": instance,
        "flag": flag,
        "budget": float(budget),
        "discopt_file": discopt.__file__,
        "kernel_enabled": bool(_convex_kernel.convex_kernel_enabled()),
        # CLAUDE.md §8 marker: the Phase 5.4 attempt clock. Its absence means the
        # child imported a pre-fix discopt and every wall number below is from the
        # wrong tree.
        "has_attempt_clock": hasattr(_convex_kernel, "last_attempt_seconds"),
    }

    # Tap adoption at the routing site so "the kernel solved this" is recorded, not
    # inferred from a suspiciously small wall.
    adopted = {"n": 0}
    _orig = _convex_kernel.try_convex_solve

    def _tapped(model, **kw):
        res = _orig(model, **kw)
        if res is not None:
            adopted["n"] += 1
        out["attempt_s"] = _convex_kernel.last_attempt_seconds()
        return res

    _convex_kernel.try_convex_solve = _tapped

    try:
        model = from_nl(str(instance_path(instance)))
        out["sense"] = "max" if model._objective.sense == ObjectiveSense.MAXIMIZE else "min"
        try:
            out["eligible"] = _convex_kernel.build_convex_spec(model) is not None
        except Exception as exc:
            out["eligible"] = None
            out["eligible_error"] = repr(exc)[:200]
        t0 = time.perf_counter()
        r = model.solve(time_limit=budget)
        out["wall"] = time.perf_counter() - t0
        out["status"] = str(r.status)
        out["objective"] = None if r.objective is None else float(r.objective)
        out["bound"] = None if r.bound is None else float(r.bound)
        out["node_count"] = int(r.node_count)
        out["gap_certified"] = bool(r.gap_certified)
        out["adopted"] = adopted["n"] > 0
        # Independent feasibility verification of the returned incumbent against the
        # PRISTINE model — the panel must not take the solver's word for its primal.
        out["verified"] = None
        if r.x is not None:
            try:
                flat = np.concatenate(
                    [np.asarray(r.x[v.name], dtype=float).ravel() for v in model._variables]
                )
                fresh = from_nl(str(instance_path(instance)))
                out["verified"] = bool(_convex_kernel._incumbent_is_feasible(fresh, flat))
            except Exception as exc:
                out["verify_error"] = repr(exc)[:200]
    except Exception as exc:
        out["status"] = "errored"
        out["error"] = repr(exc)
    finally:
        _convex_kernel.try_convex_solve = _orig

    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


def _solve(instance: str, flag: str, budget: float) -> dict:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--solve",
        instance,
        flag,
        str(budget),
    ]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=budget * 2 + _CHILD_SLACK, env=env
        )
    except subprocess.TimeoutExpired:
        return {"instance": instance, "flag": flag, "status": "child_timeout"}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {
        "instance": instance,
        "flag": flag,
        "status": "child_crashed",
        "stderr_tail": proc.stderr[-600:],
    }


def _obj_match(a, b) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= _ABS_TOL + _REL_TOL * max(abs(a), abs(b))


def _verdict(pairs: dict, budget: float) -> dict:
    """The three §5 bars over the paired rows. Every check increments a counter."""
    cert: list[str] = []
    verification_notes: list[str] = []
    quality: list[str] = []
    unstable: list[str] = []
    engaged: list[str] = []
    helped: list[str] = []
    errored: list[str] = []
    no_oracle: list[str] = []
    nonengaged_delta: list[float] = []
    executed = {"objective": 0, "optimality": 0, "bound_oracle": 0, "quality": 0, "verify": 0}

    for inst, p in sorted(pairs.items()):
        off, on = p.get("off"), p.get("on")
        if not off or not on:
            errored.append(f"{inst}: missing arm")
            continue
        off_s, on_s = str(off.get("status")), str(on.get("status"))
        if off_s in ("errored", "child_timeout", "child_crashed") or on_s in (
            "errored",
            "child_timeout",
            "child_crashed",
        ):
            errored.append(f"{inst}: OFF={off_s} ON={on_s}")
            continue
        sense = off.get("sense") or "min"
        is_engaged = bool(on.get("adopted"))
        if is_engaged:
            engaged.append(inst)
        else:
            if off.get("wall") is not None and on.get("wall") is not None:
                nonengaged_delta.append(float(on["wall"]) - float(off["wall"]))

        rep = p.get("replicates")
        if rep is not None and not rep.get("stable", False):
            unstable.append(
                f"{inst}: replicates disagree — OFF={[r.get('status') for r in rep['off']]} "
                f"ON={[r.get('status') for r in rep['on']]}"
            )
            continue

        # (1) both certified -> the certified objectives must agree.
        if off_s == "optimal" and on_s == "optimal":
            executed["objective"] += 1
            if not _obj_match(off.get("objective"), on.get("objective")):
                cert.append(
                    f"{inst}: certified objective differs OFF={off.get('objective')} "
                    f"ON={on.get('objective')}"
                )
        # (2) no certification regression.
        if off_s == "optimal":
            executed["optimality"] += 1
            if on_s != "optimal":
                cert.append(f"{inst}: OFF certified optimal, ON returned {on_s}")
        # (3) no dual bound past the reference optimum (sense-aware, proven only).
        oracle = reference_oracle(inst)
        if oracle is None or not oracle.proven:
            no_oracle.append(inst)
        else:
            for arm, row in (("OFF", off), ("ON", on)):
                b = row.get("bound")
                if b is None or not math.isfinite(b):
                    continue
                executed["bound_oracle"] += 1
                tol = _ABS_TOL + _REL_TOL * max(1.0, abs(oracle.value))
                bad = (b > oracle.value + tol) if sense == "min" else (b < oracle.value - tol)
                if bad:
                    cert.append(
                        f"{inst}: {arm} dual bound {b} past reference optimum "
                        f"{oracle.value} ({oracle.source}, sense={sense})"
                    )
        # (4) every returned incumbent independently verified feasible.
        #
        # Scoped DIFFERENTIALLY, and that is a correctness argument, not a
        # convenience. This panel's question is "does turning the flag on produce a
        # bad primal", so a verification failure is a cert violation when it is
        # ASYMMETRIC — ON fails where OFF passes. A failure that reproduces
        # identically in BOTH arms cannot have been caused by the flag; it is a
        # pre-existing property of the instance and of the verifier, and charging it
        # to the flag would make every future run of this panel FAIL for a reason it
        # cannot fix. Symmetric failures are NOT dropped: they are collected in
        # ``verification_notes`` and printed with the same prominence.
        #
        # Measured instance of exactly this: ``nvs22`` failed in both arms on two
        # defined-variable EQUALITY rows, residuals 1.71e-5 and 2.64e-4 against
        # variable values 2121.64 and 10782.7 — relative residuals 8.1e-9 and
        # 2.4e-8, with the incumbent objective matching ``=opt= 6.05822`` to 5.7e-8.
        # The verifier's tolerance was ``abs + rel*|residual|``, which on an equality
        # row degenerates to a pure absolute 1e-6 no matter how large the row is.
        # That was a real finding about the verifier and NOT evidence about
        # DISCOPT_CONVEX_KERNEL, which is why the gate is scoped this way.
        #
        # RESOLVED 2026-07-29 (plan card "the incumbent verifier's tolerance is
        # scale-blind"): every verifier now delegates to
        # ``discopt.validation.feasibility.verify_point``, whose row tolerance is
        # ``abs_tol * max(1, |b_i|, max_j |J_ij|*max(1,|x_j|))``. Re-measured on this
        # panel's own child, both arms, x2: ``nvs22`` now records
        # ``verified: true`` (worst relative row violation 1.5e-8). The differential
        # scoping stays — it is the right question for a differential panel — but it
        # is no longer load-bearing for this instance.
        off_ver, on_ver = off.get("verified"), on.get("verified")
        for row in (off, on):
            if row.get("objective") is not None:
                executed["verify"] += 1
        if on.get("objective") is not None and on_ver is False:
            if off_ver is False:
                verification_notes.append(
                    f"{inst}: incumbent fails independent verification in BOTH arms "
                    f"(pre-existing, not attributable to the flag)"
                )
            else:
                cert.append(
                    f"{inst}: ON incumbent FAILED independent feasibility check "
                    f"while OFF passed (OFF verified={off_ver})"
                )
        elif off.get("objective") is not None and off_ver is False:
            verification_notes.append(
                f"{inst}: OFF incumbent fails independent verification (ON did not "
                f"produce one to compare)"
            )
        # (5) quality (#902): ON must not lose or worsen a primal OFF found.
        off_o, on_o = off.get("objective"), on.get("objective")
        if rep is not None:
            off_o = rep.get("off_median_objective", off_o)
            on_o = rep.get("on_median_objective", on_o)
        if off_o is not None:
            executed["quality"] += 1
            if on_o is None:
                quality.append(
                    f"{inst}: PRIMAL LOST — OFF {off_o} ({off_s}), ON no incumbent ({on_s})"
                )
            else:
                qtol = _ABS_TOL + _REL_TOL * max(abs(off_o), abs(on_o))
                worse = (on_o > off_o + qtol) if sense == "min" else (on_o < off_o - qtol)
                if worse:
                    quality.append(
                        f"{inst}: INCUMBENT WORSE under ON — OFF={off_o} ({off_s}) "
                        f"vs ON={on_o} ({on_s})"
                    )
        # net-positive "helped": engaged AND ON certified where OFF did not.
        _reps_confirm_help = rep is None or (
            all(str(r.get("status")) == "optimal" for r in rep["on"])
            and not any(str(r.get("status")) == "optimal" for r in rep["off"])
        )
        if is_engaged and on_s == "optimal" and off_s != "optimal" and _reps_confirm_help:
            helped.append(inst)

    nonengaged_delta.sort()
    median = 0.0
    if nonengaged_delta:
        n = len(nonengaged_delta)
        median = (
            nonengaged_delta[n // 2]
            if n % 2
            else 0.5 * (nonengaged_delta[n // 2 - 1] + nonengaged_delta[n // 2])
        )
    overhead_ok = median <= max(0.5, 0.05 * budget)
    cert_clean = not cert
    quality_clean = not quality
    net_positive = bool(engaged) and bool(helped) and overhead_ok and quality_clean
    return {
        "cert_clean": cert_clean,
        "cert_violations": cert,
        "verification_notes": verification_notes,
        "quality_clean": quality_clean,
        "quality_violations": quality,
        "net_positive": net_positive,
        "engaged": engaged,
        "helped": helped,
        "unstable": unstable,
        "errored": errored,
        "no_oracle_instances": no_oracle,
        "median_nonengaged_wall_delta_s": median,
        "overhead_ok": overhead_ok,
        "n_nonengaged_measured": len(nonengaged_delta),
        "executed_checks": executed,
        "executed_total": sum(executed.values()),
        "graduate": cert_clean and quality_clean and net_positive,
    }


def _rescore(path: Path) -> int:
    """Recompute the verdict from a stored artifact's rows and rewrite it in place."""
    data = json.loads(path.read_text())
    pairs = data["pairs"]
    budget = float(data["budget"])
    v = _verdict(pairs, budget)
    for key in ("load_start", "load_peak", "replicates", "decisive_instances"):
        if key in data.get("verdict", {}):
            v[key] = data["verdict"][key]
    eligible = sorted(i for i, p in pairs.items() if p.get("on", {}).get("eligible"))
    print(f"RESCORED {path.name} ({len(pairs)} paired rows, budget {budget:.0f}s)")
    print("")
    print("## VERDICT")
    print(
        f"  cert-clean    : {'PASS' if v['cert_clean'] else 'FAIL'} ({len(v['cert_violations'])})"
    )
    for line in v["cert_violations"][:20]:
        print(f"      - {line}")
    _vn = v.get("verification_notes", [])
    print(f"  verification notes (symmetric, NOT charged to the flag): {len(_vn)}")
    for line in _vn[:20]:
        print(f"      - {line}")
    print(
        f"  quality-clean : {'PASS' if v['quality_clean'] else 'FAIL'} "
        f"({len(v['quality_violations'])})"
    )
    for line in v["quality_violations"][:20]:
        print(f"      - {line}")
    print(
        f"  net-positive  : {'PASS' if v['net_positive'] else 'FAIL'} "
        f"(engaged {len(v['engaged'])}, helped {len(v['helped'])}, "
        f"median non-engaged wall delta {v['median_nonengaged_wall_delta_s']:+.3f}s over "
        f"{v['n_nonengaged_measured']}, overhead_ok={v['overhead_ok']})"
    )
    print(f"  GRADUATE      : {'YES' if v['graduate'] else 'NO'}")
    print("")
    print(f"  eligible : {len(eligible)} -> {eligible}")
    print(f"  adopted  : {len(v['engaged'])} -> {v['engaged']}")
    print(f"  helped   : {len(v['helped'])} -> {v['helped']}")
    print(f"  unresolved: {len(v['unstable'])}")
    for line in v["unstable"][:10]:
        print(f"      - {line}")
    print(f"  EXECUTED CHECKS : {v['executed_total']} {v['executed_checks']}")
    data["verdict"] = v
    data["rescored"] = True
    path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str))
    print(f"\nartifact rewritten: {path}")
    if v["executed_total"] == 0:
        print("FAIL: zero executed checks", file=sys.stderr)
        return 2
    return 0 if (v["cert_clean"] and v["quality_clean"]) else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--solve", nargs=3, metavar=("INSTANCE", "FLAG", "BUDGET"))
    ap.add_argument("--budget", type=float, default=45.0)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--subset", default=None)
    ap.add_argument(
        "--rescore",
        default=None,
        help=(
            "re-evaluate the verdict from a stored artifact instead of re-solving. "
            "The per-instance rows are the measurement; the verdict is a function of "
            "them, so a corrected gate must not require another 3-hour run."
        ),
    )
    args = ap.parse_args(argv)
    if args.solve:
        return _run_child(args.solve[0], args.solve[1], float(args.solve[2]))
    if args.rescore:
        return _rescore(Path(args.rescore))

    instances = corpus_instances()
    if args.subset:
        if args.subset.isdigit():
            instances = instances[: int(args.subset)]
        else:
            want = {s.strip() for s in args.subset.split(",") if s.strip()}
            instances = [i for i in instances if i in want]

    load_start = _load1()
    load_peak = load_start
    t_start = time.perf_counter()

    # ---- stage 1: screen every instance once, both arms, interleaved ------- #
    pairs: dict[str, dict] = {}
    for i, inst in enumerate(instances, 1):
        off = _solve(inst, "0", args.budget)
        on = _solve(inst, "1", args.budget)
        pairs[inst] = {"off": off, "on": on}
        load_peak = max(load_peak, _load1())
        print(
            f"[{i:3d}/{len(instances)}] {inst:<26} "
            f"OFF {str(off.get('status')):<11} {off.get('wall', float('nan')):6.2f}s | "
            f"ON {str(on.get('status')):<11} {on.get('wall', float('nan')):6.2f}s "
            f"{'ADOPTED' if on.get('adopted') else ''}"
            f"{' ELIGIBLE' if on.get('eligible') else ''}",
            flush=True,
        )

    # ---- stage 2: replicate the decisive rows ------------------------------ #
    # A row is decisive when the flag could plausibly have changed its answer.
    # ``_obj_match`` returns False for ``(None, None)`` by design (an absent
    # objective never "matches"), so testing it directly made every
    # no-incumbent-in-both-arms instance decisive: the first run of this panel
    # replicated 20 rows where 6 were real, and 14 of those were instances that
    # return nothing under either arm at any budget. Treat "neither arm produced an
    # incumbent" as agreement — there is nothing there for replication to resolve.
    def _objectives_agree(a, b) -> bool:
        if a is None and b is None:
            return True
        return _obj_match(a, b)

    decisive = [
        inst
        for inst, p in pairs.items()
        if p["on"].get("adopted")
        or str(p["off"].get("status")) != str(p["on"].get("status"))
        or not _objectives_agree(p["off"].get("objective"), p["on"].get("objective"))
    ]
    print(f"\nstage 2: replicating {len(decisive)} decisive instance(s) x{args.reps}", flush=True)
    for inst in decisive:
        off_runs, on_runs = [], []
        for _ in range(args.reps):
            off_runs.append(_solve(inst, "0", args.budget))
            on_runs.append(_solve(inst, "1", args.budget))
            load_peak = max(load_peak, _load1())
        off_st = {str(r.get("status")) for r in off_runs}
        on_st = {str(r.get("status")) for r in on_runs}
        off_o = [r.get("objective") for r in off_runs if r.get("objective") is not None]
        on_o = [r.get("objective") for r in on_runs if r.get("objective") is not None]
        pairs[inst]["replicates"] = {
            "off": off_runs,
            "on": on_runs,
            "stable": len(off_st) == 1 and len(on_st) == 1,
            "off_median_objective": sorted(off_o)[len(off_o) // 2] if off_o else None,
            "on_median_objective": sorted(on_o)[len(on_o) // 2] if on_o else None,
        }
        print(
            f"  {inst:<26} OFF={sorted(off_st)} ON={sorted(on_st)} "
            f"stable={pairs[inst]['replicates']['stable']}",
            flush=True,
        )

    v = _verdict(pairs, args.budget)
    v["load_start"] = load_start
    v["load_peak"] = load_peak
    v["replicates"] = args.reps
    v["decisive_instances"] = decisive

    eligible = sorted(i for i, p in pairs.items() if p["on"].get("eligible"))
    stale = sorted(i for i, p in pairs.items() if p["on"].get("has_attempt_clock") is False)

    print("\n" + "=" * 96)
    print("PHASE 5.4 — DISCOPT_CONVEX_KERNEL DIFFERENTIAL PANEL (§5 Regime C)")
    print("=" * 96)
    print(
        f"corpus {len(instances)} instances (union of both in-repo dirs), budget "
        f"{args.budget:.0f}s, OFF vs ON, subprocess-isolated, {args.reps}x replication "
        f"of {len(decisive)} decisive row(s)."
    )
    _panel_wall = time.perf_counter() - t_start
    print(f"load start {load_start:.2f} peak {load_peak:.2f}; wall {_panel_wall:.0f}s")
    if stale:
        print(f"!! {len(stale)} child(ren) lacked the Phase 5.4 attempt clock: {stale[:5]}")
    print("")
    print("## VERDICT")
    print(
        f"  cert-clean    : {'PASS' if v['cert_clean'] else 'FAIL'} ({len(v['cert_violations'])})"
    )
    for line in v["cert_violations"][:20]:
        print(f"      - {line}")
    _vn = v.get("verification_notes", [])
    print(f"  verification notes (symmetric, NOT charged to the flag): {len(_vn)}")
    for line in _vn[:20]:
        print(f"      - {line}")
    print(
        f"  quality-clean : {'PASS' if v['quality_clean'] else 'FAIL'} "
        f"({len(v['quality_violations'])})"
    )
    for line in v["quality_violations"][:20]:
        print(f"      - {line}")
    print(
        f"  net-positive  : {'PASS' if v['net_positive'] else 'FAIL'} "
        f"(engaged {len(v['engaged'])}, helped {len(v['helped'])}, "
        f"median non-engaged wall delta {v['median_nonengaged_wall_delta_s']:+.3f}s over "
        f"{v['n_nonengaged_measured']}, overhead_ok={v['overhead_ok']})"
    )
    print(f"  GRADUATE      : {'YES' if v['graduate'] else 'NO'}")
    print("")
    print(f"  eligible (build_convex_spec accepted): {len(eligible)} -> {eligible}")
    print(f"  adopted  (try_convex_solve served)   : {len(v['engaged'])} -> {v['engaged']}")
    print(f"  helped   (ON certified, OFF did not) : {len(v['helped'])} -> {v['helped']}")
    print(f"  unresolved (replicates disagreed)    : {len(v['unstable'])}")
    for line in v["unstable"][:10]:
        print(f"      - {line}")
    print(f"  errored rows                         : {len(v['errored'])}")
    for line in v["errored"][:10]:
        print(f"      - {line}")
    print(f"  no proven oracle (unscored)          : {len(v['no_oracle_instances'])}")
    print(f"  EXECUTED CHECKS                      : {v['executed_total']} {v['executed_checks']}")
    print("")
    print("## POPULATION LIMIT — this panel cannot graduate the flag on its own")
    print(
        f"  the family this flag targets is syn*/rsyn*/clay*/cvxnonsep* — "
        f"{_FAMILY_POOL_SNAPSHOT} .nl files in the MINLPLib snapshot; the in-repo corpus "
        f"vendors {len(eligible)} eligible instance(s)."
    )
    print("  full convex-family panel: SKIPPED — local only (no MINLPLib snapshot here)")
    print("  watercontamination0202 misroute counter-case: SKIPPED — local only")

    _REPORTS_DIR.mkdir(exist_ok=True)
    out = _REPORTS_DIR / f"phase5_convex_kernel_diff_panel_{_short_sha()}.json"
    out.write_text(
        json.dumps(
            {
                "schema": "phase5_convex_diff_panel/1",
                "git_sha": _short_sha(),
                "budget": args.budget,
                "reps": args.reps,
                "instances": instances,
                "eligible": eligible,
                "verdict": v,
                "pairs": pairs,
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    print(f"\nartifact: {out.relative_to(_REPO_ROOT)}")

    if v["executed_total"] == 0:
        print("FAIL: zero executed checks — the panel measured nothing", file=sys.stderr)
        return 2
    return 0 if (v["cert_clean"] and v["quality_clean"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""#2021 section 1 entry experiments on p_ball (no library edit).

``trace`` -- solve one instance with the ``DISCOPT_OA_SUPPORT_CUTS`` /
``DISCOPT_CONVEX_ROUTE_FALLBACK_RESERVE`` pair on and record every #1066 route
guard consultation (elapsed, dual, primal, gap, verdict). ``--never-hand-over``
replaces the guard's verdict with "continue" so the route keeps the whole budget:
that measures the master's own certification time. Also captures the final
lp_nlp_bb master (MPS), the accepted incumbent ``x_out`` and the #1634
cross-solve's arguments and diagnostics.

``cross`` -- reload a captured master and run the #1634 cross-solve
(``milp_highs._cross_check_lazy_master``) cold vs seeded with ``x_out``
(``Highs.setSolution`` right after the cross-solve's ``passModel``), interleaved.

``replay`` -- replay a bound-trend extrapolation rule offline on ``trace`` output.

Every mode prints an EXECUTED count and exits non-zero when it is zero.

    python -u .../issue2021_pball_probe.py trace --nl X.nl --time-limit 60 --out d/
    python -u .../issue2021_pball_probe.py cross --capture d/capture.json --budget 60 --reps 3
    python -u .../issue2021_pball_probe.py replay --trace d/trace.json --cert-time 43.9
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _f(v):
    return None if v is None or not math.isfinite(float(v)) else float(v)


def trace(a) -> int:
    import discopt
    import discopt.solver as solver_mod
    import numpy as np
    from discopt.modeling.core import from_nl
    from discopt.solvers import milp_highs as mh
    from discopt.validation.feasibility import verify_point

    assert discopt.__file__.startswith(str(REPO)), discopt.__file__
    for marker in a.expect_marker or []:
        mod, _, text = marker.partition(":")
        src = Path(sys.modules[mod].__file__).read_text() if mod in sys.modules else None
        if src is None:
            __import__(mod)
            src = Path(sys.modules[mod].__file__).read_text()
        assert text in src, f"marker {text!r} absent from {mod}"
    for marker in a.expect_absent or []:
        mod, _, text = marker.partition(":")
        __import__(mod)
        assert text not in Path(sys.modules[mod].__file__).read_text(), f"{text!r} in {mod}"
    os.environ["DISCOPT_OA_SUPPORT_CUTS"] = "1"
    os.environ["DISCOPT_CONVEX_ROUTE_FALLBACK_RESERVE"] = "1"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    calls: list[dict] = []
    orig_call = solver_mod._RouteProgressGuard.__call__
    orig_verdict = solver_mod._RouteProgressGuard._verdict

    def _call(self, context):
        stop = orig_call(self, context)
        calls.append(
            {
                "elapsed": float(context["elapsed"]),
                "dual": _f(context["current_dual_bound"]),
                "primal": _f(context["current_primal_bound"]),
                "gap": _f(context["relative_gap"]),
                "checkpoint": self.checkpoint,
                "window": self.window,
                "limit": self.time_limit,
                "reserve": self.fallback_reserve,
                "stop": bool(stop),
                "reason": self.reason if stop else None,
            }
        )
        return stop

    def _never(self, now):
        stop, reason = orig_verdict(self, now)
        calls.append({"would_stop": bool(stop), "would_reason": reason, "at": now})
        if self.history and self.history[-1][1] <= 0.0:
            return False, None
        return False, None

    solver_mod._RouteProgressGuard.__call__ = _call
    if a.never_hand_over:
        solver_mod._RouteProgressGuard._verdict = _never
    cap: dict = {}
    orig_cross = mh._cross_check_lazy_master
    orig_lazy = mh.solve_milp_with_lazy_cuts

    def _cross(h, highspy, status, bound, time_left, objective=None, **kw):
        h.writeModel(str(out / "master.mps"))
        t0 = time.perf_counter()
        res = orig_cross(h, highspy, status, bound, time_left, objective=objective, **kw)
        cap.update(
            claim_status=status.value,
            claim_bound=_f(bound),
            time_left=_f(time_left),
            objective=_f(objective),
            cross_wall=time.perf_counter() - t0,
            out_status=res[0].value,
            out_bound=_f(res[1]),
            diag=res[2],
        )
        return res

    def _lazy(*args, **kw):
        t_enter = time.perf_counter() - t_solve[0]
        res = orig_lazy(*args, **kw)
        cap.update(
            x_out=None if res.x is None else [float(v) for v in res.x],
            master_enter=t_enter,
            master_wall=res.wall_time,
            master_bound=_f(res.bound),
            master_restarts=(res.callback_stats or {}).get("restarts"),
        )
        return res

    mh._cross_check_lazy_master = _cross
    mh.solve_milp_with_lazy_cuts = _lazy
    t_solve = [time.perf_counter()]
    r = from_nl(a.nl).solve(time_limit=a.time_limit, gap_tolerance=1e-4)
    wall = time.perf_counter() - t_solve[0]
    bad = None
    if r.x is not None and r.objective is not None:
        fresh = from_nl(a.nl)
        flat = np.concatenate(
            [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in fresh._variables]
        )
        bad = not verify_point(fresh, flat).ok
    result = {
        "nl": a.nl,
        "time_limit": a.time_limit,
        "never_hand_over": a.never_hand_over,
        "status": str(r.status),
        "gap_certified": bool(r.gap_certified),
        "objective": _f(r.objective),
        "bound": _f(r.bound),
        "wall": wall,
        "verify_failed": bad,
        "guard_calls": calls,
    }
    (out / "trace.json").write_text(json.dumps(result, default=str, indent=1))
    (out / "capture.json").write_text(json.dumps(cap, default=str))
    n = sum(1 for c in calls if "elapsed" in c)
    print(
        f"{Path(a.nl).stem} tl={a.time_limit} never={a.never_hand_over}: {r.status} "
        f"cert={r.gap_certified} obj={r.objective} bound={r.bound} wall={wall:.1f} "
        f"verify_failed={bad} guard_calls={n}"
    )
    for c in calls:
        print("  ", json.dumps(c))
    print(
        "  capture:",
        json.dumps({k: v for k, v in cap.items() if k != "x_out"}, default=str),
    )
    print(f"EXECUTED {n}")
    return 0 if n else 1


def cross(a) -> int:
    import discopt
    import highspy
    import numpy as np
    from discopt.solvers import milp_highs as mh

    assert discopt.__file__.startswith(str(REPO)), discopt.__file__
    cap = json.loads(Path(a.capture).read_text())
    mps = Path(a.capture).with_name("master.mps")
    seed = np.asarray(cap["x_out"], dtype=np.float64)
    status = mh.SolveStatus(cap["claim_status"])
    executed = 0
    rows = []
    orig_pass = highspy.Highs.passModel
    for rep in range(a.reps):
        for arm in ("cold", "warm") if rep % 2 == 0 else ("warm", "cold"):
            h = highspy.Highs()
            h.setOptionValue("output_flag", False)
            h.setOptionValue("mip_rel_gap", 1e-4)
            if h.readModel(str(mps)) != highspy.HighsStatus.kOk:
                raise RuntimeError(f"HiGHS could not read {mps}")
            seeded = [0]

            def _pass(self, *args, _arm=arm, _seeded=seeded, **kw):
                st = orig_pass(self, *args, **kw)
                if _arm == "warm":
                    sol = highspy.HighsSolution()
                    sol.col_value = seed
                    self.setSolution(sol)
                    _seeded[0] += 1
                return st

            highspy.Highs.passModel = _pass
            t0 = time.perf_counter()
            st, bound, diag = mh._cross_check_lazy_master(
                h, highspy, status, cap["claim_bound"], a.budget, objective=cap["objective"]
            )
            wall = time.perf_counter() - t0
            highspy.Highs.passModel = orig_pass
            executed += 1
            row = {
                "rep": rep,
                "arm": arm,
                "seeded": seeded[0],
                "wall": wall,
                "status": st.value,
                "bound": _f(bound),
                "reached_claim": diag.get("reached_claim"),
                "cross_nodes": diag.get("cross_nodes"),
                "cross_bound": diag.get("cross_bound"),
                "cross_status": diag.get("cross_status"),
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    print(f"claim {cap['claim_bound']} objective {cap['objective']} budget {a.budget}")
    print(f"EXECUTED {executed}")
    return 0 if executed else 1


def replay(a) -> int:
    """Replay the extrapolation rule: keep the route iff the trailing-window bound
    rate projects the dual bound onto the incumbent before ``limit - reserve``."""
    tr = json.loads(Path(a.trace).read_text())
    calls = [c for c in tr["guard_calls"] if "elapsed" in c]
    # the guard's own limit and #2013 reserve, as recorded (not the solve's limit)
    limit = float(calls[0]["limit"])
    reserve = float(calls[0]["reserve"])
    hist = [(c["elapsed"], c["dual"], c["primal"]) for c in calls if c["dual"] is not None]
    executed = 0
    first_handover = None
    for c in calls:
        now = c["elapsed"]
        if now < c["checkpoint"]:
            continue
        executed += 1
        win = [(t, d, p) for t, d, p in hist if now - c["window"] <= t <= now]
        if len(win) < 2:
            keep, why = False, "fewer than two bounds in the window"
        else:
            (t0, d0, _), (t1, d1, p1) = win[0], win[-1]
            rate = (d1 - d0) / max(t1 - t0, 1e-9)
            if p1 - d1 <= 0:
                keep, why = True, "converged"
            elif rate <= 0:
                keep, why = False, f"rate {rate:.3g}"
            else:
                eta = now + (p1 - d1) / rate
                keep = eta <= limit - reserve
                why = (
                    f"rate {rate:.3g}/s, projected certify at {eta:.1f}s vs {limit - reserve:.1f}s"
                )
        print(
            f"  t={now:6.2f} dual={c['dual']} primal={c['primal']} actual_stop={c['stop']} "
            f"trend_keep={keep} ({why})"
        )
        if not keep and first_handover is None:
            first_handover = now
    keeps = first_handover is None or (a.cert_time is not None and first_handover >= a.cert_time)
    print(
        f"first trend-rule handover: {first_handover}; measured certification time: "
        f"{a.cert_time}; rule {'KEEPS' if keeps else 'HANDS OVER BEFORE'} the route to it"
    )
    print(f"EXECUTED {executed}")
    return 0 if executed else 1


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="mode", required=True)
    t = sub.add_parser("trace")
    t.add_argument("--nl", required=True)
    t.add_argument("--time-limit", type=float, required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--never-hand-over", action="store_true")
    t.add_argument("--expect-marker", action="append", help="module:text that must be present")
    t.add_argument("--expect-absent", action="append", help="module:text that must be absent")
    c = sub.add_parser("cross")
    c.add_argument("--capture", required=True)
    c.add_argument("--budget", type=float, default=60.0)
    c.add_argument("--reps", type=int, default=3)
    r = sub.add_parser("replay")
    r.add_argument("--trace", required=True)
    r.add_argument("--cert-time", type=float, default=None)
    a = ap.parse_args()
    sys.exit({"trace": trace, "cross": cross, "replay": replay}[a.mode](a))


if __name__ == "__main__":
    main()

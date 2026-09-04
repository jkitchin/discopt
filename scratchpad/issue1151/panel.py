"""#1151 differential panel: incumbent-verifier row-scale fix, ON vs OFF.

Both arms run **in one process, interleaved per instance** (§9): the OFF arm is
produced by restoring the pre-#1151 floored row scale through a monkeypatch, so
the two arms see the same machine state on the same instance rather than being
separated by a 30-minute sequential run.

``deterministic=True`` was tried first and abandoned, with the measurement: it
renders the role-2 sub-budgets inert, and on this corpus that let the root phase
run 181 s (bchoco07) and 299 s (bchoco08) against a 30 s budget and still return
``nodes=0, bound=None`` — the #1152 root-setup overrun class. Those runs carry no
comparable content, so the panel runs on the real clock and the analysis
separates instances that terminate on WORK in both arms (comparable) from those
truncated by ``time_limit`` (two different amounts of search, not a comparison).

§6: every arm keeps an executed-call counter for the verifier and for the
tolerance-form divergence; the panel exits non-zero if either is zero.
§8: the patched sources are asserted present before anything is measured.
"""

import argparse
import inspect
import json
import os
import sys
import time

import numpy as np

import discopt
import discopt.modeling as dm
from discopt.validation import feasibility as F

MARKER = "#1151"

# ── the pre-#1151 row scale, transcribed, plus counters ──────────────────────
_NEW_ROW_SCALES = F._row_scales
STATS = {"verify_calls": 0, "row_scale_calls": 0, "divergent_rows": 0}


def _old_row_scales(evaluator, x_flat, rows):
    """``max_j |J_ij| * max(1, |x_j|)`` — the form this fix replaces."""
    STATS["row_scale_calls"] += 1
    try:
        J = np.asarray(evaluator.evaluate_jacobian(x_flat), dtype=np.float64)
    except Exception:
        return None
    if J.ndim != 2 or J.shape[0] <= int(rows.max()):
        return None
    xw = np.maximum(1.0, np.abs(np.asarray(x_flat, dtype=np.float64)))
    sub = np.abs(J[rows, :]) * xw[None, :]
    if not np.all(np.isfinite(sub)):
        return None
    return np.asarray(sub.max(axis=1), dtype=np.float64)


def _counting_new_row_scales(evaluator, x_flat, rows):
    STATS["row_scale_calls"] += 1
    new = _NEW_ROW_SCALES(evaluator, x_flat, rows)
    old = _old_row_scales(evaluator, x_flat, rows)
    STATS["row_scale_calls"] -= 1  # _old_row_scales double-counted
    if new is not None and old is not None:
        STATS["divergent_rows"] += int(np.count_nonzero(new < old - 1e-12))
    return new


_VERIFY = F.verify_point


def _counting_verify(*a, **k):
    STATS["verify_calls"] += 1
    return _VERIFY(*a, **k)


def set_arm(arm):
    F.verify_point = _counting_verify
    F._row_scales = _old_row_scales if arm == "off" else _counting_new_row_scales


def solve_one(path, time_limit):
    m = dm.from_nl(path)
    t0 = time.perf_counter()
    r = m.solve(time_limit=time_limit, gap_tolerance=1e-4)
    rec = {
        "status": r.status,
        "objective": None if r.objective is None else float(r.objective),
        "bound": None if r.bound is None else float(r.bound),
        "node_count": int(r.node_count or 0),
        "gap_certified": bool(getattr(r, "gap_certified", False)),
        "wall": time.perf_counter() - t0,
    }
    # #1151 oracle: re-evaluate the model objective at the solver's own point.
    if r.x is not None and r.objective is not None:
        flat = []
        for v in m._variables:
            val = r.x.get(v.name)
            if val is None:
                flat = None
                break
            flat.extend(np.asarray(val, dtype=float).ravel().tolist())
        if flat is not None:
            vr = _VERIFY(m, np.asarray(flat, dtype=float), with_objective=True)
            rec["oracle_ok"] = bool(vr.ok)
            rec["oracle_obj"] = None if vr.objective is None else float(vr.objective)
            rec["oracle_reason"] = vr.reason
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    n_marker = inspect.getsource(F).count(MARKER)
    print(f"[§8] discopt at {discopt.__file__}", flush=True)
    print(f"[§8] feasibility.py '{MARKER}' marker count = {n_marker}", flush=True)
    if n_marker == 0:
        sys.exit("ARM MISMATCH: the #1151 sources are NOT loaded; nothing to compare")

    roots = ["python/tests/data/minlplib_nl", "python/tests/data/minlplib"]
    paths = {}
    for root in roots:
        for fn in sorted(os.listdir(root)):
            if fn.endswith(".nl"):
                paths.setdefault(fn[:-3], os.path.join(root, fn))

    out = {"marker": n_marker, "time_limit": args.time_limit, "results": {}}
    items = sorted(paths.items())
    for i, (name, path) in enumerate(items, 1):
        rec = {"path": path}
        for arm in ("on", "off"):
            before = dict(STATS)
            set_arm(arm)
            try:
                rec[arm] = solve_one(path, args.time_limit)
            except Exception as exc:  # reported, never swallowed
                rec[arm] = {"error": f"{type(exc).__name__}: {exc}"}
            rec[arm]["verify_calls"] = STATS["verify_calls"] - before["verify_calls"]
            rec[arm]["row_scale_calls"] = STATS["row_scale_calls"] - before["row_scale_calls"]
            rec[arm]["divergent_rows"] = STATS["divergent_rows"] - before["divergent_rows"]
        out["results"][name] = rec
        a, b = rec["on"], rec["off"]
        flag = ""
        if a.get("status") != b.get("status") or (
            a.get("objective") is None
        ) != (b.get("objective") is None):
            flag = "  <<< DIFF"
        print(
            f"[{i:3d}/{len(items)}] {name:<22} "
            f"ON {a.get('status', a.get('error'))!s:<11} obj={a.get('objective')!r} "
            f"| OFF {b.get('status', b.get('error'))!s:<11} obj={b.get('objective')!r} "
            f"| divrows={a.get('divergent_rows')} ({a.get('wall', 0):.1f}s/"
            f"{b.get('wall', 0):.1f}s){flag}",
            flush=True,
        )

    out["stats"] = STATS
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"totals: {STATS}", flush=True)
    if not out["results"] or STATS["verify_calls"] == 0:
        sys.exit("PANEL MEASURED NOTHING (no instances, or the verifier never ran)")


main()

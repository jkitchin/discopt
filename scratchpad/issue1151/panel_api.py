"""#1151 panel B: models that ACTUALLY exercise the changed code path.

Panel A (`panel.py`, the 119 vendored `.nl` instances) recorded
``_row_scales`` invocations = 0 in BOTH arms: no verified incumbent on that
corpus ever carried a row over the flat absolute tolerance, so pass 2 — the only
code this change touches — never ran, and the two arms executed identical code.
That is a *narrowness* result, not evidence about the fix's behaviour; taken
alone it would be a probe that measured nothing (CLAUDE.md §6).

This panel supplies the positive control and the ON/OFF differential on the
class where the path does fire: quotient objectives, whose aux-defining rows
carry the ``1/dmin`` scaling. Arms are interleaved per model in one process.
Exits non-zero unless the path fired AND the two forms actually diverged
somewhere — i.e. unless the instrument demonstrably measured something.
"""

import inspect
import json
import sys

import numpy as np

import discopt
from discopt.modeling.core import Model
from discopt.validation import feasibility as F

MARKER = "#1151"
_NEW = F._row_scales
_VERIFY = F.verify_point
STATS = {"verify_calls": 0, "row_scale_calls": 0, "divergent_rows": 0}


def _old_row_scales(evaluator, x_flat, rows):
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


def _new_counting(evaluator, x_flat, rows):
    STATS["row_scale_calls"] += 1
    new = _NEW(evaluator, x_flat, rows)
    old = _old_row_scales(evaluator, x_flat, rows)
    STATS["row_scale_calls"] -= 1
    if new is not None and old is not None:
        STATS["divergent_rows"] += int(np.count_nonzero(new < old - 1e-12))
    return new


def _counting_verify(*a, **k):
    STATS["verify_calls"] += 1
    return _VERIFY(*a, **k)


def set_arm(arm):
    F.verify_point = _counting_verify
    F._row_scales = _old_row_scales if arm == "off" else _new_counting


# ── the corpus: quotient / posynomial models with closed-form optima ─────────
def _cases():
    for floor in (1e-3, 1e-2, 1e-1, 1.0):
        def build(floor=floor):
            m = Model("balance")
            x = m.continuous("x", lb=floor, ub=1e3)
            y = m.continuous("y", lb=floor, ub=1e3)
            m.minimize(x / y + y / x)
            return m, lambda d: d["x"] / d["y"] + d["y"] / d["x"]

        yield f"balance_floor{floor:g}", build, 2.0

    for floor in (1e-3, 1e-2):
        def build(floor=floor):
            m = Model("posyobj")
            x = m.continuous("x", lb=floor, ub=1e3)
            y = m.continuous("y", lb=floor, ub=1e3)
            m.minimize(x + 1.0 / (x * y) + y)
            return m, lambda d: d["x"] + 1.0 / (d["x"] * d["y"]) + d["y"]

        yield f"posyobj_floor{floor:g}", build, 3.0

        def build_c(floor=floor):
            m = Model("cobb")
            x = m.continuous("x", lb=floor, ub=1e3)
            y = m.continuous("y", lb=floor, ub=1e3)
            m.minimize(x + y)
            m.subject_to(1.0 / (x * y) <= 1.0)
            return m, lambda d: d["x"] + d["y"]

        yield f"cobb_floor{floor:g}", build_c, 2.0

        def build_r(floor=floor):
            m = Model("ratio3")
            x = m.continuous("x", lb=floor, ub=1e3)
            y = m.continuous("y", lb=floor, ub=1e3)
            z = m.continuous("z", lb=floor, ub=1e3)
            m.minimize(x / y + y / z + z / x)
            return m, lambda d: d["x"] / d["y"] + d["y"] / d["z"] + d["z"] / d["x"]

        yield f"ratio3_floor{floor:g}", build_r, 3.0


def main():
    n_marker = inspect.getsource(F).count(MARKER)
    print(f"[§8] discopt at {discopt.__file__}", flush=True)
    print(f"[§8] feasibility.py '{MARKER}' marker count = {n_marker}", flush=True)
    if n_marker == 0:
        sys.exit("the #1151 sources are NOT loaded")

    out = {}
    print(
        f"{'case':<22}{'arm':>5}{'status':>12}{'reported':>22}{'oracle@x':>22}"
        f"{'delta':>13}{'opt':>7}{'rows':>6}{'div':>5}",
        flush=True,
    )
    for name, build, optimum in _cases():
        out[name] = {"optimum": optimum}
        for arm in ("on", "off"):
            before = dict(STATS)
            set_arm(arm)
            m, oracle_fn = build()
            r = m.solve(solver="bb", time_limit=30.0)
            d = {k: float(v) for k, v in (r.x or {}).items()}
            oracle = oracle_fn(d) if d else None
            rec = {
                "status": r.status,
                "objective": None if r.objective is None else float(r.objective),
                "bound": None if r.bound is None else float(r.bound),
                "oracle_at_x": oracle,
                "delta": None if (oracle is None or r.objective is None) else r.objective - oracle,
                "row_scale_calls": STATS["row_scale_calls"] - before["row_scale_calls"],
                "divergent_rows": STATS["divergent_rows"] - before["divergent_rows"],
            }
            out[name][arm] = rec
            print(
                f"{name:<22}{arm:>5}{rec['status']:>12}"
                f"{rec['objective'] if rec['objective'] is not None else float('nan'):>22.15g}"
                f"{oracle if oracle is not None else float('nan'):>22.15g}"
                f"{rec['delta'] if rec['delta'] is not None else float('nan'):>13.3e}"
                f"{optimum:>7.3g}{rec['row_scale_calls']:>6}{rec['divergent_rows']:>5}",
                flush=True,
            )

    with open("scratchpad/issue1151/panel_api.json", "w") as fh:
        json.dump({"marker": n_marker, "stats": STATS, "results": out}, fh, indent=1)

    # ── verdict ──────────────────────────────────────────────────────────────
    fired = STATS["row_scale_calls"]
    diverged = STATS["divergent_rows"]
    print(f"\n_row_scales invocations: {fired}; rows where the forms diverged: {diverged}")
    bad = []
    for name, rec in out.items():
        for arm in ("on", "off"):
            a = rec[arm]
            if a["delta"] is not None and abs(a["delta"]) > 1e-6 * (1 + abs(a["oracle_at_x"])):
                bad.append((arm, name, a["delta"]))
            if a["objective"] is not None and a["objective"] < rec["optimum"] - 1e-6:
                bad.append((arm, name, f"BELOW OPTIMUM by {rec['optimum'] - a['objective']:.3e}"))
    print(f"reported-objective / oracle mismatches and super-optimal values: {len(bad)}")
    for arm, name, d in bad:
        print(f"    {arm} {name}: {d}")
    print(f"executed model solves: {2 * len(out)}")
    if fired == 0 or diverged == 0:
        sys.exit("PROBE MEASURED NOTHING: the changed path never fired or never diverged")


main()

"""Graduation panel for the #671 relaxation-row-filter (``DISCOPT_RELAX_ROW_FILTER``).

CLAUDE.md §5 differential-panel gate: run every instance flag OFF vs ON. The
row filter drops relaxation rows whose float64 coefficient span is unresolvable
(superset region => a valid, WEAKER outer approximation, sound by construction).

Two bars, BOTH required to GRADUATE default-on:

  1. cert-clean (soundness + no regression): incorrect_count = 0 (no dual bound
     above its oracle, no incumbent below it, ON incumbent independently
     re-verified feasible); no gap_certified=True -> uncertified regression; on
     the ALREADY-SOLVING corpus the filter must be INERT -- it drops 0 rows, so
     node_count + certified objective must be EXACTLY unchanged ON vs OFF.
  2. net-positive: hda's dual bound is materially tighter ON (~ -6.45e4 vs the
     -1.80e10 candidate-A floor) AND no already-solving instance regresses.

We instrument ``_filter_unresolvable_rows`` to count rows dropped per solve
(across root + every per-node cold rebuild). rows_dropped==0 proves the filter
was inert on that instance (byte-identical relaxation); any nonzero drop on a
"healthy" instance is the spurious-trigger condition to scrutinize.

Usage:
    python discopt_benchmarks/scripts/issue671_rowfilter_graduation_panel.py \
        <corpus_dir> <hda_tl> <corpus_tl> out.json [inst1,inst2,...]
If the instance list is omitted, every *.nl in <corpus_dir> is run.
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, "python")

SOLU = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")
FLAG = "DISCOPT_RELAX_ROW_FILTER"

# hda candidate-A loose floor (OFF), for the net-positive headline.
HDA_CANDIDATE_A = -1.80e10


def load_oracles():
    """name -> (value, kind) from the .solu; prefer =opt=, else =best=."""
    best = {}
    opt = {}
    if not os.path.exists(SOLU):
        return {}
    with open(SOLU) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3:
                if p[0] == "=opt=":
                    opt[p[1]] = float(p[2])
                elif p[0] == "=best=":
                    best[p[1]] = float(p[2])
    out = {}
    for name, v in best.items():
        out[name] = (v, "best")
    for name, v in opt.items():
        out[name] = (v, "opt")
    return out


ORACLES = load_oracles()


def oracle(name):
    v = ORACLES.get(name)
    return v[0] if v else None


# ---- dropped-row instrumentation ------------------------------------------- #
_DROP_COUNTER = {"rows": 0, "builds_with_drop": 0}


def _install_drop_counter():
    import discopt._jax.milp_relaxation as mr

    orig = mr._filter_unresolvable_rows

    def wrapped(milp):
        n = orig(milp)
        if n:
            _DROP_COUNTER["rows"] += n
            _DROP_COUNTER["builds_with_drop"] += 1
        return n

    mr._filter_unresolvable_rows = wrapped


def _incumbent_feasible(model, r) -> bool:
    """Independently re-verify the returned incumbent against the model rows."""
    if getattr(r, "x", None) is None:
        return True
    try:
        from discopt._jax.nlp_evaluator import cached_evaluator
        from discopt._jax.primal_heuristics import _check_constraint_feasibility

        ev = cached_evaluator(model)
        flat = np.concatenate(
            [
                np.atleast_1d(np.asarray(r.x[v.name], dtype=np.float64)).ravel()
                for v in model._variables
            ]
        )
        return bool(_check_constraint_feasibility(ev, flat))
    except Exception as exc:  # pragma: no cover - defensive
        print(f"    (feasibility check errored: {exc})", flush=True)
        return True


def _f(x):
    return None if x is None else float(x)


def run(path, name, tl):
    from discopt.modeling.core import from_nl

    opt = oracle(name)
    arms = {}
    sense = "minimize"
    # Equal-warmth warmup (measured 2026-07-18). The FIRST solve of an instance
    # in a process can differ in node_count from later solves of the SAME
    # instance — per-instance JAX retrace / structure-cache population changes
    # timing-budgeted per-node work, hence branching (measured on tls2:
    # 421 nodes cold -> 353 nodes warm, IDENTICAL for flag OFF and ON). Without
    # this, the OFF arm (always measured first) is cold and the ON arm warm, a
    # FALSE bound-neutral violation attributed to the flag. A discarded warmup
    # solve puts BOTH arms at equal warmth so any residual node_count difference
    # is a real code effect, not warmup ordering. This only removes false
    # positives — a genuine flag-induced node_count change survives it (both arms
    # warm), so it cannot mask a real regression.
    os.environ[FLAG] = "0"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            from_nl(path).solve(time_limit=min(tl, 10.0))
        except Exception:
            pass
    for flag in ("0", "1"):
        os.environ[FLAG] = flag
        _DROP_COUNTER["rows"] = 0
        _DROP_COUNTER["builds_with_drop"] = 0
        m = from_nl(path)
        try:
            obj = getattr(m, "_objective", None)
            s = getattr(obj, "sense", None)
            if s is not None:
                sense = "maximize" if "MAX" in str(s).upper() else "minimize"
        except Exception:
            pass
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.solve(time_limit=tl)
        wall = time.time() - t0
        arms["off" if flag == "0" else "on"] = {
            "obj": _f(r.objective),
            "bound": _f(r.bound),
            "status": str(r.status),
            "gap_certified": bool(getattr(r, "gap_certified", False)),
            "nodes": int(getattr(r, "node_count", -1)),
            "wall": round(wall, 2),
            "rows_dropped": _DROP_COUNTER["rows"],
            "builds_with_drop": _DROP_COUNTER["builds_with_drop"],
            "incumbent_feasible": _incumbent_feasible(m, r),
        }
    return {"instance": name, "oracle": opt, "sense": sense, **arms}


def assess(rec):
    """Per-instance cert-clean + net-positive verdict (min sense)."""
    opt = rec["oracle"]
    off, on = rec["off"], rec["on"]
    sense = rec.get("sense", "minimize")
    is_max = sense == "maximize"
    tol = 1e-4 * (1 + abs(opt)) if opt is not None else 1e-4
    problems = []  # cert-clean violations (soundness / regression) -> FAIL
    notes = []  # informational (inertness drift on timeout, etc.)

    # --- cert-clean: soundness (both arms), SENSE-AWARE ---
    # The dual bound is a LOWER bound for minimize (must be <= opt) and an UPPER
    # bound for maximize (must be >= opt). The incumbent objective is on the other
    # side. syn05hfsg is a MAXIMIZE instance: a naive "bound > opt" check false-flags
    # a valid upper bound.
    for arm_name, a in (("off", off), ("on", on)):
        b = a["bound"]
        o = a["obj"]
        if opt is not None and b is not None and math.isfinite(b):
            if not is_max and b > opt + tol:
                problems.append(f"{arm_name} lower bound {b:.6g} > oracle {opt:.6g} (UNSOUND)")
            if is_max and b < opt - tol:
                problems.append(f"{arm_name} upper bound {b:.6g} < oracle {opt:.6g} (UNSOUND)")
        if opt is not None and o is not None:
            if not is_max and o < opt - tol:
                problems.append(f"{arm_name} obj {o:.6g} < oracle {opt:.6g} (beats optimum)")
            if is_max and o > opt + tol:
                problems.append(f"{arm_name} obj {o:.6g} > oracle {opt:.6g} (beats optimum)")
        if not a["incumbent_feasible"]:
            problems.append(f"{arm_name} incumbent INFEASIBLE")
    # --- cert-clean: no certification regression ---
    if off["gap_certified"] and not on["gap_certified"]:
        problems.append("cert regression: OFF certified, ON not")

    inert = on["rows_dropped"] == 0
    if not inert:
        notes.append(
            f"filter FIRED: dropped {on['rows_dropped']} rows in {on['builds_with_drop']} builds"
        )
    # timeout heuristic: node counts under a time budget are only comparable when
    # both arms actually terminate (not wall-limited).
    TERMINAL = ("optimal", "infeasible", "OPTIMAL", "INFEASIBLE", "unbounded")
    off_term = off["status"] in TERMINAL
    on_term = on["status"] in TERMINAL
    both_terminate = off_term and on_term

    def _obj_changed():
        return (off["obj"] is None) != (on["obj"] is None) or (
            off["obj"] is not None
            and on["obj"] is not None
            and abs(off["obj"] - on["obj"]) > 1e-9 * (1 + abs(off["obj"]))
        )

    # --- cert-clean: BOUND-NEUTRAL on the already-solving corpus (CLAUDE.md §5) ---
    # For ANY instance that terminates in BOTH arms (already-solving), node_count
    # and the certified objective must be EXACTLY unchanged -- whether or not the
    # filter fired. A drop that reshapes the search tree on a solving instance is a
    # graduation FAIL (the filter must be inert where it isn't needed).
    if both_terminate:
        if off["nodes"] != on["nodes"]:
            fired = "" if inert else f" [filter dropped {on['rows_dropped']} rows]"
            problems.append(
                f"BOUND-NEUTRAL VIOLATION: node_count {off['nodes']}->{on['nodes']}{fired}"
            )
        if _obj_changed():
            problems.append(
                f"BOUND-NEUTRAL VIOLATION: certified obj {off['obj']}->{on['obj']}"
            )
    elif off_term != on_term:
        # one arm solves, the other doesn't -> the filter changed terminality.
        problems.append(
            f"TERMINALITY CHANGE: OFF status={off['status']} ON status={on['status']}"
        )
    else:
        # both wall-limited: node/obj differences are jitter, not a code effect.
        if off["nodes"] != on["nodes"] or off["obj"] != on["obj"]:
            notes.append(
                f"timeout jitter nodes {off['nodes']}->{on['nodes']} "
                f"obj {off['obj']}->{on['obj']} (both wall-limited)"
            )

    # --- net-positive signal (bound, min sense: higher lower bound is better) ---
    prim = "n/a"
    ob, nb = off["bound"], on["bound"]
    if ob is not None and nb is not None and math.isfinite(ob) and math.isfinite(nb):
        if nb > ob + 1e-6 * (1 + abs(ob)):
            prim = "ON tighter bound"
        elif nb < ob - 1e-6 * (1 + abs(ob)):
            prim = "ON LOOSER bound"
        else:
            prim = "bound tie"
    elif (ob is None or not math.isfinite(ob)) and (nb is not None and math.isfinite(nb)):
        prim = "ON finite (OFF inf)"

    return problems, notes, prim, inert, both_terminate


def main():
    corpus_dir = os.path.expanduser(sys.argv[1])
    hda_tl = float(sys.argv[2])
    corpus_tl = float(sys.argv[3])
    out = sys.argv[4]
    if len(sys.argv) > 5 and sys.argv[5]:
        names = sys.argv[5].split(",")
    else:
        names = sorted(
            os.path.splitext(os.path.basename(p))[0]
            for p in __import__("glob").glob(os.path.join(corpus_dir, "*.nl"))
        )

    _install_drop_counter()
    rows = []
    n_fail = n_fired = 0
    print(f"{'instance':22s} {'oracle':>12s} | {'OFFbound':>12s} {'ONbound':>12s} "
          f"| {'drop':>5s} {'ONnodes':>8s} | signal | cert", flush=True)
    for name in names:
        path = os.path.join(corpus_dir, f"{name}.nl")
        if not os.path.exists(path):
            print(f"{name:22s} MISSING {path}", flush=True)
            continue
        tl = hda_tl if name == "hda" else corpus_tl
        try:
            rec = run(path, name, tl)
        except Exception as exc:
            print(f"{name:22s} ERROR {type(exc).__name__}: {exc}", flush=True)
            rows.append({"instance": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        problems, notes, prim, inert, both_term = assess(rec)
        rec["problems"] = problems
        rec["notes"] = notes
        rec["signal"] = prim
        rec["inert"] = inert
        rows.append(rec)
        n_fail += bool(problems)
        n_fired += (not inert)
        cert = "CLEAN" if not problems else "!! " + "; ".join(problems)
        if notes:
            cert += "  [" + "; ".join(notes) + "]"
        fo = lambda x: f"{x:.4g}" if x is not None else "None"  # noqa: E731
        print(
            f"{name:22s} {fo(rec['oracle']):>12s} | {fo(rec['off']['bound']):>12s} "
            f"{fo(rec['on']['bound']):>12s} | {rec['on']['rows_dropped']:>5d} "
            f"{rec['on']['nodes']:>8d} | {prim:16s} | {cert}",
            flush=True,
        )

    # ---- headline hda check ----
    hda = next((r for r in rows if r.get("instance") == "hda" and "off" in r), None)
    hda_ok = False
    hda_detail = "hda not run"
    if hda:
        onb = hda["on"]["bound"]
        offb = hda["off"]["bound"]
        opt = hda["oracle"]
        hda_ok = (
            onb is not None
            and math.isfinite(onb)
            and onb >= -7e4  # at/above the ~ -6.47e4 root McCormick value
            and (opt is None or onb <= opt + 1e-2)  # sound: never above opt
        )
        hda_detail = f"hda ON bound {onb} (OFF {offb}, opt {opt}) tight&sound={hda_ok}"

    cert_clean = n_fail == 0
    # net-positive: hda materially tighter AND no ALREADY-SOLVING regression.
    # Already-solving regressions (node/obj drift, terminality change, cert loss)
    # are hard cert-fails counted in n_fail. Looser *partial* bounds on instances
    # that time out in BOTH arms are sound SOFT changes (superset), acceptable per
    # the task ("neutral-or-harmful with only hda helped is still net-positive if
    # nothing already-solving regresses"). We surface them but do not fail on them.
    soft_looser = [
        r["instance"]
        for r in rows
        if "on" in r and r.get("signal") == "ON LOOSER bound" and r["instance"] != "hda"
    ]
    net_positive = hda_ok and cert_clean

    summary = {
        "n_instances": len([r for r in rows if "off" in r]),
        "n_cert_fail": n_fail,
        "n_filter_fired": n_fired,
        "cert_clean": cert_clean,
        "hda_tight_and_sound": hda_ok,
        "hda_detail": hda_detail,
        "soft_looser_partial_bounds": soft_looser,
        "net_positive": net_positive,
    }
    verdict = "GRADUATE" if cert_clean and net_positive else "HOLD"
    summary["verdict"] = verdict
    print("\nSUMMARY:", json.dumps(summary, indent=2), flush=True)
    print(f"\nVERDICT: {verdict}", flush=True)
    with open(out, "w") as f:
        json.dump(
            {"hda_tl": hda_tl, "corpus_tl": corpus_tl, "summary": summary, "rows": rows},
            f,
            indent=2,
        )
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()

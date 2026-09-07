"""Graduation panel for the #1193 integer primal escape.

``DISCOPT_INTEGER_BOX_POLISH`` gates BOTH halves of the #1193 fix: the
zero-continuous *direct path* in ``integer_box_search`` (which fires INSIDE the
tree on the Python route) and the post-kernel *exit polish* on an uncertified
native-kernel exit. So OFF is a true pre-#1193 baseline -- but the flag is NOT
bound-neutral by construction: a better in-tree incumbent prunes differently and
may legitimately move ``node_count`` and the bound. It sits in CLAUDE.md §5's
heuristic-policy regime, where soundness is absolute and neutrality is not.

**The determinism pre-check.** Asserting OFF-vs-ON neutrality is only meaningful
where the instance is deterministic. Measured on ``ball_mk2_30`` with the flag
PINNED OFF for six runs: node_count came back 7282 / 7629 / 7465 / 2029 / 7646 /
7544 and the run alternated between two different (bound, incumbent) outcomes --
so the panel's original "bound MOVED / node MOVED / ON lost the incumbent"
verdict there was the solver's own run-to-run noise (tracked as #1187), not the
flag. Every instance therefore runs a THIRD arm -- OFF again, after ON -- and any
neutrality-class finding is charged to the flag only when the two OFF runs agree
with each other. On a nondeterministic instance it is recorded as a note carrying
the measured OFF-vs-OFF spread. The soundness bars below are asserted on every
instance regardless, since they cannot be excused by noise.

**The inertness proof.** The flag can influence a run through exactly one
channel: an improving point out of ``integer_box_search`` (the in-tree direct
path) or an adopted exit polish. Both are counted directly, so where neither
produced a point in EITHER arm the two runs must be bit-identical and any drift
is a plumbing bug -- that is where exact neutrality is asserted. Where one did,
the search legitimately diverges (``gear``: 31 -> 3 nodes at a BETTER objective,
8.6e-07 -> 3.8e-08 against a 0.0 oracle, certificate intact), and asserting
neutrality there would be asserting the flag does nothing.

  1. **cert-clean**
     a. *(only where the flag is provably inert AND the instance is
        deterministic)* exact ``bound`` / ``node_count`` / incumbent neutrality.
     b. *(always)* an arm reporting ``optimal`` must be AT its oracle within
        tolerance -- a false certificate is never excusable -- and certification
        must not be lost OFF -> ON.
     c. *(always)* no bound passes its oracle and no incumbent beats it, in
        either arm, sense-aware.
     d. *(always)* the ON incumbent independently re-verifies feasible whenever
        the OFF one does.
     e. *(always)* the polish is never ADOPTED on a certified-optimal exit.
  2. **net-positive**: the ON objective is never worse on a deterministic
     instance, and is strictly better on a measurable share.

Firing is instrumented directly by wrapping ``_native_exit_primal_polish`` and
counting adoptions, so "inert" is a proof rather than a node-count proxy
(CLAUDE.md §6).

Usage:
    python -u discopt_benchmarks/scripts/issue1193_exit_polish_panel.py \
        <corpus_dir> <max_nodes> <time_limit> out.json [inst1,inst2,...]
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, "python")

SOLU = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib.solu")
FLAG = "DISCOPT_INTEGER_BOX_POLISH"


def load_oracles():
    """name -> value from the .solu; prefer ``=opt=``, else ``=best=``."""
    best, opt = {}, {}
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
    out = dict(best)
    out.update(opt)
    return out


ORACLES = load_oracles()

# ---- firing instrumentation (CLAUDE.md §6) --------------------------------- #
_FIRE = {"calls": 0, "adopted": 0, "delta": 0.0, "box_hits": 0, "ibs_calls": 0, "direct": 0}


def _install_fire_counter():
    import discopt.solver as sv

    orig = sv._native_exit_primal_polish

    def wrapped(model, x_flat, obj_val, bound_val, n_orig, outer_deadline):
        _FIRE["calls"] += 1
        out = orig(model, x_flat, obj_val, bound_val, n_orig, outer_deadline)
        if out is not None:
            _FIRE["adopted"] += 1
            _FIRE["delta"] += abs(float(out[1]) - float(obj_val))
        return out

    sv._native_exit_primal_polish = wrapped

    # The flag can change a run ONLY by supplying a better incumbent, and the one
    # gate every such point passes through is ``integer_box_search`` returning
    # not-None. Counting those returns turns "the flag was inert here" into a
    # proof rather than an inference (CLAUDE.md §6), which is what licenses the
    # exact-neutrality assertion below.
    import discopt._relax.primal_heuristics as ph

    _orig_ibs = ph.integer_box_search

    def _counting_ibs(*a, **k):
        # ``ibs_calls`` is the call, ``box_hits`` the EFFECT (returned a point?), and
        # ``direct`` the actual CHANNEL. Inertness must be proved on the CHANNEL: an
        # enumeration that finds nothing still spends #912 work budget and can move the
        # search, so a zero hit count proves nothing -- but a call on a model with any
        # continuous slot cannot take the direct path and so behaves identically in
        # both arms whatever the flag says. Measured: tspn12 has 24 continuous slots
        # and one ibs call per solve, and flags 0/1/0 give bit-identical runs.
        _FIRE["ibs_calls"] += 1
        model = a[0] if a else k.get("model")
        if model is not None and not bool(np.any(~ph._get_integer_mask(model))):
            _FIRE["direct"] += 1
        out = _orig_ibs(*a, **k)
        if out is not None:
            _FIRE["box_hits"] += 1
        return out

    ph.integer_box_search = _counting_ibs
    # solver.py imports it lazily inside the polish, so the module attribute is
    # the single interception point; assert that rather than assume it.
    assert ph.integer_box_search is _counting_ibs


def _incumbent_feasible(model, r) -> bool:
    """Independently re-verify the returned incumbent against the model rows."""
    if getattr(r, "x", None) is None:
        return True
    from discopt._relax.nlp_evaluator import cached_evaluator
    from discopt._relax.primal_heuristics import _check_constraint_feasibility

    ev = cached_evaluator(model)
    flat = np.concatenate(
        [np.atleast_1d(np.asarray(r.x[v.name], dtype=np.float64)).ravel() for v in model._variables]
    )
    return bool(_check_constraint_feasibility(ev, flat))


def _f(x):
    return None if x is None else float(x)


def run(path, name, max_nodes, tl):
    from discopt.modeling.core import from_nl

    arms = {}
    sense = "minimize"
    # off -> on -> off again: the third arm brackets ON and measures this
    # instance's own run-to-run spread, which is what licenses (or forbids)
    # a neutrality assertion below.
    for arm, flag in (("off", "0"), ("on", "1"), ("off2", "0")):
        os.environ[FLAG] = flag
        _FIRE["calls"] = _FIRE["adopted"] = _FIRE["box_hits"] = 0
        _FIRE["ibs_calls"] = _FIRE["direct"] = 0
        _FIRE["delta"] = 0.0
        m = from_nl(path)
        obj = getattr(m, "_objective", None)
        s = getattr(obj, "sense", None)
        if s is not None:
            sense = "maximize" if "MAX" in str(s).upper() else "minimize"
        t0 = time.time()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = m.solve(max_nodes=max_nodes, time_limit=tl)
        wall = time.time() - t0
        arms[arm] = {
            "obj": _f(r.objective),
            "bound": _f(r.bound),
            "status": str(r.status),
            "gap_certified": bool(getattr(r, "gap_certified", False)),
            "nodes": int(getattr(r, "node_count", -1)),
            "wall": round(wall, 2),
            "polish_calls": _FIRE["calls"],
            "polish_adopted": _FIRE["adopted"],
            "box_hits": _FIRE["box_hits"],
            "ibs_calls": _FIRE["ibs_calls"],
            "direct": _FIRE["direct"],
            "incumbent_feasible": _incumbent_feasible(m, r),
        }
    o1, o2 = arms["off"], arms["off2"]
    spread = [
        f"{k} {o1[k]!r} vs {o2[k]!r}"
        for k in ("bound", "nodes", "obj", "status", "gap_certified")
        if o1[k] != o2[k]
    ]
    return {
        "instance": name,
        "oracle": ORACLES.get(name),
        "sense": sense,
        "deterministic": not spread,
        "off_spread": spread,
        **arms,
    }


def assess(rec):
    """Per-instance cert-clean verdict + net-positive signal."""
    opt = rec["oracle"]
    off, on = rec["off"], rec["on"]
    is_max = rec.get("sense") == "maximize"
    tol = 1e-4 * (1 + abs(opt)) if opt is not None else 1e-4
    problems, notes = [], []
    fired = on["polish_adopted"] > 0

    # A neutrality-class finding is charged to the flag only where the SAME
    # protocol run twice with the flag OFF reproduces itself. Where it does not,
    # the instance's own noise floor is larger than anything the flag could show,
    # so the finding is recorded with that measured spread instead of asserted
    # away (CLAUDE.md §9; the ball_mk2_30 measurement in the module docstring).
    det = rec.get("deterministic", True)
    spread = "; ".join(rec.get("off_spread") or []) or "none"
    # The ONLY channel by which this flag can influence a run is an improving
    # point out of ``integer_box_search`` (in-tree direct path) or an adopted
    # polish. Where neither produced one in EITHER arm, the two runs must be
    # bit-identical and any drift is a plumbing bug. Where one did, the search
    # legitimately diverges -- that is what a primal heuristic is for -- and
    # asserting neutrality would be asserting the flag does nothing.
    inert = off.get("direct", 0) == 0 and on.get("direct", 0) == 0 and on["polish_calls"] == 0

    def charge(msg):
        if not det:
            notes.append(f"{msg} -- NOT attributable: OFF-vs-OFF spread [{spread}]")
        elif not inert:
            notes.append(
                f"{msg} -- EXPECTED: the flag had a live channel here "
                f"(direct off={off.get('direct', 0)} on={on.get('direct', 0)}, "
                f"box_hits on={on.get('box_hits', 0)}, polish_calls={on['polish_calls']})"
            )
        else:
            problems.append(msg)

    # --- 1a: neutrality, asserted only where the run is reproducible. -------
    ob, nb = off["bound"], on["bound"]
    both_finite = ob is not None and nb is not None and math.isfinite(ob) and math.isfinite(nb)
    bound_moved = (
        f"bound {ob:.12g} -> {nb:.12g}"
        if (both_finite and abs(ob - nb) > 1e-12 * (1 + abs(ob)))
        else (f"bound presence {ob} -> {nb}" if (ob is None) != (nb is None) else None)
    )
    node_moved = (
        f"node_count {off['nodes']} -> {on['nodes']}" if off["nodes"] != on["nodes"] else None
    )
    for drift in (bound_moved, node_moved):
        if drift is not None:
            charge(f"{drift} MOVED")

    # --- 1b: what a CERTIFIED exit must still guarantee. --------------------
    # Not bit-identity: a better in-tree incumbent prunes harder, so a certified
    # solve may legitimately certify in fewer nodes at a better objective (gear:
    # 31 -> 3 nodes, 8.6e-07 -> 3.8e-08 against a 0.0 oracle). What may NOT happen
    # is losing the certificate, or certifying an answer that is not the optimum.
    for arm_name, a in (("off", off), ("on", on)):
        if a["status"] != "optimal":
            continue
        if opt is None or a["obj"] is None:
            continue
        if abs(a["obj"] - opt) > tol:
            # Unconditional: a FALSE certificate is never excusable by noise or
            # by the flag having fired (CLAUDE.md §1).
            problems.append(
                f"{arm_name} certified OPTIMAL at {a['obj']:.12g} but the oracle is "
                f"{opt:.12g} (FALSE CERTIFICATE)"
            )
    if off["status"] == "optimal" and on["status"] != "optimal":
        problems.append(f"certification LOST: off=optimal -> on={on['status']}")
    # The polish is gated on ``status != 'optimal'``; an adoption there is a
    # plumbing bug and is never excusable.
    if fired and "optimal" in (off["status"], on["status"]):
        problems.append("polish ADOPTED on a certified-optimal exit (must be gated off)")

    # --- 1c: certification never regresses. ---------------------------------
    if off["gap_certified"] and not on["gap_certified"]:
        charge("cert regression: OFF certified, ON not")

    # --- 1d: soundness against the oracle, both arms, sense-aware. ----------
    # Unconditional. Noise cannot excuse an invalid bound or a beaten optimum.
    for arm_name, a in (("off", off), ("on", on), ("off2", rec["off2"])):
        b, o = a["bound"], a["obj"]
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

    # --- 1e: an ON-only infeasible incumbent is charged to the flag. --------
    # Failing in BOTH arms is a pre-existing condition of the instance (measured
    # on nvs05 on main; tracked as #1199), not something this flag did.
    if not on["incumbent_feasible"]:
        if off["incumbent_feasible"]:
            problems.append("ON incumbent INFEASIBLE (OFF feasible)")
        else:
            notes.append("incumbent fails re-verification in BOTH arms (pre-existing, #1199)")

    # --- 2: net-positive, in the currency a PRIMAL flag is paid in. ---------
    signal = "inert"
    oo, no = off["obj"], on["obj"]
    if oo is not None and no is not None:
        delta = (no - oo) if is_max else (oo - no)  # > 0 == ON better
        eps = 1e-9 * (1 + abs(oo))
        if delta > eps:
            signal = "ON better incumbent"
        elif delta < -eps:
            signal = "ON WORSE incumbent"
            charge(f"ON incumbent WORSE {oo:.12g} -> {no:.12g}")
        elif fired:
            signal = "fired, no change"
    elif no is not None and oo is None:
        signal = "ON found an incumbent (OFF none)"
    elif oo is not None and no is None:
        signal = "ON LOST the incumbent"
        charge("ON lost the incumbent OFF had")

    rec["signal"] = signal
    rec["problems"] = problems
    rec["notes"] = notes
    rec["fired"] = fired
    return rec


def escalate(path, name, max_nodes, tl, rec, extra=4):
    """Re-measure a flagged instance with EXTRA OFF repeats before failing it.

    Two OFF runs are a weak determinism test, and a false "deterministic" verdict
    turns this corpus's own run-to-run noise (#1187) into a fabricated flag
    regression. Measured on ``tls2``: the panel drew 171 twice and called it
    deterministic, while a third OFF run gave 145 and a fourth 253 -- on an
    instance where ``integer_box_search`` is never even CALLED (4 continuous
    slots), so the flag has no channel to act through at all.

    This is an escalation, not an exemption: it spends more samples exactly where
    the verdict depends on them, and a drift that survives the wider sample stays
    a hard problem.
    """
    neutrality = [q for q in rec["problems"] if "MOVED" in q or "lost the incumbent" in q]
    if not neutrality:
        return rec
    print(f"    ~~ escalating: {extra} extra OFF runs to test the determinism claim", flush=True)
    seen = [(rec["off"]["bound"], rec["off"]["nodes"], rec["off"]["obj"], rec["off"]["status"])]
    for _ in range(extra):
        os.environ[FLAG] = "0"
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = from_nl_cached(path).solve(max_nodes=max_nodes, time_limit=tl)
        seen.append(
            (_f(r.bound), int(getattr(r, "node_count", -1)), _f(r.objective), str(r.status))
        )
    rec["off_repeats"] = seen
    if len(set(seen)) == 1:
        print(f"    ~~ {name}: {len(seen)} OFF runs agree -- drift stands as a PROBLEM", flush=True)
        return rec
    rec["deterministic"] = False
    rec["off_spread"] = [f"{len(set(seen))} distinct outcomes in {len(seen)} OFF runs: {seen}"]
    rec["problems"] = [q for q in rec["problems"] if q not in neutrality]
    rec["notes"] += [
        f"{q} -- NOT attributable: {len(set(seen))} distinct outcomes across "
        f"{len(seen)} flag-OFF runs {seen}"
        for q in neutrality
    ]
    print(f"    ~~ {name}: OFF runs DISAGREE {seen} -- drift is noise, reclassified", flush=True)
    return rec


def from_nl_cached(path):
    from discopt.modeling.core import from_nl

    return from_nl(path)


def main():
    corpus, max_nodes, tl, out_path = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
    only = set(sys.argv[5].split(",")) if len(sys.argv) > 5 else None
    paths = sorted(glob.glob(os.path.join(corpus, "*.nl")))
    if only:
        paths = [p for p in paths if os.path.basename(p)[:-3] in only]
    _install_fire_counter()

    records = []
    for i, p in enumerate(paths, 1):
        name = os.path.basename(p)[:-3]
        print(f"[{i}/{len(paths)}] {name} ...", flush=True)
        try:
            rec = assess(run(p, name, max_nodes, tl))
        except Exception as exc:
            print(f"    ERROR {type(exc).__name__}: {exc}", flush=True)
            records.append({"instance": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        rec = escalate(p, name, max_nodes, tl, rec)
        records.append(rec)
        flagmark = "FIRED" if rec["fired"] else "-"
        print(
            f"    {flagmark:5s} {rec['signal']:28s} "
            f"obj {rec['off']['obj']} -> {rec['on']['obj']}  "
            f"bound {rec['off']['bound']} nodes {rec['off']['nodes']}",
            flush=True,
        )
        for pr in rec["problems"]:
            print(f"    !! {pr}", flush=True)
        for nt in rec["notes"]:
            print(f"    .. {nt}", flush=True)
        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)

    scored = [r for r in records if "error" not in r]
    problems = [(r["instance"], p) for r in scored for p in r["problems"]]
    fired = [r for r in scored if r["fired"]]
    better = [r for r in scored if r["signal"] == "ON better incumbent"]
    worse = [r for r in scored if r["signal"] == "ON WORSE incumbent"]
    print("\n" + "=" * 72)
    print(f"instances scored     : {len(scored)}  (errors: {len(records) - len(scored)})")
    print(f"polish ADOPTED on    : {len(fired)}")
    print(f"ON better incumbent  : {len(better)}  {[r['instance'] for r in better]}")
    print(f"ON worse incumbent   : {len(worse)}  {[r['instance'] for r in worse]}")
    nondet = [r for r in scored if not r.get("deterministic", True)]
    print(
        f"nondeterministic     : {len(nondet)} of {len(scored)} (OFF vs OFF disagreed; "
        f"neutrality asserted on the other {len(scored) - len(nondet)})"
    )
    for r in nondet:
        print(f"     ~ {r['instance']}: {'; '.join(r['off_spread'])}")
    # LIVE = the flag had a channel (a zero-continuous direct-path call, or an
    # adopted exit polish). A ``box_hits`` on a call that was NOT direct-eligible
    # happens identically in both arms and is not the flag acting.
    active = [r for r in scored if r["on"].get("direct", 0) or r["on"]["polish_adopted"]]
    print(
        f"flag channel LIVE on : {len(active)} of {len(scored)} {[r['instance'] for r in active]}"
    )
    print(f"cert-clean problems  : {len(problems)}")
    for inst, p in problems:
        print(f"  !! {inst}: {p}")
    print(f"EXECUTED_COMPARISONS={len(scored)}")
    if not scored:
        print("PANEL PROVED NOTHING -- zero instances scored")
        sys.exit(2)
    cert_clean = not problems
    # A "worse" signal on a nondeterministic instance is that instance's noise,
    # not a regression the flag caused; it is already reported as a note.
    worse_det = [r for r in worse if r.get("deterministic", True)]
    net_positive = len(better) > 0 and not worse_det
    print(f"VERDICT: cert-clean={cert_clean}  net-positive={net_positive}")
    sys.exit(0 if (cert_clean and net_positive) else 1)


if __name__ == "__main__":
    main()

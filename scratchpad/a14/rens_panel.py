"""A14 graduation gate: the CLAUDE.md 5 differential panel for `DISCOPT_RENS`.

The entry experiment (`scratchpad/a14/rens_entry.py`) showed RENS *can* pay --
7/15 hits, median primal-gap reduction 80.3% -- on boxes built from a single
stored relaxation. That is not the same claim as "it pays inside a live search",
and the `DISCOPT_CUT_INHERIT` lesson is exactly the gap between the two: a
mechanism can be sound, cheap in isolation, and still neutral-or-harmful when it
is competing for the same budget as the tree it is helping.

DESIGN
------
* One build, two arms. Both arms are the SAME binary with `DISCOPT_RENS` unset
  vs `=1`, so nothing but the flag differs. The load gate below asserts the
  binary is an A14 build FUNCTIONALLY (a garbage flag value must be refused);
  a pre-A14 build accepts it silently, so this cannot pass against the wrong
  tree (CLAUDE.md 8).
* Interleaved, never sequential. For each instance the arms run back to back and
  the whole panel repeats, so machine drift lands on both arms of a pair rather
  than on whichever arm ran second (CLAUDE.md 9). Wall is reported with a
  spread, and the machine was NOT quiet during this run -- another session held
  a core throughout -- so no wall claim is load-clean and none is made. The
  bars below are stated on bounds and incumbents, which are load-independent
  VALUES; the time limit only decides how much work fits, and it fits into both
  arms of an interleaved pair alike.
* A time limit, not a node budget, is the shared budget on purpose. RENS's cost
  is sub-MIP nodes, and those are NOT counted in the parent's `nodes`, so a
  node-budgeted comparison would hand the ON arm free work and flatter it.

PRE-REGISTERED BARS (written before the run; CLAUDE.md 4 and 5).

BAR 1 -- CERT-CLEAN. Hard, zero slack, checked on BOTH arms:
  a) no dual bound above the reference optimum;
  b) no incumbent below the reference optimum;
  c) bound <= incumbent on every instance that has both;
  d) no certification regression: every instance the OFF arm drives to
     `optimal` must still be `optimal` in the ON arm;
  e) every ON-arm incumbent independently feasibility-verified against the
     ORIGINAL rows, bounds and integrality -- not taken on the solver's word.
  Any violation and the flag stays OFF and NO performance reading from this run
  is valid. This bar is never traded against bar 2.

BAR 2 -- NET-POSITIVE, over the population where the flag can matter (instances
not driven to `optimal` by BOTH arms), all three required:
  a) ON's median primal gap strictly below OFF's;
  b) incumbent WINS strictly greater than LOSSES;
  c) dual-bound LOSSES on at most 20% of the population -- the sub-MIP nodes
     must not be paid for out of the bound side.

KILL CRITERION: if 2a fails, or losses >= wins in 2b, or 2c fails, the flag
STAYS OFF, the measurement is recorded in `docs/dev/milp-competitiveness-plan.md`
and A14 is reported as sound-but-not-helpful. That is a publishable outcome, not
a failure of the run, and it is the outcome `DISCOPT_CUT_INHERIT` had.

Every count this script reports is an EXECUTED-COMPARISON count, and it exits
non-zero when any of them is zero, so "0 violations" can never come from a probe
that compared nothing (CLAUDE.md 6). Nothing here catches an exception: a probe
that swallows one reports a broken path as a healthy one (CLAUDE.md 7).
"""
import math
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import discopt  # noqa: E402
from discopt._rust import solve_milp_py  # noqa: E402
from loader import read_mps, to_engine  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TL = float(os.environ.get("RENS_TL", "20"))
REPS = int(os.environ.get("RENS_REPS", "2"))
BASE = dict(gap_tol=1e-4, max_nodes=10_000_000, gmi_cuts=True, root_cut_prune=True,
            time_limit_s=TL, root_cuts=500, cut_rounds=50, cut_select=True,
            root_cut_time_s=max(0.5, 0.5 * TL))
D_STATUSES = {"optimal", "feasible", "infeasible", "unbounded", "node_limit"}
HAS_INCUMBENT = {"optimal", "feasible", "node_limit"}


def load_gate():
    """Assert the binary under test is an A14 build, functionally (CLAUDE.md 8).

    A string marker in a source file proves nothing about which extension module
    got loaded. Refusing an unrecognized `DISCOPT_RENS` is behaviour only the A14
    driver has; every earlier build ignores the variable and solves.
    """
    print("discopt:", discopt.__file__, flush=True)
    c = np.array([-1.0, -2.0, 0.0])
    A = np.array([[1.0, 1.0, -1.0]])
    b = np.array([0.0])
    lo = np.array([0.0, 0.0, 0.0])
    up = np.array([3.0, 3.0, 4.0])
    ic = np.array([0, 1], dtype=np.int64)
    os.environ["DISCOPT_RENS"] = "not-a-flag-value"
    try:
        solve_milp_py(c, A, b, lo, up, ic, 2, 0.0, gap_tol=1e-6, max_nodes=1000)
    except BaseException as exc:            # noqa: BLE001 -- re-raised below
        if "DISCOPT_RENS" not in str(exc):
            sys.exit(f"LOAD GATE: unexpected failure, not the A14 refusal: {exc!r}")
    else:
        sys.exit("LOAD GATE: a garbage DISCOPT_RENS was ACCEPTED -- this is not an A14 build")
    finally:
        del os.environ["DISCOPT_RENS"]
    print("load gate: A14 build confirmed (garbage DISCOPT_RENS refused)", flush=True)


def verify_incumbent(x, d, eng, obj_reported, name):
    """Re-derive feasibility and objective from the ORIGINAL model. Returns a
    list of violation strings -- never an exception, and never silence."""
    c, A, b, lo, up, ic, ns, off, sgn = eng
    x = np.asarray(x, float)
    bad = []
    if x.shape[0] == A.shape[1]:
        z = x
    elif x.shape[0] == ns:
        # Rebuild the slacks the engine carries past `n_struct`.
        z = np.concatenate([x, d["A"] @ x])
    else:
        return [f"solution length {x.shape[0]} is neither n+m={A.shape[1]} nor n_struct={ns}"]
    r = float(np.max(np.abs(A @ z - b))) if A.shape[0] else 0.0
    if r > 1e-5:
        bad.append(f"row residual {r:.3e}")
    vlo = float(np.max(lo - z))
    vup = float(np.max(z - up))
    if max(vlo, vup) > 1e-6:
        bad.append(f"bound violation {max(vlo, vup):.3e}")
    if ic.size:
        vint = float(np.max(np.abs(z[ic] - np.round(z[ic]))))
        if vint > 1e-5:
            bad.append(f"integrality violation {vint:.3e}")
    recomputed = float(c @ z) + off
    if not math.isfinite(obj_reported) or abs(recomputed - obj_reported) > 1e-4 * (1 + abs(recomputed)):
        bad.append(f"objective {obj_reported} != recomputed {recomputed}")
    return [f"{name}: {m}" for m in bad]


def run(arm, eng, d, name):
    c, A, b, lo, up, ic, ns, off, sgn = eng
    if arm == "ON":
        os.environ["DISCOPT_RENS"] = "1"
    else:
        os.environ.pop("DISCOPT_RENS", None)
    t0 = time.perf_counter()
    st, x, obj, bound, nodes, iters = solve_milp_py(c, A, b, lo, up, ic, ns, off, **BASE)
    wall = time.perf_counter() - t0
    os.environ.pop("DISCOPT_RENS", None)
    if st not in D_STATUSES:
        sys.exit(f"UNKNOWN discopt status {st!r} on {name} ({arm})")
    has_inc = (st in HAS_INCUMBENT and obj is not None
               and math.isfinite(obj) and abs(obj) < 1e19)
    return dict(arm=arm, status=st, obj=obj if has_inc else None, bound=bound,
                nodes=nodes, wall=wall, has_inc=has_inc,
                x=(np.asarray(x, float) if (has_inc and x is not None) else None))


def gap(obj, bound, ref):
    """Primal gap against the reference optimum, in the panel's min sense.

    An instance with no incumbent at the limit is the strongest possible primal
    failure and scores 1.0 -- reported separately so the accounting is visible
    rather than buried in a median.
    """
    if obj is None:
        return 1.0
    den = max(abs(ref), 1e-6)
    return abs(obj - ref) / den


def main():
    load_gate()
    print("load at start:", os.popen("uptime").read().strip(), flush=True)
    import json
    panel = json.load(open(os.path.join(HERE, "panel.json")))
    # A smoke setting only. The graduation run leaves RENS_LIMIT unset; the
    # printed instance count below is what says which of the two happened, so a
    # truncated run can never be mistaken for the gate.
    lim = int(os.environ.get("RENS_LIMIT", "0"))
    if lim:
        panel = panel[:lim]
        print(f"*** SMOKE RUN: panel truncated to {lim} instances -- NOT the graduation gate ***",
              flush=True)

    reps = {}          # (name, arm) -> list of run dicts
    order = []
    for rep in range(REPS):
        for p in panel:
            f = os.path.join(HERE, "mps", p["name"] + ".mps.gz")
            if not os.path.exists(f):
                continue
            d = read_mps(f)
            try:
                eng = to_engine(d)
            finally:
                os.unlink(d["mps"])
            if rep == 0:
                order.append(p)
            # Interleaved: the pair runs back to back on the same loaded model.
            for arm in ("OFF", "ON"):
                r = run(arm, eng, d, p["name"])
                r["ref"] = p.get("opt_min")
                r["viol"] = (verify_incumbent(r["x"], d, eng, r["obj"], p["name"])
                             if (arm == "ON" and r["has_inc"]) else [])
                r["x"] = None                       # drop the vector; keep the verdict
                reps.setdefault((p["name"], arm), []).append(r)
            a, b_ = reps[(p["name"], "OFF")][-1], reps[(p["name"], "ON")][-1]
            print(f"  rep{rep} {p['name']:<22} OFF {a['status']:>10} obj={a['obj']!s:>12} "
                  f"bnd={a['bound']!s:>12} | ON {b_['status']:>10} obj={b_['obj']!s:>12} "
                  f"bnd={b_['bound']!s:>12}", flush=True)
    print("load at end:", os.popen("uptime").read().strip(), flush=True)

    if not order:
        sys.exit("VACUOUS: no instance ran")

    # ---- pick the representative replicate: the LAST one, and report spread ----
    def pick(name, arm):
        rs = reps[(name, arm)]
        return rs[-1], [r["wall"] for r in rs]

    # ================= BAR 1 -- CERT-CLEAN =================
    cert, checked_cert, verified_inc = [], 0, 0
    for p in order:
        ref = p.get("opt_min")
        for arm in ("OFF", "ON"):
            for r in reps[(p["name"], arm)]:
                if ref is not None:
                    tol = 1e-6 * (1 + abs(ref))
                    if r["bound"] is not None and r["bound"] > ref + tol:
                        cert.append(f"{p['name']} [{arm}] bound {r['bound']} above optimum {ref}")
                    if r["has_inc"] and r["obj"] < ref - 1e-4 * (1 + abs(ref)):
                        cert.append(f"{p['name']} [{arm}] incumbent {r['obj']} below optimum {ref}")
                    checked_cert += 1
                if r["has_inc"] and r["bound"] is not None and r["bound"] > r["obj"] + 1e-6 * (1 + abs(r["obj"])):
                    cert.append(f"{p['name']} [{arm}] bound {r['bound']} exceeds incumbent {r['obj']}")
                if arm == "ON" and r["has_inc"]:
                    verified_inc += 1
                    cert.extend(f"[ON feasibility] {v}" for v in r["viol"])
        # (d) certification regression
        off_last, _ = pick(p["name"], "OFF")
        on_last, _ = pick(p["name"], "ON")
        if off_last["status"] == "optimal" and on_last["status"] != "optimal":
            cert.append(f"{p['name']} CERTIFICATION REGRESSION: OFF optimal -> ON {on_last['status']}")

    print(f"\nBAR 1 -- CERT-CLEAN")
    print(f"  reference comparisons executed : {checked_cert}")
    print(f"  ON incumbents feasibility-verified against the original model : {verified_inc}")
    if checked_cert == 0:
        sys.exit("VACUOUS: no certificate comparison executed -- this probe measured nothing")
    if cert:
        for c_ in cert:
            print("  CERT VIOLATION", c_)
        sys.exit("BAR 1 FAILED -- flag stays OFF and no performance reading from this run is valid")
    print("  PASS: 0 violations over the comparisons above")

    # ================= BAR 2 -- NET-POSITIVE =================
    pop = []
    for p in order:
        off_last, off_w = pick(p["name"], "OFF")
        on_last, on_w = pick(p["name"], "ON")
        if p.get("opt_min") is None:
            continue
        if off_last["status"] == "optimal" and on_last["status"] == "optimal":
            continue                     # the flag cannot matter here
        pop.append((p["name"], p["opt_min"], off_last, on_last, off_w, on_w))

    print(f"\nBAR 2 -- NET-POSITIVE   population (not optimal in BOTH arms): {len(pop)}")
    if not pop:
        sys.exit("VACUOUS: no instance where the flag could matter; nothing was compared")

    print(f"\n{'instance':<22}{'OFF inc':>13}{'ON inc':>13}{'OFF gap%':>10}{'ON gap%':>10}"
          f"{'OFF bnd':>13}{'ON bnd':>13}")
    wins = losses = bnd_losses = bnd_wins = 0
    goff, gon = [], []
    for name, ref, o, n, _, _ in pop:
        g0, g1 = gap(o["obj"], o["bound"], ref), gap(n["obj"], n["bound"], ref)
        goff.append(g0)
        gon.append(g1)
        eps = 1e-9 * (1 + abs(ref))
        if n["obj"] is not None and (o["obj"] is None or n["obj"] < o["obj"] - eps):
            wins += 1
        elif o["obj"] is not None and (n["obj"] is None or n["obj"] > o["obj"] + eps):
            losses += 1
        if o["bound"] is not None and n["bound"] is not None:
            if n["bound"] < o["bound"] - eps:
                bnd_losses += 1
            elif n["bound"] > o["bound"] + eps:
                bnd_wins += 1
        f = lambda v: f"{v:13.6g}" if v is not None else f"{'-':>13}"
        print(f"{name:<22}{f(o['obj'])}{f(n['obj'])}{100*g0:10.2f}{100*g1:10.2f}"
              f"{f(o['bound'])}{f(n['bound'])}", flush=True)

    m0, m1 = statistics.median(goff), statistics.median(gon)
    print(f"\n  median primal gap  OFF {100*m0:.2f}%   ON {100*m1:.2f}%")
    print(f"  incumbent          wins {wins}   losses {losses}")
    print(f"  dual bound         wins {bnd_wins}   losses {bnd_losses}"
          f"   (cap: {int(0.2*len(pop))} of {len(pop)})")

    a2a = m1 < m0
    a2b = wins > losses
    a2c = bnd_losses <= 0.2 * len(pop)
    print(f"\n  2a median primal gap strictly better : {'PASS' if a2a else 'FAIL'}")
    print(f"  2b incumbent wins > losses           : {'PASS' if a2b else 'FAIL'}")
    print(f"  2c dual-bound losses <= 20%          : {'PASS' if a2c else 'FAIL'}")

    # Wall, reported with a spread and with the load caveat attached, never as a claim.
    tot_off = sum(o["wall"] for _, _, o, _, _, _ in pop)
    tot_on = sum(n["wall"] for _, _, _, n, _, _ in pop)
    spread_off = [statistics.pstdev(w) for _, _, _, _, w, _ in pop if len(w) > 1]
    spread_on = [statistics.pstdev(w) for _, _, _, _, _, w in pop if len(w) > 1]
    line = f"\n  wall over the population   OFF {tot_off:.1f}s   ON {tot_on:.1f}s"
    if spread_off and spread_on:
        line += (f"   (median per-instance sd over {REPS} reps: "
                 f"OFF {statistics.median(spread_off):.2f}s, ON {statistics.median(spread_on):.2f}s)")
    print(line)
    print("  NOT a timing claim: the machine was loaded throughout (another session held a core).")

    verdict = ("GRADUATE: both bars met -- DISCOPT_RENS may default ON"
               if (a2a and a2b and a2c) else
               "STAYS OFF: cert-clean but not net-positive -- record the measurement "
               "(the DISCOPT_CUT_INHERIT outcome)")
    print(f"\nVERDICT: {verdict}")
    json.dump({"tl": TL, "reps": REPS,
               "rows": {f"{k[0]}|{k[1]}": v for k, v in reps.items()},
               "median_gap_off": m0, "median_gap_on": m1,
               "wins": wins, "losses": losses,
               "bnd_wins": bnd_wins, "bnd_losses": bnd_losses,
               "bars": {"2a": a2a, "2b": a2b, "2c": a2c},
               "verdict": verdict},
              open(os.path.join(HERE, "rens_panel.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()

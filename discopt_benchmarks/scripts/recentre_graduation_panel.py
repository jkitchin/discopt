"""Graduation panel for ``DISCOPT_RECENTRE`` (#1537 workstream C; CLAUDE.md §5).

Flag OFF vs ON, interleaved per instance, over:

* **A** the in-repo MINLPLib corpus as written -- the pass must be *bound-neutral*
  wherever it moves nothing (identical status / objective / bound / node_count);
* **B** the same corpus under ``x = y - c`` (c ~ 1e3, 1e6) via the invariance
  harness;
* **C** the harness's generated families under the same shifts.

Every certificate is compared with the unshifted, flag-OFF certified base, and
every published incumbent is independently re-verified with ``verify_point`` on
the model that was solved. Reports, per arm: false certificates, incumbents that
fail verification, certificates lost relative to the other arm, and total wall.
Exits non-zero when it compared nothing (CLAUDE.md §6).

    python -u discopt_benchmarks/scripts/recentre_graduation_panel.py [--time-limit 20]

``--flag NAME`` A/Bs another default-off gate over the same three panels (e.g.
``DISCOPT_LIFT_AFFINE_MONOMIALS``); ``--hold NAME=VALUE`` pins a flag in both arms.
For a flag other than ``DISCOPT_RECENTRE`` the bound-neutrality check (panel A)
applies wherever the flag's static detector says it does not fire.
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TESTS = os.path.join(HERE, "..", "..", "python", "tests")
sys.path.insert(0, TESTS)

import discopt  # noqa: E402
import discopt.modeling as dm  # noqa: E402
from _invariance import certified_answer_changed, translate  # noqa: E402
from discopt.validation.feasibility import verify_point  # noqa: E402
from test_1537_invariance import FAMILIES  # noqa: E402

FLAG = "DISCOPT_RECENTRE"


def _affine_monomial_fires(model) -> bool:
    """Static detector for ``DISCOPT_LIFT_AFFINE_MONOMIALS``: does the reform's own
    scan (maximal ``*`` chains, the walk the prelift makes) find a translated
    monomial in the objective or a row?"""
    from discopt._relax import factorable_reform as fr

    exprs = [model._objective.expression] + [c.body for c in model._constraints]
    return any(fr._scan_for_translated_monomial(e) for e in exprs)


def _solve(model, flag: str, tl: float):  # model: freshly built by the caller
    os.environ[FLAG] = flag
    fires = FLAG == "DISCOPT_LIFT_AFFINE_MONOMIALS" and _affine_monomial_fires(model)
    t0 = time.perf_counter()
    try:
        r = model.solve(time_limit=tl)
        err = None
    except Exception as exc:  # recorded as an outcome, never hidden
        r, err = None, f"{type(exc).__name__}: {str(exc)[:80]}"
    wall = time.perf_counter() - t0
    bad_point = False
    if r is not None and r.x is not None and r.status in ("optimal", "feasible"):
        flat = np.concatenate(
            [np.ravel(np.asarray(r.x[v.name], dtype=np.float64)) for v in model._variables]
        )
        bad_point = not verify_point(model, flat).ok
    if r is not None and fires:
        r._panel_fires = True
    return r, err, wall, bad_point


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--time-limit", type=float, default=20.0)
    ap.add_argument("--flag", default="DISCOPT_RECENTRE")
    ap.add_argument("--hold", action="append", default=[], metavar="NAME=VALUE")
    ap.add_argument(
        "--from",
        dest="start",
        default="",
        metavar="NAME",
        help="resume: skip corpus instances sorting before NAME (sections A/B only)",
    )
    args = ap.parse_args()
    global FLAG
    FLAG = args.flag
    for kv in args.hold:
        k, v = kv.split("=", 1)
        os.environ[k] = v
    print(f"A/B flag {FLAG}; held {args.hold}", flush=True)
    tl = args.time_limit
    assert discopt.__file__.startswith(os.path.abspath(os.path.join(HERE, "..", "..", "python")))
    print(f"discopt from {discopt.__file__}; load {os.getloadavg()}", flush=True)

    corpus = sorted(glob.glob(os.path.join(TESTS, "data", "minlplib_nl", "*.nl")))
    corpus = [p for p in corpus if os.path.basename(p) >= args.start]
    assert corpus, f"--from {args.start!r} skips the whole corpus"
    tally = {
        arm: {"false": 0, "bad_point": 0, "raised": 0, "cert": 0, "wall": 0.0} for arm in ("0", "1")
    }
    lost = {"0": 0, "1": 0}  # certified in the OTHER arm but not this one
    neutral_checked = neutral_diff = compared = 0

    def record(label, base, make, neutral: bool):
        nonlocal compared, neutral_checked, neutral_diff
        out = {}
        for flag in ("0", "1"):  # interleaved: same instance, back to back
            # A fresh model per arm: a solve writes constraint-implied bounds back
            # onto its model (propagate_bounds_to_model), so reusing one object gave
            # each arm a different, already-tightened starting box.
            r, err, wall, bad = _solve(make(), flag, tl)
            t = tally[flag]
            t["wall"] += wall
            if err:
                t["raised"] += 1
            if bad:
                t["bad_point"] += 1
            if r is not None and r.gap_certified:
                t["cert"] += 1
            why = certified_answer_changed(base, r) if r is not None else ""
            if why:
                t["false"] += 1
                print(f"FALSE flag={flag} {label}: {why}", flush=True)
            out[flag] = (r, err)
        compared += 1
        r0, r1 = out["0"][0], out["1"][0]
        c0 = bool(r0 is not None and r0.gap_certified)
        c1 = bool(r1 is not None and r1.gap_certified)
        if c0 and not c1:
            lost["1"] += 1
            print(f"LOST flag=1 {label}", flush=True)
        if c1 and not c0:
            lost["0"] += 1
        if FLAG == "DISCOPT_RECENTRE":
            moved = r1 is not None and (r1.solver_stats or {}).get("recentre/variables_moved")
        else:
            moved = int(bool(getattr(r1, "_panel_fires", False)))
        note = ""
        finished = r0 is not None and r0.status in ("optimal", "infeasible")
        if neutral and not moved and finished and r1 is not None:
            neutral_checked += 1
            same = (r0.status, r0.node_count, r0.objective, r0.bound) == (
                r1.status,
                r1.node_count,
                r1.objective,
                r1.bound,
            )
            if not same:
                neutral_diff += 1
                note = "  NEUTRALITY DRIFT"

        def fmt(r, err):
            return err if err else f"{r.status}/{'C' if r.gap_certified else '-'} n={r.node_count}"

        print(
            f"{label:34s} OFF {fmt(*out['0']):28s} ON {fmt(*out['1']):28s}"
            f"{' moved=' + str(int(moved)) if moved else ''}{note}",
            flush=True,
        )

    for path in corpus:  # A + B
        name = os.path.basename(path)
        os.environ[FLAG] = "0"
        try:
            base = dm.from_nl(path).solve(time_limit=tl)
        except Exception as exc:
            print(f"{name}: base raised {type(exc).__name__}", flush=True)
            continue
        record(f"A {name}", base, lambda p=path: dm.from_nl(p), neutral=True)
        if not base.gap_certified:
            continue
        for off, seed in ((1e3, 1), (1e6, 2)):
            record(
                f"B {name} shift{off:g}",
                base,
                lambda p=path, o=off, sd=seed: translate(dm.from_nl(p), o, seed=sd),
                neutral=False,
            )

    for fam, build in FAMILIES.items():  # C
        for s in range(4):
            os.environ[FLAG] = "0"
            base = build(s).solve(time_limit=tl)
            if not base.gap_certified:
                continue
            for off, seed in ((1e3, 1), (1e6, 2)):
                record(
                    f"C {fam}[{s}] shift{off:g}",
                    base,
                    lambda b=build, i=s, o=off, sd=seed: translate(b(i), o, seed=sd),
                    False,
                )

    print(f"\nCOMPARED {compared}; bound-neutral checked {neutral_checked}, drifted {neutral_diff}")
    for flag, t in tally.items():
        print(
            f"flag={flag}: false={t['false']} bad_point={t['bad_point']} raised={t['raised']} "
            f"certified={t['cert']} lost_vs_other={lost[flag]} wall={t['wall']:.0f}s"
        )
    return 0 if compared else 1


if __name__ == "__main__":
    sys.exit(main())

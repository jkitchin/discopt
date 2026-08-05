"""Adversarial soundness probe for #930: can the re-admitted probe bound be WRONG?

The #930 change re-admits a bound the root LP probe proved. The catastrophic
failure mode is not "unhelpful" but "invalid": a dual bound past the true optimum
is a false certificate (CLAUDE.md Sec.1). The in-PR panel made only 11 oracle
checks, all at one time limit on 17 instances. This widens that enormously.

HYPOTHESIS: every value ``_admissible_probe_bound`` admits is a valid global bound.
KILL CRITERION: any admitted value V with V > opt (min) / V < opt (max), beyond
tolerance, against the MINLPLib ``=opt=`` oracle. One counterexample sinks the
unconditional half of the PR.

Two independent assertions per instance:
  (a) every ADMITTED value, checked at the moment of admission -- this tests the
      box-equality gate directly, independent of what ``max`` later did with it;
  (b) the REPORTED bound, which catches anything the merge did downstream.

Sec.6: counts admissions and exits non-zero if zero -- an all-declining run would
otherwise print "0 violations" and read as a pass while testing nothing.
Sec.7: no bare excepts around the probe; a worker crash is reported, not hidden.
Short time limit on purpose: it is when the fallback is starved that the #930
direct-report path fires, so short limits are the DISCRIMINATING regime, not a
convenience.
"""

import json
import os
import random
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

BENCH = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark")
NLDIR = os.path.join(BENCH, "minlplib/nl")
NEW = "/Users/jkitchin/projects/discopt"
TL = float(os.environ.get("ADV_TL", "4.0"))
LIMIT = int(os.environ.get("ADV_N", "300"))
WORKERS = int(os.environ.get("ADV_W", "3"))
HARD = float(os.environ.get("ADV_HARD", "90"))  # per-instance wall cap
SCR = os.path.dirname(os.path.abspath(__file__))

solu = {}
with open(os.path.join(BENCH, "minlplib.solu")) as fh:
    for line in fh:
        p = line.split()
        if len(p) >= 3 and p[0] == "=opt=":  # only exact optima are a valid oracle
            solu[p[1]] = float(p[2])

avail = sorted(n[:-3] for n in os.listdir(NLDIR) if n.endswith(".nl") and n[:-3] in solu)
random.Random(20260805).shuffle(avail)  # fixed seed: reproducible, not cherry-picked
INSTS = avail[:LIMIT]

WORKER = os.path.join(SCR, "_adv930_worker.py")
with open(WORKER, "w") as fh:
    fh.write(
        r"""
import json, sys
nl, tl = sys.argv[1], float(sys.argv[2])
import discopt.solver as S
assert hasattr(S, "_admissible_probe_bound"), "MARKER ABSENT: not the #930 tree"

# Record every value the gate ADMITS, at the moment it admits it.
admitted = []
_orig = S._admissible_probe_bound
def _spy(probe, root_lb, root_ub):
    v = _orig(probe, root_lb, root_ub)
    if v is not None:
        admitted.append(float(v))
    return v
S._admissible_probe_bound = _spy

from discopt.modeling.core import from_nl, ObjectiveSense
m = from_nl(nl)
sense = "max" if m._objective.sense == ObjectiveSense.MAXIMIZE else "min"
r = m.solve(time_limit=tl)
print("RESULT" + json.dumps(dict(
    sense=sense, status=r.status, admitted=admitted,
    obj=(None if r.objective is None else float(r.objective)),
    bound=(None if r.bound is None else float(r.bound)))))
"""
    )


def run(inst):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(NEW, "python")
    env.pop("DISCOPT_ROOT_PROBE_SEEDS_FALLBACK", None)
    try:
        p = subprocess.run(
            [sys.executable, "-u", WORKER, os.path.join(NLDIR, inst + ".nl"), repr(TL)],
            capture_output=True,
            text=True,
            timeout=HARD,
            cwd=NEW,
            env=env,
        )
    except subprocess.TimeoutExpired:
        # NOT a swallowed exception: the only class caught, counted below, and
        # printed in the summary. An instance that blows through its own
        # time_limit by >10x is a finding in its own right (arki0020 ran past
        # 240 s under time_limit=8), but it must not abort the soundness sweep.
        return inst, None, f"WALL-CAP: exceeded {HARD}s at time_limit={TL}s"

    for ln in p.stdout.splitlines():
        if ln.startswith("RESULT"):
            return inst, json.loads(ln[6:]), None
    return inst, None, (p.stderr or "")[-600:]


def violates(val, opt, sense):
    """True if `val` is not a valid dual bound for `opt`."""
    tol = 1e-4 * max(1.0, abs(opt))
    return val > opt + tol if sense == "min" else val < opt - tol


n_admit = n_report = n_done = n_err = 0
bad_admit, bad_report, errs = [], [], []
print(f"# adv930: {len(INSTS)} instances w/ =opt= oracle, TL={TL}s, {WORKERS} workers", flush=True)

n_timeout = 0
with ThreadPoolExecutor(max_workers=WORKERS) as ex:
    futs = [ex.submit(run, i) for i in INSTS]
    for fut in as_completed(futs):
        inst, r, err = fut.result()
        n_done += 1
        if r is None:
            n_err += 1
            if err and err.startswith("WALL-CAP"):
                n_timeout += 1
            errs.append(f"{inst}: {err.splitlines()[-1] if err.strip() else 'no RESULT'}")
            print(f"[{n_done:3d}/{len(INSTS)}] {inst:24s} WORKER-ERROR", flush=True)
            continue
        opt, sense = solu[inst], r["sense"]
        n_bad_before = len(bad_admit) + len(bad_report)
        for v in r["admitted"]:
            n_admit += 1
            # _admissible_probe_bound returns MIN-SPACE (a lower bound on -obj);
            # solver.py:12113 negates it for maximize. The instrument must apply
            # the same conversion -- comparing raw min-space values against a
            # max-sense oracle produced a spurious "VIOLATION" on syn10m02hfsg.
            signed = -v if sense == "max" else v
            if violates(signed, opt, sense):
                bad_admit.append(f"{inst}: admitted {signed!r} vs {sense} opt {opt!r}")
        if r["bound"] is not None:
            n_report += 1
            if violates(r["bound"], opt, sense):
                bad_report.append(f"{inst}: reported {r['bound']!r} vs {sense} opt {opt!r}")
        n_bad_here = len(bad_admit) + len(bad_report) - n_bad_before
        flag = "ADMIT" if r["admitted"] else "     "
        print(
            f"[{n_done:3d}/{len(INSTS)}] {inst:24s} {sense} {flag} "
            f"n_adm={len(r['admitted'])} bound={r['bound']} opt={opt}"
            f"{'  <<< VIOLATION' if n_bad_here else ''}",
            flush=True,
        )

print()
print(
    f"# instances solved      : {n_done - n_err} "
    f"({n_err} worker errors, of which {n_timeout} wall-cap overruns)"
)
print(f"# ADMITTED bounds checked: {n_admit}")
print(f"# REPORTED bounds checked: {n_report}")
print(f"# INVALID ADMITTED       : {len(bad_admit)}")
for s in bad_admit[:25]:
    print(f"    {s}")
print(f"# INVALID REPORTED       : {len(bad_report)}")
for s in bad_report[:25]:
    print(f"    {s}")
if errs:
    print(f"# worker errors ({len(errs)}), first 10:")
    for s in errs[:10]:
        print(f"    {s}")
with open(os.path.join(SCR, "adv930.json"), "w") as _fh:
    json.dump(
        {
            "n_admit": n_admit,
            "n_report": n_report,
            "bad_admit": bad_admit,
            "bad_report": bad_report,
        },
        _fh,
        indent=1,
    )

if n_admit == 0:
    print(
        "PROBE NEVER FIRED: the #930 gate admitted nothing; this run proves nothing",
        file=sys.stderr,
    )
    sys.exit(1)
if bad_admit or bad_report:
    print("KILL CRITERION MET: an invalid dual bound was found", file=sys.stderr)
    sys.exit(2)
print("# PASS: every admitted and reported bound is oracle-valid")

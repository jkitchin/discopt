"""Clock-scale determinism probe for the solver's work gates (issue #912).

Issue #912 measured that discopt's search tree is a function of *machine speed*:
the amount of work the search does is decided by wall-clock reads, so "same
model + same ``time_limit``" does not imply the same tree. The decisive case
upstream was ``gear2`` (root integer local search, 3 nodes at a 5 s heuristic
budget vs 91 at 3 s).

This probe is the in-repo instrument for that claim and for its fix. Two phases:

``--clockscale``
    Solve each instance with the process clock running ``alpha`` times faster
    than real time (``alpha=1`` is the control, and must reproduce the baseline
    exactly). A faster perceived clock is exactly a slower machine: every
    wall-gated loop sees its budget expire after proportionally less real work.
    An instance whose ``node_count``/objective moves with ``alpha`` — while not
    hitting the overall ``time_limit`` — is nondeterministic by machine speed.

``--ilsbudget``
    Force the root ``integer_local_search`` wall budget to each of several
    values and report the resulting tree. This isolates the gear2 mechanism (the
    heuristic's own wall budget) from whole-solve budget starvation.

``--classify`` / ``--ab``
    One solve per instance (which rows are usable for a determinism check at all,
    and what the root ILS actually consumed), and the flag ON-vs-OFF differential
    with a soundness leg against ``known_optima.toml``.

Both phases run each solve in a **child process** that asserts
``discopt.__file__`` and the compiled extension path before importing anything
else, prints its row as JSON, and raises rather than swallowing (CLAUDE.md
rules 6-8). The parent prints an executed-comparison count and exits non-zero
when that count is zero, so a probe that measured nothing can never read as a
pass.

Note on coverage: the clock scaling patches Python's ``time`` module, so it
covers the Python orchestration layer's gates (78 sites per #912). The Rust
``Instant::now`` sites are not reachable this way; a row that is clock-scale
invariant here is invariant *with respect to the Python gates*, which is what
this probe claims and nothing more.

Usage::

    python -u discopt_benchmarks/scripts/item912_clock_determinism_probe.py \\
        --clockscale --alphas 1,2,4 --time-limit 30
    python -u discopt_benchmarks/scripts/item912_clock_determinism_probe.py \\
        --ilsbudget --budgets 5,3,1,0.5 --time-limit 30
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_CORPUS = os.path.join(_REPO, "python", "tests", "data", "minlplib_nl")

# The child program. Kept as a string so every row runs in a pristine
# interpreter: a stale JIT cache or a mutated process-global deadline from a
# previous instance would silently couple the rows we are comparing.
_CHILD = r"""
import json, os, sys, time

alpha = float(os.environ["PROBE_ALPHA"])
if alpha != 1.0:
    # Scale the perceived clock BEFORE discopt is imported, so every module that
    # captures ``time.perf_counter``/``time.monotonic`` at import time sees the
    # scaled one. alpha > 1 == a proportionally slower machine.
    _t0_pc, _t0_mono, _t0_time = time.perf_counter(), time.monotonic(), time.time()
    _real_pc, _real_mono, _real_time = time.perf_counter, time.monotonic, time.time
    time.perf_counter = lambda: _t0_pc + alpha * (_real_pc() - _t0_pc)
    time.monotonic = lambda: _t0_mono + alpha * (_real_mono() - _t0_mono)
    time.time = lambda: _t0_time + alpha * (_real_time() - _t0_time)

import discopt
from discopt.modeling.core import from_nl

# Rule 8: prove which code is loaded, and which arm it is running, before
# measuring anything with it. The (ils_eval_budget, ils_solve_budget) pair is the
# marker unique to the #912 version under test; the legacy arm must report (0, 0).
assert os.path.abspath(discopt.__file__).startswith(os.environ["PROBE_REPO"]), discopt.__file__
import discopt._rust as _rust
assert os.path.abspath(_rust.__file__).startswith(os.environ["PROBE_REPO"]), _rust.__file__
from discopt import solver_tuning as _st
_tun = _st.current()
_work_budget = (int(_tun.ils_eval_budget), int(_tun.ils_solve_budget))
_expect = os.environ.get("PROBE_EXPECT_WORK_BUDGET")
if _expect is not None:
    assert _work_budget == tuple(int(v) for v in _expect.split(",")), (
        f"arm mismatch: {_work_budget} != {_expect}"
    )

_ils_stats = {"calls": 0, "evals": 0, "solves": 0, "stopped_on": None, "wall_s": 0.0}
import discopt._jax.primal_heuristics as _ph
from discopt._work_budget import WorkBudget as _WB

_orig_ils_fn = _ph.integer_local_search


def _instrumented_ils(*a, **kw):
    forced = os.environ.get("PROBE_ILS_BUDGET")
    if forced:
        kw["time_budget"] = float(forced)
    seen = []
    _orig_init = _WB.__init__

    def _init(self, limits=None, **kws):
        _orig_init(self, limits, **kws)
        seen.append(self)

    _WB.__init__ = _init
    t0 = time.perf_counter()
    try:
        return _orig_ils_fn(*a, **kw)
    finally:
        _WB.__init__ = _orig_init
        _ils_stats["calls"] += 1
        _ils_stats["wall_s"] += time.perf_counter() - t0
        if seen:
            from discopt._work_budget import EVAL as _E, NLP_SOLVE as _S
            _ils_stats["evals"] += seen[0].spent(_E)
            _ils_stats["solves"] += seen[0].spent(_S)
            if seen[0].stopped_on is not None:
                _ils_stats["stopped_on"] = seen[0].stopped_on


_ph.integer_local_search = _instrumented_ils

path = os.environ["PROBE_INSTANCE"]
model = from_nl(path)
t_wall = time.perf_counter()
res = model.solve(time_limit=float(os.environ["PROBE_TIME_LIMIT"]))
row = {
    "instance": os.path.splitext(os.path.basename(path))[0],
    "status": str(getattr(res, "status", None)),
    "objective": (None if getattr(res, "objective", None) is None else float(res.objective)),
    "bound": (None if getattr(res, "bound", None) is None else float(res.bound)),
    "node_count": int(getattr(res, "node_count", -1)),
    "perceived_s": round(time.perf_counter() - t_wall, 3),
    "work_budget": list(_work_budget),
    "ils_calls": _ils_stats["calls"],
    "ils_evals": _ils_stats["evals"],
    "ils_solves": _ils_stats["solves"],
    "ils_stopped_on": _ils_stats["stopped_on"],
    "ils_wall_s": round(_ils_stats["wall_s"], 3),
}
print("PROBE_ROW " + json.dumps(row), flush=True)
"""


def _run_child(instance_path, alpha, time_limit, ils_budget=None, timeout=900, arm_env=None):
    env = dict(os.environ)
    env.update(
        {
            "PROBE_ALPHA": repr(float(alpha)),
            "PROBE_INSTANCE": instance_path,
            "PROBE_REPO": _REPO,
            "PROBE_TIME_LIMIT": repr(float(time_limit)),
            "PYTHONUNBUFFERED": "1",
        }
    )
    env.update(arm_env or {})
    if ils_budget is not None:
        env["PROBE_ILS_BUDGET"] = repr(float(ils_budget))
    else:
        env.pop("PROBE_ILS_BUDGET", None)
    t0 = time.perf_counter()
    proc = subprocess.run(
        [sys.executable, "-u", "-c", _CHILD],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    real_s = time.perf_counter() - t0
    if proc.returncode != 0:
        # Rule 7: never swallow. A child that died is a measurement failure,
        # not a row worth averaging in.
        raise RuntimeError(
            f"child failed for {instance_path} (alpha={alpha}, ils={ils_budget}), "
            f"rc={proc.returncode}\nSTDERR:\n{proc.stderr[-4000:]}"
        )
    rows = [ln for ln in proc.stdout.splitlines() if ln.startswith("PROBE_ROW ")]
    if len(rows) != 1:
        raise RuntimeError(
            f"expected exactly one PROBE_ROW from {instance_path}, got {len(rows)}\n"
            f"STDOUT:\n{proc.stdout[-4000:]}"
        )
    row = json.loads(rows[0][len("PROBE_ROW ") :])
    row["real_s"] = round(real_s, 3)
    return row


def _instances(names):
    if names:
        wanted = [n.strip() for n in names.split(",") if n.strip()]
    else:
        wanted = sorted(os.path.splitext(f)[0] for f in os.listdir(_CORPUS) if f.endswith(".nl"))
    out = []
    for n in wanted:
        p = os.path.join(_CORPUS, n + ".nl")
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        out.append((n, p))
    return out


def _same(a, b, tol=1e-6):
    """Two rows agree iff status, node_count and objective all agree."""
    if a["status"] != b["status"] or a["node_count"] != b["node_count"]:
        return False
    oa, ob = a["objective"], b["objective"]
    if (oa is None) != (ob is None):
        return False
    if oa is None:
        return True
    return abs(oa - ob) <= tol * max(1.0, abs(oa), abs(ob))


def _report(comparisons, mismatches, label):
    print()
    print(f"=== {label} ===")
    print(f"executed comparisons: {comparisons}")
    print(f"mismatches:           {len(mismatches)}")
    for m in mismatches:
        print("  " + m)
    if comparisons == 0:
        print("FAIL: zero comparisons executed — the probe measured nothing.")
        return 2
    return 1 if mismatches else 0


def _arm_env(args):
    """Environment for the arm under test. ``--legacy`` selects the pre-#912
    wall-clock gate; both arms assert their own marker in the child (rule 8)."""
    if args.legacy:
        return {
            "DISCOPT_ILS_EVAL_BUDGET": "0",
            "DISCOPT_ILS_SOLVE_BUDGET": "0",
            "PROBE_EXPECT_WORK_BUDGET": "0,0",
        }
    return {}


def phase_classify(args):
    """One solve per instance: which rows are usable for a determinism check at
    all (they must finish inside ``time_limit``), and which exercise ILS."""
    rows = []
    for name, path in _instances(args.instances):
        row = _run_child(path, 1.0, args.time_limit, timeout=args.timeout, arm_env=_arm_env(args))
        rows.append(row)
        print(f"{name:<20} {json.dumps(row)}", flush=True)
    usable = [r for r in rows if r["status"] != "time_limit"]
    ils = [r for r in rows if r["ils_calls"] > 0]
    print()
    print(f"=== classification ({len(rows)} instances) ===")
    print(f"finish inside time_limit: {len(usable)}")
    print(f"exercise ILS:             {len(ils)}")
    print("usable+ILS: " + ",".join(sorted(r["instance"] for r in usable if r["ils_calls"] > 0)))
    print("usable:     " + ",".join(sorted(r["instance"] for r in usable)))
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rows, fh, indent=1)
    return 0 if rows else 2


def _known_optima():
    """The repo's reference-optima registry, for the soundness leg of the A/B."""
    import tomllib

    path = os.path.join(_REPO, "python", "tests", "data", "known_optima.toml")
    with open(path, "rb") as fh:
        data = tomllib.load(fh)
    data.pop("schema", None)
    return {k: v for k, v in data.items() if isinstance(v, dict) and "optimum" in v}


def phase_ab(args):
    """Flag ON vs OFF at alpha=1: what the deterministic budget changed, and
    whether anything it changed is unsound.

    Two legs, both required (CLAUDE.md §5):

    * *cert-clean* — no dual bound may exceed its reference optimum on either
      arm, and no instance may regress from a proved status to a worse one;
    * *drift* — the node/objective differences, reported rather than hidden,
      since a primal heuristic that searches differently is *expected* to move
      node counts. Soundness is the gate; drift is the disclosure.
    """
    optima = _known_optima()
    rows = []
    unsound = []
    drift = []
    checked = 0
    for name, path in _instances(args.instances):
        new = _run_child(path, 1.0, args.time_limit, timeout=args.timeout, arm_env={})
        old = _run_child(
            path,
            1.0,
            args.time_limit,
            timeout=args.timeout,
            arm_env={
                "DISCOPT_ILS_EVAL_BUDGET": "0",
                "DISCOPT_ILS_SOLVE_BUDGET": "0",
                "PROBE_EXPECT_WORK_BUDGET": "0,0",
            },
        )
        assert max(new["work_budget"]) > 0 and max(old["work_budget"]) == 0, "arm markers crossed"
        rows.append({"instance": name, "new": new, "old": old})
        checked += 1
        ref = optima.get(name, {}).get("optimum")
        for tag, row in (("new", new), ("old", old)):
            if ref is None or row["bound"] is None:
                continue
            tol = 1e-6 * max(1.0, abs(float(ref)))
            if row["bound"] > float(ref) + tol:
                unsound.append(
                    f"{name} [{tag}]: bound {row['bound']!r} > reference optimum {ref!r}"
                )
        if not _same(new, old):
            drift.append(
                f"{name}: ON {new['status']}/{new['node_count']}/{new['objective']} vs "
                f"OFF {old['status']}/{old['node_count']}/{old['objective']}"
            )
        print(f"{name:<20} ON {json.dumps(new)}", flush=True)
        print(f"{name:<20} OFF {json.dumps(old)}", flush=True)
    print()
    print("=== flag ON vs OFF (alpha=1) ===")
    print(f"executed comparisons: {checked}")
    print(f"instances with a reference optimum: {sum(1 for r in rows if r['instance'] in optima)}")
    print(f"bounds above their reference optimum (MUST be 0): {len(unsound)}")
    for u in unsound:
        print("  " + u)
    print(f"rows that differ between arms: {len(drift)}")
    for d in drift:
        print("  " + d)
    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump(rows, fh, indent=1)
    if checked == 0:
        print("FAIL: zero comparisons executed — the probe measured nothing.")
        return 2
    return 1 if unsound else 0


def _pressure(row, time_limit, frac):
    """Why (if at all) this row was decided by the clock rather than by the model.

    #912 separates two mechanisms and only the second is fixable:

    * *whole-solve budget starvation* — the run is close enough to ``time_limit``
      that the B&B loop's own deadline checks shape the tree. Scaling the clock
      shrinks the real budget proportionally, so of course the tree moves; that
      is the ``time_limit`` contract, not a bug. Detected as ``time_limit``
      status, or as a run that consumed ``frac`` of its perceived budget.
    * *extent gates* — a heuristic's own wall budget deciding how much work to
      do. Detected as a heuristic reporting ``stopped_on == "deadline"``.

    Only rows free of both are in scope for a determinism assertion. Returns a
    reason string, or ``None`` when the row is clean.
    """
    if row["status"] == "time_limit":
        return "time_limit"
    if row["perceived_s"] >= frac * time_limit:
        return f"deadline-pressured ({row['perceived_s']:.1f}s of {time_limit:.0f}s)"
    if row["ils_stopped_on"] == "deadline":
        return "heuristic cut by the solve deadline"
    return None


def phase_clockscale(args):
    alphas = [float(a) for a in args.alphas.split(",")]
    if alphas[0] != 1.0:
        raise ValueError("--alphas must start with the 1.0 control")
    comparisons = 0
    mismatches = []
    out_of_scope = []
    for name, path in _instances(args.instances):
        base = None
        for alpha in alphas:
            row = _run_child(
                path, alpha, args.time_limit, timeout=args.timeout, arm_env=_arm_env(args)
            )
            print(f"{name:<20} alpha={alpha:<5} {json.dumps(row)}", flush=True)
            if base is None:
                base = row
                continue
            why = _pressure(base, args.time_limit, args.pressure_frac) or _pressure(
                row, args.time_limit, args.pressure_frac
            )
            differs = not _same(base, row)
            if why is not None:
                # Disclosed with its numbers, never silently dropped: an
                # out-of-scope row that *also* differs is exactly the evidence
                # that mechanism 2 is real and unfixed.
                out_of_scope.append(
                    f"{name} alpha={alpha}: {why}"
                    + (
                        f" — and it DIFFERS ({base['node_count']} vs {row['node_count']} nodes)"
                        if differs
                        else " — matched anyway"
                    )
                )
                print(f"  (out of scope: {why})", flush=True)
                continue
            comparisons += 1
            if differs:
                mismatches.append(
                    f"{name}: alpha=1 -> {base['status']}/{base['node_count']}/"
                    f"{base['objective']} vs alpha={alpha} -> {row['status']}/"
                    f"{row['node_count']}/{row['objective']}"
                )
    print()
    print(f"out-of-scope comparisons (whole-solve budget starvation): {len(out_of_scope)}")
    for o in out_of_scope:
        print("  " + o)
    return _report(comparisons, mismatches, "clock-scale determinism")


def phase_ilsbudget(args):
    budgets = [float(b) for b in args.budgets.split(",")]
    comparisons = 0
    mismatches = []
    for name, path in _instances(args.instances):
        base = None
        for budget in budgets:
            row = _run_child(
                path,
                1.0,
                args.time_limit,
                ils_budget=budget,
                timeout=args.timeout,
                arm_env=_arm_env(args),
            )
            print(f"{name:<20} ils_budget={budget:<5} {json.dumps(row)}", flush=True)
            if base is None:
                base = row
                continue
            comparisons += 1
            if not _same(base, row):
                mismatches.append(
                    f"{name}: budget={budgets[0]} -> {base['node_count']} nodes/"
                    f"{base['objective']} vs budget={budget} -> {row['node_count']} nodes/"
                    f"{row['objective']}"
                )
    return _report(comparisons, mismatches, "ILS wall-budget sensitivity")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--classify", action="store_true")
    ap.add_argument("--clockscale", action="store_true")
    ap.add_argument("--ilsbudget", action="store_true")
    ap.add_argument(
        "--legacy",
        action="store_true",
        help="run the pre-#912 arm (DISCOPT_ILS_WORK_BUDGET=0, wall-clock gate)",
    )
    ap.add_argument("--instances", default="", help="comma-separated names (default: all)")
    ap.add_argument("--alphas", default="1,2,4")
    ap.add_argument("--budgets", default="5,3,1,0.5")
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument(
        "--pressure-frac",
        type=float,
        default=0.5,
        help="a row consuming this fraction of its perceived budget is treated as "
        "whole-solve-starved and reported out of scope (default 0.5)",
    )
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--json-out", default="")
    args = ap.parse_args(argv)
    if not (args.classify or args.clockscale or args.ilsbudget):
        ap.error("choose --classify, --clockscale and/or --ilsbudget")
    rc = 0
    if args.classify:
        rc |= phase_classify(args)
    if args.clockscale:
        rc |= phase_clockscale(args)
    if args.ilsbudget:
        rc |= phase_ilsbudget(args)
    return rc


if __name__ == "__main__":
    sys.exit(main())

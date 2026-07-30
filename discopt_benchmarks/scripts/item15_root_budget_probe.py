"""Open-ledger item 15 — why does the Regime-N panel flag a *different* instance each run?

Two runs of the same tree each failed ``panel_baseline.py --check`` on a different
single instance (``ex1266``, then ``gear2``), and each flagged instance reproduced
the frozen baseline bit-identically when re-run alone (3/3 and 5/5). That makes the
Regime-N gate unusable: a real drift and a substrate flake are indistinguishable.

This probe asks *which* mechanism actually fires, with executed counts (CLAUDE.md
§6) and no swallowed exceptions (§7). Three arms:

``--arm observe``
    Solve each instance on **default settings** with ``obbt_tighten_root`` wrapped,
    recording the deadline it was handed, the wall it consumed, and whether it
    returned at-or-past that deadline (i.e. it broke out of the candidate sweep at
    ``obbt.py:1199/1229`` rather than converging). Also records the whole-solve
    slack ``wall / budget`` — the *other* candidate mechanism (plain budget
    starvation, which is what run A's ``ex1266`` looks like).

``--arm forcebudget``
    The causal arm. Re-solve one instance repeatedly with the root OBBT deadline
    **forced** to a range of values, everything else identical. If ``node_count``
    moves with the OBBT budget alone, then any process that changes how much OBBT
    fits in its wall budget — i.e. ambient load — changes the gated node count.
    This is the falsifiable claim; if node_count is flat across budgets, the
    hypothesis is dead.

``--arm load``
    The reproduction arm. Solve one instance on defaults N times with nothing else
    running, then N times against ``--load K`` busy processes, and report the
    node_count multiset in each condition.

Every arm prints an executed-observation count and exits non-zero when it is zero.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
if str(_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(_BENCH_ROOT))

from scripts.panel_baseline import corpus_instances, instance_path  # noqa: E402

_REPORTS = _REPO_ROOT / "reports"

# The PyO3 string unique to the newest Rust commit; CLAUDE.md §8 — assert the
# binary under test is the one we think it is, and crash loudly if not.
_SO_MARKER = "subtol_crossings_repaired"


def _assert_loaded_build() -> dict:
    import discopt  # noqa: PLC0415

    so = sorted(Path(discopt.__file__).parent.glob("_rust*.so"))
    if not so:
        raise SystemExit("REFUSED: no _rust*.so next to the loaded discopt")
    blob = so[0].read_bytes()
    if _SO_MARKER.encode() not in blob:
        raise SystemExit(f"REFUSED: {so[0]} lacks marker {_SO_MARKER!r} — stale build")
    return {"discopt_file": discopt.__file__, "rust_so": str(so[0]), "marker": _SO_MARKER}


# --------------------------------------------------------------------------- #
# Child: one solve, one JSON line.                                            #
# --------------------------------------------------------------------------- #
def _child(
    instance: str,
    budget: float,
    force_obbt: float | None,
    force_ils: float | None = None,
    alpha: float = 1.0,
) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")

    # ---- contention emulation (clock scaling) --------------------------------
    # Every wall-clock-budgeted phase computes its deadline as
    # ``perf_counter() + budget`` and polls ``perf_counter()``. Running the clock
    # ``alpha`` times faster is therefore indistinguishable, from the solver's
    # point of view, from running on a machine ``alpha`` times slower — which is
    # what ambient load does. This reaches ALL 78 Python wall-clock branch sites
    # at once instead of one hand-picked phase. (It does NOT reach the ~50
    # ``Instant::now`` sites in ``crates/``, so it measures a LOWER bound on
    # budget sensitivity.) No module does ``from time import perf_counter``, so
    # patching the module attribute is sufficient — asserted below.
    _real_perf = time.perf_counter
    if alpha != 1.0:
        _t0 = _real_perf()
        _real_mono = time.monotonic
        _m0 = _real_mono()
        time.perf_counter = lambda: _t0 + (_real_perf() - _t0) * alpha  # type: ignore[assignment]
        time.monotonic = lambda: _m0 + (_real_mono() - _m0) * alpha  # type: ignore[assignment]

    build = _assert_loaded_build()
    from discopt._jax import obbt as _obbt_mod  # noqa: PLC0415
    from discopt._jax import primal_heuristics as _ph_mod  # noqa: PLC0415
    from discopt.modeling.core import from_nl  # noqa: PLC0415

    # ---- root primal heuristic (integer_local_search) ------------------------
    # solver/__init__.py:9660 hands it ``time_budget=min(5.0, 0.15*time_limit)``
    # and the descent loops ``while improved and perf_counter() < deadline``
    # (primal_heuristics.py:768). How many moves fit is a function of machine
    # speed, so this arm forces the budget to make that dependence causal.
    ils_calls: list[dict] = []
    _orig_ils = _ph_mod.integer_local_search

    def _wrapped_ils(*a, **kw):
        given = kw.get("time_budget")
        if force_ils is not None:
            kw["time_budget"] = float(force_ils)
        t_in = time.perf_counter()
        res = _orig_ils(*a, **kw)
        ils_calls.append(
            {
                "budget_original": None if given is None else float(given),
                "budget_handed": kw.get("time_budget"),
                "consumed": time.perf_counter() - t_in,
                "found": res is not None,
                "objective": None if res is None else float(res[1]),
            }
        )
        return res

    _ph_mod.integer_local_search = _wrapped_ils

    calls: list[dict] = []
    _orig = _obbt_mod.obbt_tighten_root

    def _wrapped(*a, **kw):
        # The solve path passes ``deadline=`` as a keyword (solver/__init__.py:6586).
        given = kw.get("deadline")
        t_in = time.perf_counter()
        if force_obbt is not None:
            kw["deadline"] = t_in + float(force_obbt)
        used_deadline = kw.get("deadline")
        res = _orig(*a, **kw)
        t_out = time.perf_counter()
        calls.append(
            {
                "budget_handed": None if used_deadline is None else used_deadline - t_in,
                "budget_original": None if given is None else given - t_in,
                "consumed": t_out - t_in,
                # Returned at or past its deadline => it broke out of the candidate
                # sweep on the clock, not on convergence.
                "deadline_hit": bool(used_deadline is not None and t_out >= used_deadline),
                "n_tightened": int(getattr(res, "n_tightened", -1)),
                "infeasible": bool(getattr(res, "infeasible", False)),
            }
        )
        return res

    _obbt_mod.obbt_tighten_root = _wrapped

    out: dict = {
        "instance": instance,
        "budget": budget,
        "force_obbt": force_obbt,
        "force_ils": force_ils,
        "alpha": alpha,
        **build,
    }
    model = from_nl(str(instance_path(instance)))
    t0 = _real_perf()
    r = model.solve(time_limit=budget)
    out["wall"] = _real_perf() - t0
    out["status"] = str(r.status)
    out["node_count"] = int(r.node_count)
    out["objective"] = None if r.objective is None else float(r.objective)
    out["bound"] = None if r.bound is None else float(r.bound)
    out["root_bound"] = None if r.root_bound is None else float(r.root_bound)
    out["root_time"] = None if r.root_time is None else float(r.root_time)
    out["gap_certified"] = bool(r.gap_certified)
    out["obbt_calls"] = calls
    out["obbt_call_count"] = len(calls)
    out["obbt_deadline_hits"] = sum(1 for c in calls if c["deadline_hit"])
    out["ils_calls"] = ils_calls
    out["ils_call_count"] = len(ils_calls)
    print("RESULT_JSON " + json.dumps(out), flush=True)
    return 0


def _solve(
    instance: str,
    budget: float,
    force_obbt: float | None = None,
    force_ils: float | None = None,
    alpha: float = 1.0,
) -> dict:
    cmd = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        "--child",
        instance,
        str(budget),
        "none" if force_obbt is None else str(force_obbt),
        "none" if force_ils is None else str(force_ils),
        str(alpha),
    ]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=budget + 180.0, env=env)
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    # No swallowed failures (§7): a child that produced nothing is a hard error.
    raise SystemExit(
        f"child produced no RESULT_JSON for {instance} (rc={proc.returncode})\n"
        f"stdout tail:\n{proc.stdout[-2000:]}\nstderr tail:\n{proc.stderr[-2000:]}"
    )


def _load1() -> float:
    try:
        return float(os.getloadavg()[0])
    except (OSError, AttributeError):
        return float("nan")


# --------------------------------------------------------------------------- #
# Arms                                                                        #
# --------------------------------------------------------------------------- #
def arm_observe(args) -> int:
    insts = _pick(args)
    print(
        f"observe: {len(insts)} instance(s) @ {args.budget:.0f}s, load1={_load1():.2f}",
        flush=True,
    )
    rows = []
    for i, inst in enumerate(insts, 1):
        r = _solve(inst, args.budget)
        rows.append(r)
        print(
            f"  [{i:3d}/{len(insts)}] {inst:24s} {r['status']:11s} nodes={r['node_count']:>7d} "
            f"cert={'Y' if r['gap_certified'] else '.'} wall={r['wall']:6.1f} "
            f"root={0.0 if r['root_time'] is None else r['root_time']:6.2f} "
            f"obbt_calls={r['obbt_call_count']} hits={r['obbt_deadline_hits']} "
            f"budget_frac={r['wall'] / args.budget:.2f}",
            flush=True,
        )
    n_obs = len(rows)
    n_obbt = sum(r["obbt_call_count"] for r in rows)
    n_hit = sum(r["obbt_deadline_hits"] for r in rows)
    hit_rows = [r["instance"] for r in rows if r["obbt_deadline_hits"]]
    print("\n" + "=" * 78, flush=True)
    print(f"observations executed : {n_obs}", flush=True)
    print(f"obbt_tighten_root calls observed : {n_obbt}", flush=True)
    print(f"calls that returned at/past their deadline : {n_hit}", flush=True)
    print(f"instances with >=1 deadline-bound root OBBT : {len(hit_rows)} {hit_rows}", flush=True)
    print("=" * 78, flush=True)
    _write(
        args.out,
        {
            "arm": "observe",
            "budget": args.budget,
            "rows": rows,
            "observations": n_obs,
            "obbt_calls": n_obbt,
            "obbt_deadline_hits": n_hit,
            "deadline_bound_instances": hit_rows,
        },
    )
    if n_obs == 0:
        print("FAIL: zero observations executed.", flush=True)
        return 3
    return 0


def arm_forcebudget(args) -> int:
    budgets = [float(x) for x in args.budgets.split(",")]
    phase = args.phase
    insts = _pick(args)
    print(
        f"forcebudget[{phase}]: {len(insts)} instance(s) x {len(budgets)} budget(s) "
        f"x {args.reps} rep(s) @ {args.budget:.0f}s, load1={_load1():.2f}",
        flush=True,
    )
    rows = []
    for inst in insts:
        for ob in budgets:
            for rep in range(args.reps):
                kw = {"force_obbt": ob} if phase == "obbt" else {"force_ils": ob}
                r = _solve(inst, args.budget, **kw)
                r["rep"] = rep
                r["forced"] = ob
                rows.append(r)
                print(
                    f"  {inst:20s} {phase}_budget={ob:5.2f} rep={rep} -> "
                    f"{r['status']:11s} nodes={r['node_count']:>7d} "
                    f"obj={r['objective']} bound={r['bound']} "
                    f"root_bound={r['root_bound']} "
                    f"obbt_hits={r['obbt_deadline_hits']} "
                    f"ils={r['ils_call_count']} wall={r['wall']:.1f}",
                    flush=True,
                )
    # Per-instance: does node_count vary across the forced budgets?
    verdict = {}
    for inst in insts:
        seen = {}
        for r in rows:
            if r["instance"] == inst:
                seen.setdefault(r["forced"], set()).add(r["node_count"])
        distinct = {n for s in seen.values() for n in s}
        verdict[inst] = {
            "per_budget_node_counts": {str(k): sorted(v) for k, v in seen.items()},
            "distinct_node_counts": sorted(distinct),
            "node_count_depends_on_budget": len(distinct) > 1,
        }
    print("\n" + "=" * 78, flush=True)
    print(f"comparisons executed : {len(rows)}", flush=True)
    for inst, v in verdict.items():
        print(
            f"  {inst:20s} node_counts={v['distinct_node_counts']} "
            f"depends_on_budget={v['node_count_depends_on_budget']} "
            f"per_budget={v['per_budget_node_counts']}",
            flush=True,
        )
    print("=" * 78, flush=True)
    _write(
        args.out,
        {
            "arm": "forcebudget",
            "phase": phase,
            "budget": args.budget,
            "budgets": budgets,
            "reps": args.reps,
            "rows": rows,
            "verdict": verdict,
            "comparisons": len(rows),
        },
    )
    if not rows:
        print("FAIL: zero comparisons executed.", flush=True)
        return 3
    return 0


def _spin(n: int) -> list[subprocess.Popen]:
    procs = []
    for _ in range(n):
        procs.append(
            subprocess.Popen(
                [sys.executable, "-c", "\nwhile True:\n    pass\n"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        )
    return procs


def arm_load(args) -> int:
    insts = _pick(args)
    rows = []
    for cond, nload in (("idle", 0), ("loaded", args.load)):
        procs = _spin(nload)
        if nload:
            time.sleep(20.0)  # let the load average climb before measuring
        print(f"\ncondition={cond} spinners={nload} load1={_load1():.2f}", flush=True)
        try:
            for inst in insts:
                for rep in range(args.reps):
                    r = _solve(inst, args.budget)
                    r["condition"] = cond
                    r["rep"] = rep
                    r["load1"] = _load1()
                    rows.append(r)
                    print(
                        f"  {cond:7s} {inst:20s} rep={rep} -> {r['status']:11s} "
                        f"nodes={r['node_count']:>7d} root_bound={r['root_bound']} "
                        f"hits={r['obbt_deadline_hits']} wall={r['wall']:.1f} "
                        f"load1={r['load1']:.2f}",
                        flush=True,
                    )
        finally:
            for p in procs:
                p.kill()
            for p in procs:
                p.wait()

    verdict = {}
    for inst in insts:
        idle = sorted(
            r["node_count"] for r in rows if r["instance"] == inst and r["condition"] == "idle"
        )
        load = sorted(
            r["node_count"] for r in rows if r["instance"] == inst and r["condition"] == "loaded"
        )
        verdict[inst] = {
            "idle_node_counts": idle,
            "loaded_node_counts": load,
            "reproduced_under_load": bool(set(idle) != set(load)),
        }
    print("\n" + "=" * 78, flush=True)
    print(f"observations executed : {len(rows)}", flush=True)
    for inst, v in verdict.items():
        print(
            f"  {inst:20s} idle={v['idle_node_counts']} loaded={v['loaded_node_counts']} "
            f"differs={v['reproduced_under_load']}",
            flush=True,
        )
    print("=" * 78, flush=True)
    _write(
        args.out,
        {
            "arm": "load",
            "budget": args.budget,
            "spinners": args.load,
            "reps": args.reps,
            "rows": rows,
            "verdict": verdict,
            "observations": len(rows),
        },
    )
    if not rows:
        print("FAIL: zero observations executed.", flush=True)
        return 3
    return 0


def arm_clockscale(args) -> int:
    """Population-level: how many gated rows move when the clock runs `alpha`x fast?

    ``alpha = 1.25`` emulates a machine 25 % slower — far milder than the 3.5x
    load peak that produced the two observed panel failures.
    """
    insts = _pick(args)
    alphas = [float(x) for x in args.alphas.split(",")]
    print(
        f"clockscale: {len(insts)} instance(s) x alphas={alphas} @ {args.budget:.0f}s, "
        f"load1={_load1():.2f}",
        flush=True,
    )
    rows = []
    for i, inst in enumerate(insts, 1):
        per: dict[float, dict] = {}
        for a in alphas:
            r = _solve(inst, args.budget, alpha=a)
            r["alpha"] = a
            rows.append(r)
            per[a] = r
        base = per[alphas[0]]
        moved = any(
            per[a]["node_count"] != base["node_count"]
            or str(per[a]["status"]) != str(base["status"])
            for a in alphas[1:]
        )
        print(
            f"  [{i:3d}/{len(insts)}] {inst:24s} "
            + " ".join(f"a={a}:{per[a]['status'][:4]}/{per[a]['node_count']}" for a in alphas)
            + ("   <== MOVED" if moved else ""),
            flush=True,
        )

    verdict = {}
    for inst in insts:
        per = {r["alpha"]: r for r in rows if r["instance"] == inst}
        base = per[alphas[0]]
        verdict[inst] = {
            "node_counts": {str(a): per[a]["node_count"] for a in alphas},
            "statuses": {str(a): per[a]["status"] for a in alphas},
            "objectives": {str(a): per[a]["objective"] for a in alphas},
            "moved": any(
                per[a]["node_count"] != base["node_count"]
                or str(per[a]["status"]) != str(base["status"])
                for a in alphas[1:]
            ),
        }
    n_moved = sum(1 for v in verdict.values() if v["moved"])
    print("\n" + "=" * 78, flush=True)
    print(f"instances compared : {len(insts)}", flush=True)
    print(f"comparisons executed : {len(insts) * (len(alphas) - 1)}", flush=True)
    print(
        f"instances whose node_count/status MOVED under clock scaling : {n_moved} "
        f"({n_moved / max(1, len(insts)):.0%})",
        flush=True,
    )
    print("  " + ", ".join(k for k, v in verdict.items() if v["moved"]), flush=True)
    print("=" * 78, flush=True)
    _write(
        args.out,
        {
            "arm": "clockscale",
            "budget": args.budget,
            "alphas": alphas,
            "rows": rows,
            "verdict": verdict,
            "instances": len(insts),
            "comparisons": len(insts) * (len(alphas) - 1),
            "moved": n_moved,
        },
    )
    if not insts or len(alphas) < 2:
        print("FAIL: zero comparisons executed.", flush=True)
        return 3
    return 0


def _pick(args) -> list[str]:
    corpus = corpus_instances()
    if not args.instances:
        raise SystemExit("ERROR: --instances is required (comma list, or 'all').")
    if args.instances == "all":
        return corpus
    want = [w.strip() for w in args.instances.split(",") if w.strip()]
    missing = [w for w in want if w not in set(corpus)]
    if missing:
        raise SystemExit(f"ERROR: not in corpus: {', '.join(missing)}")
    return want


def _write(out: str | None, payload: dict) -> None:
    if not out:
        return
    p = Path(out)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload["load1_at_write"] = _load1()
    p.write_text(json.dumps(payload, indent=1) + "\n")
    print(f"report written: {p}", flush=True)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "--child":

        def _opt(i: int) -> float | None:
            return None if len(argv) <= i or argv[i] == "none" else float(argv[i])

        return _child(argv[1], float(argv[2]), _opt(3), _opt(4), _opt(5) or 1.0)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--arm", choices=("observe", "forcebudget", "load", "clockscale"), required=True)
    p.add_argument(
        "--alphas",
        default="1.0,1.25",
        help="clockscale arm: comma list; the first is the reference condition",
    )
    p.add_argument("--instances", help="comma list of instance stems, or 'all'")
    p.add_argument("--budget", type=float, default=45.0)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument(
        "--phase",
        choices=("obbt", "ils"),
        default="ils",
        help="which wall-clock-budgeted root phase the forcebudget arm drives",
    )
    p.add_argument("--budgets", default="5.0,4.0,3.0,2.0,1.0,0.5")
    p.add_argument("--load", type=int, default=6, help="busy processes for the loaded condition")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    if args.arm == "observe":
        return arm_observe(args)
    if args.arm == "forcebudget":
        return arm_forcebudget(args)
    if args.arm == "clockscale":
        return arm_clockscale(args)
    return arm_load(args)


if __name__ == "__main__":
    raise SystemExit(main())

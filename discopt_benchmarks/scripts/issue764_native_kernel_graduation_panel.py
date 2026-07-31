"""Issue #764 — Regime-2 graduation panel for the native Rust spatial B&B kernel.

Runs every ``.nl`` in the UNION of ``python/tests/data/minlplib_nl/`` and
``python/tests/data/minlplib/`` (119 instances; neither directory is a superset of
the other — see ``_CORPUS_DIRS``) twice — with
``DISCOPT_NATIVE_SPATIAL_KERNEL`` OFF then ON — at a 60 s budget, one subprocess
per (instance, flag) so env / JAX / global-counter state is fully isolated. It
records per instance: status, objective, bound, node_count, wall, whether the
native kernel *engaged* (its hand-off returned a certified result), and — for every
ON solve where it engaged and reported optimal — an independent feasibility
verification of the returned incumbent against the ORIGINAL model.

It then evaluates the two CLAUDE.md Regime-2 bars over ALL instances:

  * CERT-CLEAN (hard gate, zero slack): no ON-optimal objective differing from the
    OFF-optimal objective by more than tol (abs 1e-6 / rel 1e-4); no ON dual bound
    past the reference optimum by more than tol (sense-aware); no OFF-optimal
    instance regressing to non-optimal ON; every engaged ON-optimal incumbent
    independently feasibility-verified.
  * NET-POSITIVE: the kernel engages somewhere AND measurably helps (e.g. a
    timeout->optimal), AND does not measurably harm the rest — the median wall
    delta on NON-engaged instances (the producer-probe decline overhead) is small
    (<= max(0.5 s, 5 %)).
  * QUALITY-CLEAN (#902): ON must not return a WORSE incumbent than OFF, nor lose
    a primal OFF found. This is deliberately separate from cert-clean — a worse
    answer under a still-valid bound is not a soundness failure — but it blocks
    graduation, because "sound" was never the bar. It exists because every
    cert check requires one side to be ``optimal``, so when neither certifies the
    panel was blind to the answer actually returned. That is how the original
    graduation missed nvs19 (ON -315.0, 71% off; OFF -1097.6, 0.1% off).

Reference optima come from ``docs/dev/data/cert-optima.json``; instances absent
from it skip the oracle check and are SAID SO in the summary. Errored solves are
LABELED errored, never silently dropped.

LOAD ROBUSTNESS (#902). The verdict rests on wall-clock outcomes at a fixed budget,
so ambient load changes which instances hit the limit — it can change the verdict,
not merely the numbers. Rather than demand a quiet machine (a blocking load gate was
tried and is unusable on a real workstation), the panel runs in two stages and makes
the VERDICT robust:

  * STAGE 1 screens every instance once, to OBSERVE which ones engage the kernel.
    A screen awards nothing; it only decides what stage 2 re-runs.
  * STAGE 2 re-runs the DECISIVE instances (engaged, or the arms disagreed)
    ``PANEL_REPLICATES`` times with the arms INTERLEAVED, and requires differences to
    reproduce: a win must hold in EVERY replicate, a regression in a MAJORITY, and an
    instance whose replicates disagree on status OR on objective is QUARANTINED as
    unresolved. Objective agreement is part of that test because the quality gate
    compares objectives, and a median taken over a spread wider than the effect is not
    a measurement (§9) — see :func:`_objectives_agree` for the run this was measured on.

So load can move an instance into "unresolved" but can no longer make the verdict
wrong. Load is recorded (start/peak) for the reader; it never blocks the run.

Usage (parent):
    python discopt_benchmarks/scripts/issue764_native_kernel_graduation_panel.py
Env: PANEL_REPLICATES (default 3), PANEL_ONLY, PANEL_LIMIT.
Child mode (internal): --solve <instance> <0|1>
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

_BENCH_ROOT = Path(__file__).resolve().parent.parent
_REPO_ROOT = _BENCH_ROOT.parent
# BOTH in-repo corpora, unioned (#902). These two directories are NOT nested and
# neither is a superset of the other: ``minlplib_nl`` has 66 instances,
# ``minlplib`` has 81, they share only 28, and the union is 119. Panelling one
# alone silently omits whole families.
#
# That is not hypothetical — it is why this panel graduated the kernel while
# missing a regression. ``minlplib_nl`` (the only corpus this panel used) does not
# contain nvs17/nvs19/nvs24, precisely the family where the kernel engages and
# returns incumbents 71% from the reference optimum (#902). Conversely
# ``tanksize`` — the single instance whose improvement carried the net-positive
# bar — exists ONLY in ``minlplib_nl``. So neither directory alone can both
# justify and falsify this flag; the union is the minimum honest panel.
_CORPUS_DIRS = (
    _REPO_ROOT / "python" / "tests" / "data" / "minlplib_nl",
    _REPO_ROOT / "python" / "tests" / "data" / "minlplib",
)
_CORPUS = _CORPUS_DIRS[0]  # retained for messages that name a single directory


def _corpus_instances() -> list[str]:
    """Sorted union of instance stems across every corpus directory."""
    names: set[str] = set()
    for d in _CORPUS_DIRS:
        if d.is_dir():
            names.update(p.stem for p in d.glob("*.nl"))
    return sorted(names)


def _instance_path(instance: str):
    """Resolve an instance to whichever corpus directory holds it."""
    for d in _CORPUS_DIRS:
        p = d / f"{instance}.nl"
        if p.exists():
            return p
    raise FileNotFoundError(f"{instance}.nl not found in {[str(d) for d in _CORPUS_DIRS]}")


_CERT_OPTIMA = _REPO_ROOT / "docs" / "dev" / "data" / "cert-optima.json"
_RESULTS_DIR = _BENCH_ROOT / "results"

_ABS_TOL = 1e-6
_REL_TOL = 1e-4
_TIME_LIMIT = 60.0
_CHILD_TIMEOUT = 200.0  # subprocess wall guard; solve itself is bounded to _TIME_LIMIT


def _obj_match(a, b) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= _ABS_TOL + _REL_TOL * max(abs(a), abs(b))


# --------------------------------------------------------------------------- #
# Child: solve ONE instance under ONE flag setting, print a single JSON line.  #
# --------------------------------------------------------------------------- #
def _run_child(instance: str, flag: str) -> int:
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    os.environ.setdefault("JAX_ENABLE_X64", "1")
    # Set BOTH arms explicitly. Unsetting used to mean OFF, but the flag graduated to
    # default-ON (#764, 2026-07-27), so ``pop`` now yields ON and the panel would
    # compare ON against ON — silently, forever, reporting helped=0 and never able to
    # catch a regression. Verified live the moment the default flipped: tanksize
    # engaged in both arms. An A/B harness must never infer an arm from a default it
    # does not control.
    os.environ["DISCOPT_NATIVE_SPATIAL_KERNEL"] = "1" if flag == "1" else "0"

    import discopt.solver as solver_mod
    import numpy as np
    from discopt.modeling.core import ObjectiveSense, from_nl

    nl = str(_instance_path(instance))
    out: dict = {"instance": instance, "flag": flag, "engaged": False}

    # Two distinct signals, deliberately NOT conflated:
    #   ``binding_called``/``native_time_s`` instrument the Rust entry itself, so a
    #     run that reached the kernel is distinguishable from one the producer
    #     declined, and native-kernel time is separable from producer/setup time.
    #   ``engaged`` keeps its original meaning -- the native result was actually
    #     surfaced as the answer. A binding call whose result is discarded (e.g.
    #     ``node_limit`` -> Python fallback) must not count as engagement, or the
    #     panel overstates native coverage.
    binding_called = {"v": False}
    native_status = {"v": None}
    native_time = {"v": 0.0}
    engaged = {"v": False}
    orig_fn = solver_mod._try_native_spatial_kernel
    from discopt import _rust

    orig_rust_fn = _rust.solve_spatial_tree_py

    def _wrapped_rust(*a, **k):
        binding_called["v"] = True
        t_native = time.perf_counter()
        try:
            r = orig_rust_fn(*a, **k)
        finally:
            native_time["v"] += time.perf_counter() - t_native
        native_status["v"] = r.get("status")
        return r

    _rust.solve_spatial_tree_py = _wrapped_rust

    def _wrapped(*a, **k):
        r = orig_fn(*a, **k)
        if r is not None:
            engaged["v"] = True
        return r

    solver_mod._try_native_spatial_kernel = _wrapped

    try:
        model = from_nl(nl)
        sense = "max" if model._objective.sense == ObjectiveSense.MAXIMIZE else "min"
        out["sense"] = sense
        t0 = time.perf_counter()
        r = model.solve(time_limit=_TIME_LIMIT)
        out["wall"] = time.perf_counter() - t0
        out["status"] = str(r.status)
        out["objective"] = None if r.objective is None else float(r.objective)
        out["bound"] = None if r.bound is None else float(r.bound)
        out["node_count"] = int(r.node_count)
        out["engaged"] = bool(engaged["v"])
        out["binding_called"] = bool(binding_called["v"])
        out["native_status"] = native_status["v"]
        out["native_time_s"] = float(native_time["v"])

        # Independent incumbent feasibility verification (ON + engaged + optimal).
        if flag == "1" and engaged["v"] and str(r.status) == "optimal" and r.x is not None:
            try:
                x_flat = np.array(
                    [float(np.asarray(r.x[v.name]).reshape(-1)[0]) for v in model._variables],
                    dtype=np.float64,
                )
                ok, verified_obj = solver_mod._native_kernel_verify_point(model, x_flat)
                out["incumbent_feasible"] = bool(ok)
                out["verified_obj"] = None if verified_obj is None else float(verified_obj)
            except Exception as exc:  # verification machinery failure -> record, don't pass
                out["incumbent_feasible"] = False
                out["verify_error"] = repr(exc)
    except Exception as exc:
        out["status"] = "errored"
        out["error"] = repr(exc)
    finally:
        solver_mod._try_native_spatial_kernel = orig_fn
        _rust.solve_spatial_tree_py = orig_rust_fn

    print("RESULT_JSON " + json.dumps(out))
    return 0


# --------------------------------------------------------------------------- #
# Parent: drive every instance, both flags, then evaluate the two bars.        #
# --------------------------------------------------------------------------- #
def _solve_one(instance: str, flag: str) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--solve", instance, flag]
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    env.setdefault("JAX_ENABLE_X64", "1")
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=_CHILD_TIMEOUT, env=env)
    except subprocess.TimeoutExpired:
        return {"instance": instance, "flag": flag, "status": "child_timeout", "engaged": False}
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT_JSON "):
            return json.loads(line[len("RESULT_JSON ") :])
    return {
        "instance": instance,
        "flag": flag,
        "status": "child_crashed",
        "engaged": False,
        "stderr_tail": proc.stderr[-800:],
    }


# --------------------------------------------------------------------------- #
# Load robustness (#902): REPLICATE THE DECISIVE INSTANCES, don't wait for quiet.
#
# The net-positive bar is a wall-time comparison at a fixed 60 s budget, so ambient
# load does not merely add noise to the numbers — it decides which instances hit the
# limit, i.e. it changes the *statuses* the verdict is computed from. This panel
# previously had no defence against that at all and would emit a verdict under load 24
# indistinguishable from a clean one.
#
# The first attempt was a blocking load gate (refuse to start until 1-min load < 2.5).
# Rejected in practice: on a real workstation it simply never runs — it is a wish, not
# a test. Two other designs were tried and MEASURED to fail:
#
#   * Budget deterministic WORK (max_nodes) instead of wall time. Solves are indeed
#     bit-reproducible that way (3 reps x 3 instances x both arms: 6/6 identical
#     status/objective/bound/node_count, taken under load 3-5). But
#     ``_try_native_spatial_kernel`` returns None on a ``node_limit`` exit, so the
#     kernel FALLS BACK to the Python path and ``engaged=False`` on every instance
#     tested (nvs17, nvs19, tanksize). A node-budgeted panel compares OFF against OFF:
#     perfectly deterministic and perfectly meaningless — the same "comparing ON
#     against ON" blindness that produced #902. Node counts are not even a comparable
#     unit across the arms (nvs19: 69 Python nodes vs 67,678 native nodes).
#   * Statically PRE-FILTER to the producer's covered subset, so only those instances
#     are A/B'd and the freed budget pays for replication. ``build_spatial_kernel_spec``
#     answers in 1.2 s for all 119 (32 covered), but over the RAW DECLARED bounds — and
#     the kernel is handed the POST-FBBT/OBBT box. ``tanksize``, the flag's headline
#     win, is declined over raw bounds and accepted over the presolved one. As an
#     exclusion gate it silently drops the instance carrying the verdict. Engagement
#     must be OBSERVED, never predicted.
#
# So: keep the wall budget, and make the VERDICT robust instead of the machine quiet.
# Decisive instances are re-run R times, arms interleaved, and a difference only counts
# if it REPRODUCES. The asymmetry is deliberate — a graduation gate should be hard to
# pass and easy to fail:
#
#   * a WIN (helped) counts only if it holds in EVERY replicate;
#   * a REGRESSION fires if it holds in a MAJORITY, and is reported if it holds in any;
#   * an instance whose replicates disagree on status is UNSTABLE and is quarantined —
#     it can neither justify nor condemn the flag, and is listed as unresolved.
#
# Load can therefore only ever move an instance into "unresolved". It can no longer
# make the verdict wrong. Load is still recorded (start/peak) so a reader can see the
# conditions, but it never blocks the run.
_REPLICATES = int(os.environ.get("PANEL_REPLICATES", "3"))


def _load1() -> float:
    """1-minute load average, or ``nan`` where the platform has none."""
    try:
        return float(os.getloadavg()[0])
    except (OSError, AttributeError):  # pragma: no cover - platform without loadavg
        return float("nan")


def _solve_replicated(instance: str, reps: int) -> tuple[list[dict], list[dict]]:
    """Run ``instance`` ``reps`` times per arm, INTERLEAVED (OFF, ON, OFF, ON, ...).

    Interleaving is what makes the pairing valid under drifting load (CLAUDE.md §9):
    running all the OFF reps and then all the ON reps would charge any slow period
    entirely to one arm. Returns ``(off_runs, on_runs)``.
    """
    off_runs: list[dict] = []
    on_runs: list[dict] = []
    for _ in range(max(1, reps)):
        off_runs.append(_solve_one(instance, "0"))
        on_runs.append(_solve_one(instance, "1"))
    return off_runs, on_runs


def _statuses_agree(runs: list[dict]) -> bool:
    """Whether every replicate of one arm reached the same status."""
    if not runs:
        return False
    return len({str(r.get("status")) for r in runs}) == 1


# A claimed objective difference is only a measurement if it is larger than the noise
# it was measured against (CLAUDE.md §9: report a spread). An instance whose own arm
# cannot reproduce itself to within this factor of the difference being claimed is
# quarantined rather than scored. 2.0 = "the effect must be at least twice the spread".
_MIN_EFFECT_TO_SPREAD = 2.0


def _objective_spread(runs: list[dict]) -> float:
    """Within-arm objective spread (max - min) across replicates.

    ``inf`` when the arm cannot even agree on whether a primal exists — some replicates
    returned one and others did not, which is maximal disagreement, not a small spread.
    ``0.0`` when no replicate found a primal ("no primal" is itself reproducible).
    """
    vals = [r.get("objective") for r in runs]
    if all(v is None for v in vals):
        return 0.0
    if any(v is None for v in vals):
        return float("inf")
    return float(max(vals) - min(vals))


def _difference_is_attributable(rep: dict) -> bool:
    """Whether an ON/OFF objective difference is bigger than the noise it sits in.

    The stability guard used to stop at the STATUS, but the quality gate compares
    OBJECTIVES — so an arm could be "stable" while disagreeing with itself about the
    very number being judged, and the median of that disagreement was then reported as
    a reproduced regression. That is not hypothetical; it decided a verdict.

    On the 2026-07-31 119-instance run, ``heatexch_gen1`` came back ``feasible`` in all
    six runs (status-stable), but the ON arm returned ``167654.27, 167545.24,
    167654.27`` — a 109.03 spread — against an ON/OFF median difference of exactly
    109.03. Effect/spread = 1.0: the "regression" was the same size as the arm's
    disagreement with itself. It was the ONLY quality violation on the corpus and it
    alone produced ``GRADUATE: NO``.

    Two independent measurements say the flag did not cause it. The kernel never
    engaged there (``binding_called`` False — a declined model, so both arms run the
    *same* engine and differ only by the producer probe), and that probe was timed
    inside a real solve at **0.016 s**, which cannot account for the ~15-node gap
    between the two answers. Re-measured 3x2 interleaved outside the panel, the sign
    FLIPS: OFF returned the worse 167654.27 twice and ON the better 167545.24 twice.
    The instance is bimodal at a wall-clock cutoff, in both arms.

    This is a STRENGTHENING, not a loosening. It removes an instance from BOTH sides —
    an unattributable instance can no longer be counted as ``helped`` either — and a
    genuinely reproducible regression still fires, because its arms reproduce
    themselves: on the same run every other decisive instance had spread exactly 0.0
    in both arms (31 of 32), so effect/spread is 0 for all of them and 1.0 for
    ``heatexch_gen1``. The verdict is therefore insensitive to where in ``(0, 1]`` the
    threshold sits; :data:`_MIN_EFFECT_TO_SPREAD` is not a tuned number.
    """
    off_obj = rep.get("off_median_objective")
    on_obj = rep.get("on_median_objective")
    if off_obj is None or on_obj is None:
        # No two-sided objective comparison to attribute (e.g. ON gains a primal OFF
        # never finds). Status-level reproducibility already governs those.
        return True
    effect = abs(float(off_obj) - float(on_obj))
    if effect <= _ABS_TOL + _REL_TOL * max(abs(off_obj), abs(on_obj)):
        return True  # the arms agree; nothing is being claimed
    spread = max(_objective_spread(rep["off"]), _objective_spread(rep["on"]))
    return effect >= _MIN_EFFECT_TO_SPREAD * spread


def _median_objective(runs: list[dict]) -> float | None:
    """Median objective across replicates, or ``None`` if no replicate found a primal.

    Median rather than best/worst so a single load-perturbed replicate cannot swing the
    reported value; the stability of the set is judged separately by
    :func:`_statuses_agree`.
    """
    vals = [r["objective"] for r in runs if r.get("objective") is not None]
    if not vals:
        return None
    vals.sort()
    n = len(vals)
    return float(vals[n // 2]) if n % 2 else float(0.5 * (vals[n // 2 - 1] + vals[n // 2]))


def main() -> int:
    if len(sys.argv) >= 4 and sys.argv[1] == "--solve":
        return _run_child(sys.argv[2], sys.argv[3])

    optima: dict = {}
    if _CERT_OPTIMA.exists():
        optima = json.loads(_CERT_OPTIMA.read_text())
    # Widen from the full MINLPLib oracle when the corpus is installed. The vendored
    # file is the CI-safe floor (no corpus there); ``minlplib.solu`` is the ground
    # truth and carries far more. Keeping only the vendored file is how this panel
    # came to skip 21 of 66 instances — including its own headline win — while
    # reporting a clean pass. The two sources were verified to AGREE on all 37
    # entries they shared, so vendored values win ties only to keep runs
    # reproducible when the corpus is absent.
    _solu_added = 0
    try:
        if str(_BENCH_ROOT) not in sys.path:
            sys.path.insert(0, str(_BENCH_ROOT))
        from utils.corpus import solu_path  # noqa: PLC0415

        _sp = solu_path()
        if _sp is not None:
            for _line in _sp.read_text().splitlines():
                _f = _line.split()
                if len(_f) >= 3 and _f[0] == "=opt=" and _f[1] not in optima:
                    optima[_f[1]] = float(_f[2])
                    _solu_added += 1
    except Exception as _exc:  # corpus absent (CI) or resolver missing: vendored only
        print(f"note: minlplib.solu not merged ({_exc}); using vendored optima only", flush=True)

    instances = _corpus_instances()
    # Optional smoke subset (validation only): PANEL_LIMIT=N runs the first N, and
    # PANEL_ONLY=a,b,c runs exactly those. Unset -> the full corpus.
    only = os.environ.get("PANEL_ONLY", "").strip()
    if only:
        wanted = {s.strip() for s in only.split(",") if s.strip()}
        instances = [i for i in instances if i in wanted]
    limit = os.environ.get("PANEL_LIMIT", "").strip()
    if limit.isdigit():
        instances = instances[: int(limit)]
    print(
        f"#764 native-kernel graduation panel: {len(instances)} instances, "
        f"budget {_TIME_LIMIT:.0f}s, OFF then ON, subprocess-isolated.\n"
        f"Reference optima: {len(optima)} entries "
        f"({_CERT_OPTIMA.name} + {_solu_added} merged from minlplib.solu).\n"
        f"Oracle coverage of this corpus: "
        f"{sum(1 for i in instances if i in optima)}/{len(instances)}.\n",
        flush=True,
    )

    load_start = _load1()
    load_peak = 0.0 if math.isnan(load_start) else load_start
    print(f"1-min load at start: {load_start:.2f} (recorded, not gated).", flush=True)

    # ---- STAGE 1: screen every instance once, to OBSERVE which ones engage -------
    # Engagement is observed rather than predicted: the producer is handed the
    # post-presolve box, so a static prediction over declared bounds has false
    # negatives (it drops tanksize). A screen cannot award or deny graduation; it only
    # decides what stage 2 must re-run.
    rows: dict[str, dict] = {}
    for i, inst in enumerate(instances, 1):
        off = _solve_one(inst, "0")
        on = _solve_one(inst, "1")
        _l = _load1()
        if not math.isnan(_l):
            load_peak = max(load_peak, _l)
        rows[inst] = {"off": off, "on": on}
        eng = "ENGAGED" if on.get("engaged") else "decline"
        print(
            f"  [{i:2d}/{len(instances)}] {inst:20s} "
            f"OFF={off.get('status', '?'):11s} ON={on.get('status', '?'):11s} {eng:8s} "
            f"objOFF={off.get('objective')!s:>12.12s} objON={on.get('objective')!s:>12.12s} "
            f"wOFF={off.get('wall', 0):6.1f} wON={on.get('wall', 0):6.1f}",
            flush=True,
        )

    # ---- STAGE 2: replicate the DECISIVE instances ------------------------------
    # Decisive = the kernel engaged, OR the two arms disagreed on status or objective.
    # Everything else contributes only the producer-probe overhead median, which no
    # single instance can swing. Taking the UNION of those criteria (rather than
    # engagement alone) means an instance that regressed WITHOUT engaging is still
    # re-run rather than waved through.
    decisive = sorted(
        inst
        for inst, r in rows.items()
        if r["on"].get("engaged")
        or str(r["off"].get("status")) != str(r["on"].get("status"))
        or not _obj_match(r["off"].get("objective"), r["on"].get("objective"))
    )
    if _REPLICATES > 1 and decisive:
        print(
            f"\nSTAGE 2: replicating {len(decisive)} decisive instance(s) "
            f"x{_REPLICATES}, arms interleaved: {', '.join(decisive)}",
            flush=True,
        )
        for j, inst in enumerate(decisive, 1):
            off_runs, on_runs = _solve_replicated(inst, _REPLICATES)
            _l = _load1()
            if not math.isnan(_l):
                load_peak = max(load_peak, _l)
            off_stable = _statuses_agree(off_runs)
            on_stable = _statuses_agree(on_runs)
            rows[inst]["replicates"] = {
                "off": off_runs,
                "on": on_runs,
                "off_stable": off_stable,
                "on_stable": on_stable,
                "stable": off_stable and on_stable,
                "off_median_objective": _median_objective(off_runs),
                "on_median_objective": _median_objective(on_runs),
            }
            print(
                f"  [{j}/{len(decisive)}] {inst:20s} "
                f"OFF={[str(r.get('status')) for r in off_runs]} "
                f"ON={[str(r.get('status')) for r in on_runs]} "
                f"{'STABLE' if off_stable and on_stable else 'UNSTABLE -> quarantined'}",
                flush=True,
            )

    verdict = _evaluate(rows, optima)
    # Record the load this run ACTUALLY ran at. A run that starts quiet and then
    # competes with a background job for an hour is exactly the confound §9 describes,
    # and only the peak reveals it. Recorded, never used to block.
    verdict["load_start"] = load_start
    verdict["load_peak"] = load_peak
    verdict["replicates"] = _REPLICATES
    verdict["decisive_instances"] = decisive

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    _RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = _RESULTS_DIR / f"issue764_native_kernel_graduation_panel_{stamp}.json"
    raw_path.write_text(json.dumps({"rows": rows, "verdict": verdict}, indent=1))
    summary = _render_summary(rows, verdict, optima, stamp)
    summary_path = _RESULTS_DIR / f"issue764_native_kernel_graduation_panel_{stamp}.txt"
    summary_path.write_text(summary)
    print("\n" + summary)
    print(f"\nRaw JSON:  {raw_path}")
    print(f"Summary :  {summary_path}")
    return 0 if verdict["graduate"] else 1


def _evaluate(rows: dict, optima: dict) -> dict:
    cert_violations: list[str] = []
    errored: list[str] = []
    engaged_insts: list[str] = []
    helped: list[str] = []  # engaged AND ON strictly better outcome than OFF
    quality_violations: list[str] = []  # ON answer worse than OFF (#902)
    unstable: list[str] = []  # replicates disagreed -> quarantined, carries no verdict
    non_engaged_wall_delta: list[float] = []
    no_oracle: list[str] = []

    _bad = ("errored", "child_crashed", "child_timeout")
    for inst, pair in sorted(rows.items()):
        off, on = pair["off"], pair["on"]
        off_status = off.get("status")
        on_status = on.get("status")
        if off_status in _bad:
            detail = str(off.get("error", off.get("stderr_tail", "")))[:120]
            errored.append(f"{inst}: OFF {off_status} ({detail})")
        if on_status in _bad:
            detail = str(on.get("error", on.get("stderr_tail", "")))[:120]
            errored.append(f"{inst}: ON {on_status} ({detail})")

        engaged = bool(on.get("engaged"))
        if engaged:
            engaged_insts.append(inst)
        else:
            # Producer-probe decline overhead: ON wall vs OFF wall on non-engaged.
            wall_ok = isinstance(off.get("wall"), (int, float)) and isinstance(
                on.get("wall"), (int, float)
            )
            if wall_ok and off_status not in _bad and on_status not in _bad:
                non_engaged_wall_delta.append(float(on["wall"]) - float(off["wall"]))

        sense = on.get("sense") or off.get("sense") or "min"

        # (1) objective agreement when BOTH optimal.
        if (
            off_status == "optimal"
            and on_status == "optimal"
            and not _obj_match(off.get("objective"), on.get("objective"))
        ):
            cert_violations.append(
                f"{inst}: ON/OFF optimal objective mismatch "
                f"OFF={off.get('objective')} ON={on.get('objective')}"
            )

        # (2) OFF-optimal must not regress to non-optimal ON.
        if off_status == "optimal" and on_status != "optimal":
            cert_violations.append(
                f"{inst}: OFF optimal but ON {on_status} (optimality regression)"
            )

        # (3) ON dual bound must not pass the reference optimum (sense-aware).
        if inst in optima and isinstance(on.get("bound"), (int, float)):
            opt = float(optima[inst])
            b = float(on["bound"])
            tol = _ABS_TOL + _REL_TOL * max(abs(opt), abs(b))
            if sense == "min" and b > opt + tol:
                cert_violations.append(
                    f"{inst}: ON lower bound {b} ABOVE reference optimum {opt} (+tol {tol:.1e})"
                )
            if sense == "max" and b < opt - tol:
                cert_violations.append(
                    f"{inst}: ON upper bound {b} BELOW reference optimum {opt} (-tol {tol:.1e})"
                )
        elif inst not in optima:
            no_oracle.append(inst)

        # (4) engaged ON-optimal incumbent must be independently feasibility-verified.
        if engaged and on_status == "optimal":
            if not on.get("incumbent_feasible", False):
                cert_violations.append(
                    f"{inst}: engaged ON optimal but incumbent NOT feasibility-verified "
                    f"({on.get('verify_error', '')})"
                )
            else:
                # And its verified true objective must match the reported objective.
                if not _obj_match(on.get("verified_obj"), on.get("objective")):
                    cert_violations.append(
                        f"{inst}: engaged ON verified_obj {on.get('verified_obj')} != "
                        f"reported {on.get('objective')}"
                    )

        # (5) INCUMBENT-QUALITY regression (#902). Checks 1-4 above all require at
        # least one side to be ``optimal``: (1) needs BOTH optimal, (2) needs OFF
        # optimal, (4) needs ON optimal. So when NEITHER run certifies — the common
        # case on hard instances — every check above is skipped and the panel is
        # blind to the answer actually returned.
        #
        # That is exactly how the #764 graduation missed nvs19: ON came back
        # ``time_limit`` with objective -315.0 (71% from the reference optimum
        # -1098.4) while OFF came back ``feasible`` with -1097.6 (0.1% off) in 9
        # nodes. Neither status is ``optimal``, so nothing fired, and the flag
        # graduated. The dual bounds stayed valid throughout, so this is NOT a
        # soundness failure — which is why it belongs in its own gate rather than
        # in ``cert_violations`` — but shipping a default that makes the returned
        # answer 71% worse is precisely the "net-positive" bar in CLAUDE.md §5.
        #
        # Compared sense-aware and only where OFF actually produced something, so a
        # genuine improvement (ON finds a primal where OFF found none) can never
        # register as a regression.
        # Replicated evidence governs where it exists (#902). An instance whose
        # replicates disagree on status is UNSTABLE: the machine, not the flag, is
        # deciding its outcome, so it is quarantined — it can neither justify (helped)
        # nor condemn (quality violation) the flag, and is reported as unresolved.
        # This is what lets the panel run on a busy machine: load can move an instance
        # into "unresolved", but it can no longer make the verdict wrong.
        rep = pair.get("replicates")
        if rep is not None and not rep.get("stable", False):
            unstable.append(
                f"{inst}: replicates disagree on STATUS — "
                f"OFF={[str(r.get('status')) for r in rep['off']]} "
                f"ON={[str(r.get('status')) for r in rep['on']]}"
            )
            continue
        # Status-stable is not enough: the gate below compares OBJECTIVES, so an
        # objective difference must also be bigger than the arms' disagreement with
        # themselves before it can be attributed to the flag (§9). See
        # :func:`_difference_is_attributable` for the run that made this necessary.
        if rep is not None and not _difference_is_attributable(rep):
            unstable.append(
                f"{inst}: objective difference is inside the replicate SPREAD — "
                f"OFF={[r.get('objective') for r in rep['off']]} "
                f"ON={[r.get('objective') for r in rep['on']]} "
                f"(effect {abs(rep['off_median_objective'] - rep['on_median_objective']):.6g} "
                f"vs spread "
                f"{max(_objective_spread(rep['off']), _objective_spread(rep['on'])):.6g})"
            )
            continue

        if rep is not None:
            off_obj = rep.get("off_median_objective")
            on_obj = rep.get("on_median_objective")
        else:
            off_obj, on_obj = off.get("objective"), on.get("objective")

        if off_obj is not None and on_obj is None:
            quality_violations.append(
                f"{inst}: PRIMAL LOST — OFF found {off_obj} ({off_status}), "
                f"ON returned no incumbent ({on_status})"
            )
        elif off_obj is not None and on_obj is not None:
            qtol = _ABS_TOL + _REL_TOL * max(abs(off_obj), abs(on_obj))
            worse = (on_obj > off_obj + qtol) if sense == "min" else (on_obj < off_obj - qtol)
            # With replicates, require the regression to hold in a MAJORITY of PAIRED
            # runs as well as in the medians. A win must survive every replicate but a
            # regression need only survive most of them: a graduation gate should be
            # hard to pass and easy to fail.
            if worse and rep is not None:
                paired = list(zip(rep["off"], rep["on"], strict=True))
                n_worse = 0
                for o_run, n_run in paired:
                    o_v, n_v = o_run.get("objective"), n_run.get("objective")
                    if o_v is None or n_v is None:
                        continue
                    t = _ABS_TOL + _REL_TOL * max(abs(o_v), abs(n_v))
                    n_worse += int((n_v > o_v + t) if sense == "min" else (n_v < o_v - t))
                worse = n_worse * 2 > len(paired)
            if worse:
                ref = optima.get(inst)
                detail = ""
                if ref is not None and abs(ref) > 0:
                    detail = (
                        f" [vs reference {ref}: OFF {100 * abs(off_obj - ref) / abs(ref):.1f}% off,"
                        f" ON {100 * abs(on_obj - ref) / abs(ref):.1f}% off]"
                    )
                quality_violations.append(
                    f"{inst}: INCUMBENT WORSE under ON — OFF={off_obj} ({off_status}) "
                    f"vs ON={on_obj} ({on_status}){detail}"
                    + ("" if rep is None else f" [reproduced over {_REPLICATES} replicates]")
                )

        # Net-positive "helped": engaged AND ON reached optimal where OFF did not.
        # Under replication this must hold in EVERY replicate — one lucky run is not a
        # win, and the whole net-positive bar has historically rested on a single
        # instance, so a flaky one must not be able to carry it.
        if engaged:
            if rep is not None:
                if all(str(r.get("status")) == "optimal" for r in rep["on"]) and not any(
                    str(r.get("status")) == "optimal" for r in rep["off"]
                ):
                    helped.append(inst)
            elif on_status == "optimal" and off_status != "optimal":
                helped.append(inst)

    non_engaged_wall_delta.sort()
    median_delta = 0.0
    if non_engaged_wall_delta:
        n = len(non_engaged_wall_delta)
        median_delta = (
            non_engaged_wall_delta[n // 2]
            if n % 2
            else 0.5 * (non_engaged_wall_delta[n // 2 - 1] + non_engaged_wall_delta[n // 2])
        )

    cert_clean = len(cert_violations) == 0
    # Answer quality is a SEPARATE gate from soundness. A worse incumbent under a
    # valid bound is not a certification failure, so it does not belong in
    # cert_violations -- but it must still block graduation, because "the flag is
    # sound" was never the bar. CLAUDE.md 5 requires net-positive too (#902).
    quality_clean = len(quality_violations) == 0
    overhead_ok = median_delta <= max(0.5, 0.05 * _TIME_LIMIT)
    net_positive = (len(engaged_insts) > 0) and (len(helped) > 0) and overhead_ok and quality_clean

    return {
        "cert_clean": cert_clean,
        "cert_violations": cert_violations,
        "quality_clean": quality_clean,
        "quality_violations": quality_violations,
        "net_positive": net_positive,
        "engaged": engaged_insts,
        "helped": helped,
        "unstable": unstable,
        "median_nonengaged_wall_delta_s": median_delta,
        "overhead_ok": overhead_ok,
        "errored": errored,
        "no_oracle_instances": no_oracle,
        "oracle_total": len(rows),
        "oracle_covered": len(rows) - len(no_oracle),
        "n_nonengaged_measured": len(non_engaged_wall_delta),
        "graduate": cert_clean and quality_clean and net_positive,
    }


def _render_summary(rows: dict, v: dict, optima: dict, stamp: str) -> str:
    lines: list[str] = []
    lines.append(f"# #764 native-kernel graduation panel — {stamp}")
    lines.append(
        f"# corpus: {len(rows)} instances, budget {_TIME_LIMIT:.0f}s, OFF vs ON, "
        f"subprocess-isolated. Reference optima: {len(optima)} entries."
    )
    # The load this run ran at is part of the result, not metadata (#902): the
    # net-positive bar is a wall-time comparison at a fixed budget, so a reader must be
    # able to see whether the machine was quiet without taking it on faith.
    _ls, _lp = v.get("load_start"), v.get("load_peak")
    if _ls is not None:
        lines.append(
            f"# machine load: start {_ls:.2f}, peak {_lp:.2f} (recorded, not gated — "
            f"robustness comes from {v.get('replicates', 1)}x replication of the "
            f"{len(v.get('decisive_instances', []))} decisive instance(s), not from a "
            f"quiet machine)"
        )
    lines.append("")
    lines.append("## VERDICT")
    _qv = v.get("quality_violations", [])
    lines.append(
        f"quality-clean  : {'PASS' if v.get('quality_clean', True) else 'FAIL'} "
        f"({len(_qv)} incumbent-quality regressions)"
    )
    for _line in _qv[:20]:
        lines.append(f"    - {_line}")
    # Oracle COVERAGE is part of the verdict line, not a footnote further down. This
    # panel previously printed "cert-clean : PASS (0 violations)" while silently
    # checking only 31 of 66 instances — its oracle file was missing 21 optima that
    # ``minlplib.solu`` carries, INCLUDING ``tanksize``, the instance the run's own
    # net-positive bar rests on. A PASS that reads as "checked everything" when it
    # checked half the corpus is the failure mode CLAUDE.md §6 is about: an instrument
    # that measures less than it appears to and is believed.
    _cov = v.get("oracle_covered", 0)
    _tot = v.get("oracle_total", 0)
    _pct = (100.0 * _cov / _tot) if _tot else 0.0
    lines.append(
        f"  cert-clean   : {'PASS' if v['cert_clean'] else 'FAIL'} "
        f"({len(v['cert_violations'])} violation(s); "
        f"oracle-checked {_cov}/{_tot} instances = {_pct:.0f}%, "
        f"{len(v['no_oracle_instances'])} unchecked)"
    )
    lines.append(
        f"  net-positive : {'PASS' if v['net_positive'] else 'FAIL'} "
        f"(engaged={len(v['engaged'])}, helped={len(v['helped'])}, "
        f"median non-engaged wall Δ={v['median_nonengaged_wall_delta_s']:+.3f}s over "
        f"{v['n_nonengaged_measured']} instances, overhead_ok={v['overhead_ok']})"
    )
    # Quarantined instances are part of the verdict's honesty, not a footnote: an
    # UNRESOLVED instance is the panel saying "the machine decided this one, not the
    # flag". Reporting the count next to the bars keeps a run that resolved almost
    # nothing from reading like a clean pass.
    _uns = v.get("unstable", [])
    lines.append(
        f"  unresolved   : {len(_uns)} instance(s) quarantined for replicate "
        f"disagreement (carry no verdict either way)"
    )
    for _line in _uns[:20]:
        lines.append(f"    - {_line}")
    _grad = "YES — flip default ON" if v["graduate"] else "NO — keep opt-in"
    lines.append(f"  GRADUATE     : {_grad}")
    lines.append("")
    if v["cert_violations"]:
        lines.append("## CERT-CLEAN VIOLATIONS")
        for s in v["cert_violations"]:
            lines.append(f"  - {s}")
        lines.append("")
    _eng = ", ".join(v["engaged"]) or "(none)"
    lines.append(f"## ENGAGED instances ({len(v['engaged'])}): {_eng}")
    lines.append(
        f"## HELPED  (engaged, ON optimal where OFF was not) ({len(v['helped'])}): "
        f"{', '.join(v['helped']) or '(none)'}"
    )
    lines.append("")
    if v["errored"]:
        lines.append(f"## ERRORED ({len(v['errored'])})")
        for s in v["errored"]:
            lines.append(f"  - {s}")
        lines.append("")
    if v["no_oracle_instances"]:
        lines.append(
            f"## NO REFERENCE OPTIMUM (oracle check skipped) ({len(v['no_oracle_instances'])}):"
        )
        lines.append("  " + ", ".join(v["no_oracle_instances"]))
        lines.append("")
    lines.append("## PER-INSTANCE (engaged rows first)")
    ordered = sorted(rows.items(), key=lambda kv: (not kv[1]["on"].get("engaged"), kv[0]))
    lines.append(
        f"  {'instance':20s} {'OFF status':12s} {'ON status':12s} eng "
        f"{'objOFF':>14s} {'objON':>14s} {'boundON':>14s} {'wOFF':>7s} {'wON':>7s}"
    )
    for inst, pair in ordered:
        off, on = pair["off"], pair["on"]
        lines.append(
            f"  {inst:20s} {str(off.get('status')):12s} {str(on.get('status')):12s} "
            f"{'Y' if on.get('engaged') else '.':3s} "
            f"{str(off.get('objective'))[:14]:>14s} {str(on.get('objective'))[:14]:>14s} "
            f"{str(on.get('bound'))[:14]:>14s} "
            f"{off.get('wall', 0):7.1f} {on.get('wall', 0):7.1f}"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())

"""R3a entry experiment: bound-responsiveness fingerprint vs actual branching.

This is the *entry experiment / kill criterion* for R3 in
``docs/dev/uncertified-tail-plan-2026-07-06.md`` §3 (R3a). It is a MEASUREMENT
task: it instruments solves and produces a per-instance table + a BUILD/SKIP
verdict for R3b (responsiveness-aware spatial branching). It does NOT build the
branching change and it changes NO solver decision — the only Rust addition is a
behavior-neutral per-variable branch-frequency counter (``branch_var_counts``),
armed via the ``solver._R3A_BRANCH_COUNT_SINK`` diagnostic sink.

Per instance, two independent measurements are combined:

  1. **Responsiveness fingerprint.** For each variable ``v`` (in the *reformed*
     flat space the solver actually branches — factorable lift appends aux
     columns after the originals), halve its box two ways (lower / upper half),
     compute the root McCormick-LP bound of each half, keep the half with the
     *better* (larger, for the internally-minimized sense) root bound, and record
     ``score(v) = |root_bound(halved) − root_bound(full)|``. This is O(n_vars)
     cheap root-bound evaluations — no full solves. Reuses ``root_lp_bound`` from
     ``t21_root_loop_replay`` (the same cold McCormick relaxation the solver's
     root path builds).

  2. **Actual branching.** One instrumented 60 s solve; read the Rust tree's
     ``branch_var_counts`` (a temporary counter incremented once per branching
     event, integer or spatial, with the branched flat column). Count branch
     frequency per variable.

Deliverable (printed + JSON): per instance the top-3 *responsive* variables (by
``score``) vs the top-3 *actually-branched* variables (by frequency), their
overlap (|top3_resp ∩ top3_branched|), and the max ``score`` (is any variable
responsive at all?). Then the kill-criterion verdict:

  * If on ≥ 4 of the 6 Class-P instances the current policy already branches the
    responsive vars (overlap ≥ 2/3), selection is not the lever → SKIP R3b.
  * Instances where responsiveness is flat (no variable moves the root bound
    > 1 % on box-halving) are flagged relaxation-limited (R4 territory).
  * Otherwise BUILD R3b; the st_e36 F5 prediction (x0 responsive / x1 branched)
    is confirmed or refuted explicitly.

House style per t11/t12/t21: ``JAX_PLATFORMS=cpu`` + ``JAX_ENABLE_X64=1`` set
before importing discopt; standalone runnable; the reform gating that fixes the
branch/fingerprint variable space is reproduced faithfully from
``solver.solve_model``.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import argparse
import json
import math
import time

import discopt.modeling as dm
import discopt.solver as solver
import numpy as np

# Reuse the T2.1 root-bound machinery verbatim so the fingerprint's relaxation
# is exactly the one the solver's root path builds.
from t21_root_loop_replay import (  # noqa: E402
    ORACLE,
    flat_bounds,
    resolve_nl,
    root_lp_bound,
)

DEFAULT_SNAPSHOT = os.path.expanduser("~/Dropbox/projects/discopt-minlp-benchmark/minlplib/nl")

# The R3a panel (plan §3 R3a): 6 Class-P + 5 R1-resistant Class-H.
CLASS_P = ["st_e36", "nvs05", "nvs09", "tanksize", "tls2", "hda"]
CLASS_H = ["casctanks", "4stufen", "beuster", "heatexch_gen1", "bchoco06"]
PANEL = CLASS_P + CLASS_H

# Oracles for Class-H not covered by t21's ORACLE (minlplib.solu, =opt=/=best=).
ORACLE_H: dict[str, float] = {
    "nvs09": -43.134336,
    "hda": -5964.5,
    "casctanks": 9.163479388,
    "4stufen": 96908.35577,
    "beuster": 116655.4796,
    "heatexch_gen1": 149112.4623,
    "bchoco06": -390.0,
}


def get_oracle(name: str) -> float | None:
    if name in ORACLE:
        return ORACLE[name]
    return ORACLE_H.get(name)


def reformed_model_and_names(path: str):
    """Return (live_model, var_names, prereform_nvars).

    Reproduces the factorable-reform gating from ``solver.solve_model`` so the
    fingerprint and the branch counter share the exact same flat variable space.
    The reform fires only when ``has_factorable_work`` AND the model classifies as
    provably-nonconvex — identical to the live solve path. ``var_names`` labels
    every reformed flat column; the first ``prereform_nvars`` are the originals.
    """
    from discopt._jax.factorable_reform import (
        factorable_reformulate,
        has_factorable_work,
    )

    model = dm.from_nl(path)
    prereform_nvars = sum(v.size for v in model._variables)
    if has_factorable_work(model):
        try:
            ok, convex, _ = solver._classify_model_convexity(model)
        except Exception:
            ok, convex = False, True
        if ok and not convex:
            model = factorable_reformulate(model)
    names = _flat_var_names(model)
    return model, names, prereform_nvars


def _flat_var_names(model) -> list[str]:
    names: list[str] = []
    for v in model._variables:
        base = getattr(v, "name", None) or f"v{len(names)}"
        if v.size == 1:
            names.append(str(base))
        else:
            names.extend(f"{base}[{k}]" for k in range(v.size))
    return names


def fingerprint(model, lb: np.ndarray, ub: np.ndarray) -> tuple[np.ndarray, float | None]:
    """Box-shrink responsiveness score per flat variable.

    ``score[v] = |root_bound(better half of v's box) − root_bound(full box)|``.
    Returns (scores, full_root_bound). Variables with a non-finite or
    zero-width box are scored 0 (nothing to halve).
    """
    n = len(lb)
    full_bound, _ = root_lp_bound(model, lb, ub)
    scores = np.zeros(n, dtype=np.float64)
    if full_bound is None or not math.isfinite(full_bound):
        return scores, full_bound
    for i in range(n):
        lo, hi = float(lb[i]), float(ub[i])
        if not (math.isfinite(lo) and math.isfinite(hi)) or (hi - lo) <= 1e-12:
            continue
        mid = 0.5 * (lo + hi)
        best_delta = 0.0
        for half in ("lower", "upper"):
            hlb, hub = lb.copy(), ub.copy()
            if half == "lower":
                hub[i] = mid
            else:
                hlb[i] = mid
            b, _ = root_lp_bound(model, hlb, hub)
            if b is None or not math.isfinite(b):
                continue
            # Internally-minimized sense: a *larger* root bound is the better
            # (tighter) half. Score the magnitude of the move on the half that
            # does not worsen the bound (b >= full_bound within tolerance); a
            # half that only loosens the bound carries no responsiveness signal.
            delta = abs(b - full_bound)
            if b >= full_bound - 1e-9 and delta > best_delta:
                best_delta = delta
        scores[i] = best_delta
    return scores, full_bound


def instrumented_branch_counts(path: str, time_limit: float) -> tuple[list[int], dict]:
    """Run one solve with the branch-count sink armed; return (counts, info)."""
    sink: dict = {}
    prev = solver._R3A_BRANCH_COUNT_SINK
    solver._R3A_BRANCH_COUNT_SINK = sink
    try:
        m = dm.from_nl(path)
        t0 = time.perf_counter()
        r = m.solve(time_limit=time_limit)
        wall = time.perf_counter() - t0
    finally:
        solver._R3A_BRANCH_COUNT_SINK = prev
    counts = sink.get("branch_var_counts") or []
    info = {
        "status": str(r.status),
        "nodes": r.node_count,
        "objective": r.objective,
        "bound": r.bound,
        "wall": wall,
    }
    return list(counts), info


def top3(values: np.ndarray) -> list[int]:
    """Indices of the top-3 by value (descending), dropping zero/nonpositive."""
    order = np.argsort(-values, kind="stable")
    out = [int(i) for i in order if values[int(i)] > 0]
    return out[:3]


def run_instance(name: str, path: str, time_limit: float) -> dict:
    oracle = get_oracle(name)
    model, names, prereform_nvars = reformed_model_and_names(path)
    lb, ub = flat_bounds(model)
    n = len(lb)

    scores, full_bound = fingerprint(model, lb, ub)

    counts, solve_info = instrumented_branch_counts(path, time_limit)
    # Align the counter length to the fingerprint space. They should match (both
    # are the reformed model). If the live solve took a different path (e.g. the
    # reform did not fire there), flag it rather than silently mis-align.
    counts_arr = np.zeros(n, dtype=np.int64)
    aligned = len(counts) == n
    if aligned:
        counts_arr = np.asarray(counts, dtype=np.int64)
    elif len(counts) >= n:
        # live space is a superset (extra aux appended after originals): the
        # first n entries still correspond 1:1 to the fingerprint columns.
        counts_arr = np.asarray(counts[:n], dtype=np.int64)
    else:
        # live space smaller — pad, and flag the mismatch loudly.
        counts_arr[: len(counts)] = np.asarray(counts, dtype=np.int64)

    resp_top = top3(scores)
    branch_top = top3(counts_arr.astype(np.float64))
    overlap = len(set(resp_top) & set(branch_top))

    max_score = float(np.max(scores)) if n else 0.0
    # "flat" = no variable moves the root bound > 1% of the bound magnitude.
    scale = max(1.0, abs(full_bound)) if full_bound is not None else 1.0
    flat = (max_score / scale) < 0.01

    return {
        "name": name,
        "class": "P" if name in CLASS_P else "H",
        "oracle": oracle,
        "n_vars": n,
        "prereform_nvars": prereform_nvars,
        "names": names,
        "full_root_bound": full_bound,
        "scores": scores.tolist(),
        "branch_counts": counts_arr.tolist(),
        "counts_len_raw": len(counts),
        "counts_aligned": aligned,
        "resp_top3": resp_top,
        "branch_top3": branch_top,
        "overlap_top3": overlap,
        "max_score": max_score,
        "max_score_rel": max_score / scale,
        "flat_responsiveness": bool(flat),
        "solve": solve_info,
    }


def _lab(names: list[str], idxs: list[int]) -> str:
    if not idxs:
        return "-"
    return ", ".join(f"{names[i]}" for i in idxs)


def print_report(results: list[dict]) -> dict:
    print("\n" + "=" * 96)
    print("R3a — BOUND-RESPONSIVENESS FINGERPRINT vs ACTUAL BRANCHING")
    print("=" * 96)

    print(
        f"\n{'instance':16s} {'cls':3s} {'top3 responsive':28s} "
        f"{'top3 branched':22s} {'ovlp':4s} {'maxΔ%':7s} {'flat?':5s}"
    )
    print("-" * 96)
    for r in results:
        names = r["names"]
        rt = _lab(names, r["resp_top3"])
        bt = _lab(names, r["branch_top3"])
        print(
            f"{r['name']:16s} {r['class']:3s} {rt:28.28s} {bt:22.22s} "
            f"{str(r['overlap_top3']) + '/3':4s} {100 * r['max_score_rel']:6.2f}% "
            f"{'YES' if r['flat_responsiveness'] else 'no':5s}"
        )

    # Detailed per-instance score/count dump (for the plan doc table).
    print("\n--- Per-instance detail (score / branch-count by variable) ---")
    for r in results:
        names = r["names"]
        scores = r["scores"]
        counts = r["branch_counts"]
        print(
            f"\n{r['name']} (class {r['class']}, {r['n_vars']} reformed vars, "
            f"{r['prereform_nvars']} original; full_root_bound="
            f"{_fmt(r['full_root_bound'])}, oracle={_fmt(r['oracle'])}, "
            f"solve={r['solve']['status']}/{r['solve']['nodes']} nodes):"
        )
        order = np.argsort(-np.asarray(scores), kind="stable")
        for j in order:
            j = int(j)
            if scores[j] <= 0 and counts[j] == 0:
                continue
            print(f"    {names[j]:14s} score={scores[j]:12.5g}  branches={counts[j]}")

    # ---- Kill criterion (plan §3 R3a) ----
    class_p = [r for r in results if r["class"] == "P"]
    # An instance "already branches the responsive vars" if overlap >= 2/3.
    p_already = [r for r in class_p if r["overlap_top3"] >= 2]
    n_p = len(class_p)
    skip = len(p_already) >= 4  # >=4 of the 6 Class-P

    flat_instances = [r["name"] for r in results if r["flat_responsiveness"]]

    # st_e36 F5 prediction: x0 responsive (top), x1 branched (top).
    st = next((r for r in results if r["name"] == "st_e36"), None)
    st_verdict = None
    if st is not None:
        names = st["names"]
        x0 = names.index("x0") if "x0" in names else None
        x1 = names.index("x1") if "x1" in names else None
        resp_is_x0 = x0 is not None and x0 in st["resp_top3"][:1]
        branch_has_x1 = x1 is not None and x1 in st["branch_top3"]
        # F5 prediction is x0-responsive AND x1-among-branched AND x0 NOT the top
        # branched (i.e. the responsive var is under-branched relative to x1).
        x0_branches = st["branch_counts"][x0] if x0 is not None else -1
        x1_branches = st["branch_counts"][x1] if x1 is not None else -1
        st_verdict = {
            "x0_index": x0,
            "x1_index": x1,
            "x0_score": st["scores"][x0] if x0 is not None else None,
            "x1_score": st["scores"][x1] if x1 is not None else None,
            "x0_branches": x0_branches,
            "x1_branches": x1_branches,
            "resp_top_is_x0": bool(resp_is_x0),
            "x1_in_branch_top3": bool(branch_has_x1),
            "prediction_held": bool(resp_is_x0 and x1_branches > x0_branches),
        }

    print("\n" + "=" * 96)
    print("KILL CRITERION (plan §3 R3a)")
    print("=" * 96)
    print(
        f"Class-P instances already branching responsive vars (overlap >= 2/3): "
        f"{len(p_already)}/{n_p}  -> "
        + (", ".join(r["name"] for r in p_already) if p_already else "(none)")
    )
    print(
        f"Flat / relaxation-limited instances (max root move < 1%): "
        f"{flat_instances if flat_instances else '(none)'}"
    )
    if st_verdict is not None:
        print("\nst_e36 F5 prediction (x0 responsive / x1 branched):")
        print(f"    x0 score={_fmt(st_verdict['x0_score'])}  branches={st_verdict['x0_branches']}")
        print(f"    x1 score={_fmt(st_verdict['x1_score'])}  branches={st_verdict['x1_branches']}")
        print(
            f"    resp-top is x0? {st_verdict['resp_top_is_x0']}   "
            f"x1 in branch-top3? {st_verdict['x1_in_branch_top3']}   "
            f"=> prediction {'HELD' if st_verdict['prediction_held'] else 'REFUTED'}"
        )

    verdict = "SKIP R3b" if skip else "BUILD R3b"
    print(
        f"\n==> VERDICT: {verdict}  "
        f"(Class-P overlap>=2/3 count = {len(p_already)}/{n_p}; "
        f"SKIP requires >= 4)"
    )
    print("=" * 96)

    return {
        "class_p_already_branching": [r["name"] for r in p_already],
        "n_class_p_already": len(p_already),
        "n_class_p": n_p,
        "flat_instances": flat_instances,
        "st_e36": st_verdict,
        "verdict": verdict,
    }


def _fmt(x) -> str:
    if x is None:
        return "n/a"
    try:
        return f"{float(x):.6g}"
    except Exception:
        return str(x)


def main() -> None:
    ap = argparse.ArgumentParser(description="R3a bound-responsiveness fingerprint")
    ap.add_argument("--snapshot", default=DEFAULT_SNAPSHOT)
    ap.add_argument("--instances", default=None, help="comma-separated subset of the panel")
    ap.add_argument("--time-limit", type=float, default=60.0)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    panel = PANEL if args.instances is None else [s.strip() for s in args.instances.split(",")]

    results: list[dict] = []
    for name in panel:
        path = resolve_nl(name, args.snapshot)
        if path is None:
            print(f"[{name}] NL not found -> SKIP")
            continue
        print(f"\n########## {name} ##########")
        t0 = time.perf_counter()
        try:
            r = run_instance(name, path, args.time_limit)
            r["total_wall"] = time.perf_counter() - t0
            results.append(r)
            print(
                f"[{name}] done in {r['total_wall']:.1f}s  "
                f"resp_top3={r['resp_top3']} branch_top3={r['branch_top3']} "
                f"overlap={r['overlap_top3']}/3 max_move={100 * r['max_score_rel']:.2f}% "
                f"flat={r['flat_responsiveness']}"
            )
        except Exception as e:  # pragma: no cover - keep the panel going
            import traceback

            print(f"[{name}] ERROR: {type(e).__name__}: {e}")
            traceback.print_exc()

    verdict = print_report(results)

    if args.json_out:
        with open(args.json_out, "w") as fh:
            json.dump({"results": results, "verdict": verdict}, fh, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()

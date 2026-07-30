#!/usr/bin/env python3
"""Entry experiment for Card 3a(b) item 3 — "OBBT entry points 3 → 1".

**The card's claim.** ``obbt_tighten_root`` becomes the single door with modes;
``run_obbt`` / ``run_obbt_on_relaxation`` become internal.

**Hypothesis (H-OBBT).** The three are three doors to one room: they compute the
same root tightening, so two can be made private behind the third.

**Kill criterion.** Any instance where two doors given the same model and the same
starting box return *different* boxes proves they are different tightenings, not
different spellings — and both stay (the Phase 2/3 lesson).

**What is compared.** Only the pair that is even type-compatible:
``run_obbt(model, lb, ub, ...)`` and ``obbt_tighten_root(model, lb, ub, ...)``.
``run_obbt_on_relaxation`` takes a **pre-built relaxation object**, not a ``Model``,
so it is not substitutable for either without inventing the relaxation — that is
recorded as a structural finding rather than measured.

Per CLAUDE.md §6 executed counts are printed and zero exits non-zero; per §7
nothing is swallowed; per §8 the loaded tree is asserted first.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import time
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt
import discopt._jax.obbt as obbt_mod
import numpy as np
from discopt.modeling.core import from_nl

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))


def _assert_loaded() -> None:
    print(f"discopt.__file__ = {discopt.__file__}", flush=True)
    print(f"obbt.__file__    = {obbt_mod.__file__}", flush=True)
    for attr in ("run_obbt", "run_obbt_on_relaxation", "obbt_tighten_root"):
        if not hasattr(obbt_mod, attr):
            raise SystemExit(f"ABORT: obbt.{attr} absent (CLAUDE.md §8).")
    print("all three OBBT doors present on the loaded tree", flush=True)


def _root_box(model):
    lb, ub = [], []
    for v in model._variables:
        lb.extend(np.asarray(v.lb, dtype=np.float64).ravel().tolist())
        ub.extend(np.asarray(v.ub, dtype=np.float64).ravel().tolist())
    return np.asarray(lb), np.asarray(ub)


def main() -> int:
    _assert_loaded()

    corpus = sorted(
        glob.glob(os.path.join(_REPO, "python", "tests", "data", "minlplib_nl", "*.nl"))
    )
    limit = int(os.environ.get("CARD3A_OBBT_INSTANCES", "25"))
    corpus = corpus[:limit]
    print(f"corpus instances: {len(corpus)}", flush=True)
    if not corpus:
        raise SystemExit("ABORT: corpus empty (§6).")

    n_instances = 0
    n_col_comparisons = 0
    n_disagree = 0
    n_tighter_root = 0
    n_tighter_runobbt = 0
    errors: list[str] = []
    per_instance: list[dict] = []

    for path in corpus:
        name = os.path.basename(path)[:-3]
        try:
            m = from_nl(path)
            lb0, ub0 = _root_box(m)
            a = obbt_mod.run_obbt(
                m, lb0.copy(), ub0.copy(), time_limit_per_lp=0.2, total_time_limit=20.0
            )
            b = obbt_mod.obbt_tighten_root(
                m,
                lb0.copy(),
                ub0.copy(),
                rounds=1,
                time_limit_per_lp=0.2,
                deadline=time.perf_counter() + 20.0,
            )
        except Exception as exc:  # noqa: BLE001 — reported, never swallowed (§7)
            errors.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  ERROR {name}: {type(exc).__name__}: {exc}", flush=True)
            continue

        # NOTE the differing return contracts, itself a finding: ``run_obbt``
        # returns ``ObbtResult(tightened_lb, tightened_ub, ...)`` while
        # ``obbt_tighten_root`` returns ``RootObbtResult(lb, ub, ..., infeasible)``.
        a_lb = np.asarray(a.tightened_lb, dtype=np.float64)
        a_ub = np.asarray(a.tightened_ub, dtype=np.float64)
        b_lb = np.asarray(b.lb, dtype=np.float64)
        b_ub = np.asarray(b.ub, dtype=np.float64)
        n = min(a_lb.size, b_lb.size)
        if n == 0:
            errors.append(f"{name}: zero-width boxes returned")
            continue
        n_instances += 1
        inst_dis = 0
        for j in range(n):
            n_col_comparisons += 2
            if abs(a_lb[j] - b_lb[j]) > 1e-7:
                n_disagree += 1
                inst_dis += 1
                if b_lb[j] > a_lb[j]:
                    n_tighter_root += 1
                else:
                    n_tighter_runobbt += 1
            if abs(a_ub[j] - b_ub[j]) > 1e-7:
                n_disagree += 1
                inst_dis += 1
                if b_ub[j] < a_ub[j]:
                    n_tighter_root += 1
                else:
                    n_tighter_runobbt += 1
        per_instance.append({"instance": name, "cols": int(n), "disagreements": inst_dis})
        print(f"  {name:28s} cols={n:5d} disagreements={inst_dis}", flush=True)

    print("\n" + "=" * 72, flush=True)
    print(f"INSTANCES_COMPARED        {n_instances}", flush=True)
    print(f"ERRORS                    {len(errors)}", flush=True)
    print(f"EXECUTED_BOUND_COMPARISONS{n_col_comparisons:>6}", flush=True)
    print(f"DISAGREEMENTS             {n_disagree}", flush=True)
    print(f"  obbt_tighten_root tighter {n_tighter_root}", flush=True)
    print(f"  run_obbt tighter          {n_tighter_runobbt}", flush=True)
    for e in errors:
        print(f"  error: {e}", flush=True)

    out_path = os.path.join(_REPO, "reports", "card3a_obbt_doors_entry.json")
    with open(out_path, "w") as fh:
        json.dump(
            {
                "instances_compared": n_instances,
                "errors": errors,
                "executed_bound_comparisons": n_col_comparisons,
                "disagreements": n_disagree,
                "obbt_tighten_root_tighter": n_tighter_root,
                "run_obbt_tighter": n_tighter_runobbt,
                "per_instance": per_instance,
            },
            fh,
            indent=2,
        )
    print(f"wrote {out_path}", flush=True)

    if n_col_comparisons == 0:
        print("VACUOUS: zero bound comparisons executed.", flush=True)
        return 2
    if n_disagree:
        print(
            "H-OBBT FALSIFIED: run_obbt and obbt_tighten_root return different boxes "
            "on the same model and box — they are different tightenings, not two "
            "spellings of one. Both stay.",
            flush=True,
        )
        return 3
    print("H-OBBT HOLDS on this population for the run_obbt / obbt_tighten_root pair.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — crash loudly (§7)
        traceback.print_exc()
        sys.exit(1)

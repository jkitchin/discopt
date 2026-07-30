#!/usr/bin/env python3
"""Entry experiment for Card 3a(b) item 2 — "one Python reduced-cost-fixing".

**The card's claim.** There are ~5 Python-side reduced-cost-fixing
implementations; consolidate onto the live MILP-driver kernel semantics and delete
``_jax/node_reduce._dbbt_from_reduced_costs`` and
``solver._reduced_cost_fixing``.

**Hypothesis (H-RCF).** ``_dbbt_from_reduced_costs`` and ``_reduced_cost_fixing``
compute the same tightening on the integer columns, so one can be deleted in favour
of the other (Regime C — deleting an *active* tightening can loosen bounds).

**Kill criterion.** One integer bound on which the two disagree, on inputs drawn
from real corpus node LPs, falsifies H-RCF: the "duplicates" are differentiated and
both stay (Phase 2/3's repeated lesson, restated in the card).

**Method.** Real inputs, not synthetic. **Both** entry points are wrapped, so
whichever one the corpus actually reaches supplies the population:

* ``solver._reduced_cost_fixing`` — the ROOT MILP path, reachable on **defaults**
  (``_root_reduced_cost_fixing``, solver/__init__.py:16501, no flag);
* ``node_reduce._dbbt_from_reduced_costs`` — the per-NODE path, reachable only
  under ``DISCOPT_PHASE2_DBBT=1`` (set here) *and* an active McCormick LP relaxer.

Every live invocation records its exact arguments; each captured tuple is then
replayed through the **other** implementation and the resulting INTEGER bounds
compared elementwise. Continuous columns are outside ``_reduced_cost_fixing``'s
domain by construction (it iterates ``int_idx``) and are excluded from the bound
comparison — that asymmetry is counted and reported separately, since a deletion
would silently drop it.

Per CLAUDE.md §6 the probe prints executed-comparison counts and exits non-zero
when zero; per §7 nothing is swallowed; per §8 the loaded module is asserted first.
"""

from __future__ import annotations

import glob
import json
import os
import sys
import traceback
from typing import Any

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")
os.environ["DISCOPT_PHASE2_DBBT"] = "1"

import discopt
import discopt._jax.node_reduce as node_reduce
import discopt.solver as solver_pkg
import numpy as np
from discopt.modeling.core import from_nl

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

CAPTURES: list[dict[str, Any]] = []
MAX_CAPTURES = 4000


def _assert_loaded() -> None:
    print(f"discopt.__file__       = {discopt.__file__}", flush=True)
    print(f"node_reduce.__file__   = {node_reduce.__file__}", flush=True)
    print(f"solver.__file__        = {solver_pkg.__file__}", flush=True)
    for mod, attr in (
        (node_reduce, "_dbbt_from_reduced_costs"),
        (solver_pkg, "_reduced_cost_fixing"),
    ):
        if not hasattr(mod, attr):
            raise SystemExit(
                f"ABORT: {mod.__name__}.{attr} is absent — this is not the tree under "
                f"test, and the probe would compare nothing (CLAUDE.md §8)."
            )
    print("both implementations present on the loaded tree", flush=True)


_orig_dbbt = node_reduce._dbbt_from_reduced_costs
_orig_rcf = solver_pkg._reduced_cost_fixing


def _capturing_dbbt(node_lb, node_ub, reduced_costs, z_lp, cutoff, is_int):
    out = _orig_dbbt(node_lb, node_ub, reduced_costs, z_lp, cutoff, is_int)
    if len(CAPTURES) < MAX_CAPTURES:
        CAPTURES.append(
            {
                "seam": "node_dbbt",
                "lb": np.array(node_lb, dtype=np.float64),
                "ub": np.array(node_ub, dtype=np.float64),
                "rc": np.array(reduced_costs, dtype=np.float64),
                "z_lp": float(z_lp),
                "cutoff": float(cutoff),
                "is_int": np.array(is_int, dtype=bool),
            }
        )
    return out


def _capturing_rcf(lb, ub, int_idx, reduced_costs, z_lp, z_inc):
    out = _orig_rcf(lb, ub, int_idx, reduced_costs, z_lp, z_inc)
    if len(CAPTURES) < MAX_CAPTURES:
        lb_a = np.array(lb, dtype=np.float64)
        is_int = np.zeros(lb_a.size, dtype=bool)
        idx = np.asarray(list(int_idx), dtype=int)
        idx = idx[(idx >= 0) & (idx < lb_a.size)]
        is_int[idx] = True
        CAPTURES.append(
            {
                "seam": "root_rcf",
                "lb": lb_a,
                "ub": np.array(ub, dtype=np.float64),
                "rc": np.array(reduced_costs, dtype=np.float64),
                "z_lp": float(z_lp),
                "cutoff": float(z_inc),
                "is_int": is_int,
            }
        )
    return out


def main() -> int:
    _assert_loaded()
    node_reduce._dbbt_from_reduced_costs = _capturing_dbbt
    solver_pkg._reduced_cost_fixing = _capturing_rcf

    corpus = sorted(
        glob.glob(os.path.join(_REPO, "python", "tests", "data", "minlplib_nl", "*.nl"))
    )
    budget = float(os.environ.get("CARD3A_RCF_BUDGET", "12"))
    limit = int(os.environ.get("CARD3A_RCF_INSTANCES", "40"))
    corpus = corpus[:limit]
    print(f"corpus instances: {len(corpus)}, per-instance budget {budget}s", flush=True)
    if not corpus:
        raise SystemExit("ABORT: corpus empty (§6).")

    solved = 0
    failures: list[str] = []
    for path in corpus:
        name = os.path.basename(path)[:-3]
        before = len(CAPTURES)
        try:
            m = from_nl(path)
            m.solve(time_limit=budget, verify_incumbent=False)
            solved += 1
        except Exception as exc:  # noqa: BLE001 — reported, never swallowed (§7)
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  SOLVE-FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        print(
            f"  {name:28s} captures +{len(CAPTURES) - before} (total {len(CAPTURES)})",
            flush=True,
        )
        if len(CAPTURES) >= MAX_CAPTURES:
            print("  capture cap reached; stopping the sweep", flush=True)
            break

    node_reduce._dbbt_from_reduced_costs = _orig_dbbt
    solver_pkg._reduced_cost_fixing = _orig_rcf

    # ── Replay BOTH implementations on every captured input ────────────────
    n_inputs = 0
    n_int_cols = 0
    n_lb_disagree = 0
    n_ub_disagree = 0
    n_cont_tightened_by_dbbt_only = 0
    seams: dict[str, int] = {}
    examples: list[dict[str, Any]] = []

    for cap in CAPTURES:
        n_inputs += 1
        seams[cap["seam"]] = seams.get(cap["seam"], 0) + 1
        is_int = cap["is_int"]
        int_idx = np.nonzero(is_int)[0]
        # ``_reduced_cost_fixing(lb, ub, int_idx, reduced_costs, z_lp, z_inc)``
        a_lb, a_ub, _ = _orig_rcf(
            cap["lb"], cap["ub"], int_idx, cap["rc"], cap["z_lp"], cap["cutoff"]
        )
        # ``_dbbt_from_reduced_costs(lb, ub, rc, z_lp, cutoff, is_int)``
        b_lb, b_ub, _, _ = _orig_dbbt(
            cap["lb"], cap["ub"], cap["rc"], cap["z_lp"], cap["cutoff"], is_int
        )

        for j in int_idx:
            n_int_cols += 1
            dl = abs(float(a_lb[j]) - float(b_lb[j]))
            du = abs(float(a_ub[j]) - float(b_ub[j]))
            if dl > 1e-9:
                n_lb_disagree += 1
                if len(examples) < 12:
                    examples.append(
                        {
                            "col": int(j),
                            "which": "lb",
                            "rcf": float(a_lb[j]),
                            "dbbt": float(b_lb[j]),
                            "in_lb": float(cap["lb"][j]),
                            "in_ub": float(cap["ub"][j]),
                            "rc": float(cap["rc"][j]),
                            "gap": cap["cutoff"] - cap["z_lp"],
                        }
                    )
            if du > 1e-9:
                n_ub_disagree += 1
                if len(examples) < 12:
                    examples.append(
                        {
                            "col": int(j),
                            "which": "ub",
                            "rcf": float(a_ub[j]),
                            "dbbt": float(b_ub[j]),
                            "in_lb": float(cap["lb"][j]),
                            "in_ub": float(cap["ub"][j]),
                            "rc": float(cap["rc"][j]),
                            "gap": cap["cutoff"] - cap["z_lp"],
                        }
                    )

        cont = np.nonzero(~is_int)[0]
        for j in cont:
            if abs(float(b_lb[j]) - float(cap["lb"][j])) > 1e-9 or (
                abs(float(b_ub[j]) - float(cap["ub"][j])) > 1e-9
            ):
                n_cont_tightened_by_dbbt_only += 1

    print("\n" + "=" * 72, flush=True)
    print(f"INSTANCES_SOLVED              {solved}", flush=True)
    print(f"SOLVE_FAILURES                {len(failures)}", flush=True)
    print(f"CAPTURED_INPUTS               {n_inputs}  by seam: {seams}", flush=True)
    print(f"EXECUTED_INT_COL_COMPARISONS  {n_int_cols}", flush=True)
    print(f"LB_DISAGREEMENTS              {n_lb_disagree}", flush=True)
    print(f"UB_DISAGREEMENTS              {n_ub_disagree}", flush=True)
    print(f"CONT_COLS_TIGHTENED_DBBT_ONLY {n_cont_tightened_by_dbbt_only}", flush=True)
    for e in examples:
        print(f"  EXAMPLE {e}", flush=True)
    for f in failures:
        print(f"  solve failure: {f}", flush=True)

    out_path = os.path.join(_REPO, "reports", "card3a_rcf_dedup_entry.json")
    with open(out_path, "w") as fh:
        json.dump(
            {
                "instances_solved": solved,
                "solve_failures": failures,
                "captured_inputs": n_inputs,
                "captures_by_seam": seams,
                "executed_int_col_comparisons": n_int_cols,
                "lb_disagreements": n_lb_disagree,
                "ub_disagreements": n_ub_disagree,
                "cont_cols_tightened_dbbt_only": n_cont_tightened_by_dbbt_only,
                "examples": examples,
            },
            fh,
            indent=2,
        )
    print(f"wrote {out_path}", flush=True)

    if n_int_cols == 0:
        print(
            "VACUOUS: zero integer-column comparisons executed — the node DBBT never "
            "fired, so this probe measured nothing and must not be read as agreement.",
            flush=True,
        )
        return 2
    if n_lb_disagree or n_ub_disagree:
        print("H-RCF FALSIFIED: the two implementations disagree on real inputs.", flush=True)
        return 3
    print("H-RCF HOLDS on the integer columns of this population.", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — crash loudly (§7)
        traceback.print_exc()
        sys.exit(1)

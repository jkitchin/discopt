#!/usr/bin/env python3
"""Entry experiment for Card 3a(b) item 1 — "a single `tighten_root_bounds_with_fbbt`
invocation policy".

**The card's claim.** ``tighten_root_bounds_with_fbbt`` has "5+ uncoordinated call
sites"; Card 3a(b) wants one invocation policy.

**Hypothesis (H-FBBT-INV).** More than one call site fires within a *single* solve,
so the root FBBT runs redundantly and an invocation policy would remove real work.

**Kill criterion.** If every solve invokes it at most once, there is no redundant
invocation to coordinate: the "5 sites" are five *engines* each doing its own root
presolve exactly once, and the card's item 1 is already satisfied by construction.
A policy object would then be pure ceremony — and per §0.4 / the Phase 2-3 lesson
("these duplicates keep turning out to be differentiated") the right outcome is to
leave them.

**Method.** Wrap ``discopt.solvers._root_presolve.tighten_root_bounds_with_fbbt``,
count invocations per solve and record each one's ``root_changed`` flag and the
caller's file:line. Real corpus instances; no synthetic proxy.

Per CLAUDE.md §6 executed counts are printed and a zero count exits non-zero; per §7
nothing is swallowed; per §8 the loaded tree is asserted first.
"""

from __future__ import annotations

import collections
import glob
import json
import os
import sys
import traceback

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("JAX_ENABLE_X64", "1")

import discopt
import discopt.solvers._root_presolve as rp
from discopt.modeling.core import from_nl

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(os.path.dirname(_HERE))

PER_SOLVE: list[dict] = []
_current: list[dict] = []


def _assert_loaded() -> None:
    print(f"discopt.__file__      = {discopt.__file__}", flush=True)
    print(f"_root_presolve.__file__= {rp.__file__}", flush=True)
    if not hasattr(rp, "tighten_root_bounds_with_fbbt"):
        raise SystemExit("ABORT: tighten_root_bounds_with_fbbt absent (CLAUDE.md §8).")


_orig = rp.tighten_root_bounds_with_fbbt


def _wrapped(*args, **kwargs):
    frame = sys._getframe(1)
    caller = f"{os.path.basename(frame.f_code.co_filename)}:{frame.f_lineno}"
    out = _orig(*args, **kwargs)
    changed = None
    infeasible = None
    try:
        infeasible = bool(out[2])
        changed = bool(out[3])
    except (IndexError, TypeError):
        pass
    _current.append({"caller": caller, "root_changed": changed, "root_infeasible": infeasible})
    return out


def main() -> int:
    _assert_loaded()
    # The wrapper must be installed where the CALLERS look it up. Every site does
    # `from discopt.solvers._root_presolve import tighten_root_bounds_with_fbbt`
    # *inside* the function, so patching the module attribute is sufficient and is
    # picked up on the next call — asserted below by a non-zero invocation count.
    rp.tighten_root_bounds_with_fbbt = _wrapped

    corpus = sorted(
        glob.glob(os.path.join(_REPO, "python", "tests", "data", "minlplib_nl", "*.nl"))
    )
    limit = int(os.environ.get("CARD3A_FBBT_INSTANCES", "66"))
    budget = float(os.environ.get("CARD3A_FBBT_BUDGET", "8"))
    corpus = corpus[:limit]
    print(f"corpus instances: {len(corpus)}, budget {budget}s", flush=True)
    if not corpus:
        raise SystemExit("ABORT: corpus empty (§6).")

    failures: list[str] = []
    for path in corpus:
        name = os.path.basename(path)[:-3]
        _current.clear()
        try:
            m = from_nl(path)
            m.solve(time_limit=budget, verify_incumbent=False)
        except Exception as exc:  # noqa: BLE001 — reported, never swallowed (§7)
            failures.append(f"{name}: {type(exc).__name__}: {exc}")
            print(f"  SOLVE-FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
            continue
        rec = {"instance": name, "invocations": list(_current)}
        PER_SOLVE.append(rec)
        print(
            f"  {name:28s} invocations={len(_current)} {[c['caller'] for c in _current]}",
            flush=True,
        )

    rp.tighten_root_bounds_with_fbbt = _orig

    total = sum(len(r["invocations"]) for r in PER_SOLVE)
    hist = collections.Counter(len(r["invocations"]) for r in PER_SOLVE)
    callers = collections.Counter(c["caller"] for r in PER_SOLVE for c in r["invocations"])
    multi = [r["instance"] for r in PER_SOLVE if len(r["invocations"]) > 1]
    # Of the solves with >1 invocation, how many had a SECOND call that changed
    # anything? A second call that tightens nothing is the redundancy the card wants
    # removed; a second call that tightens is a differentiated, load-bearing call.
    redundant = 0
    load_bearing = 0
    for r in PER_SOLVE:
        for c in r["invocations"][1:]:
            if c["root_changed"]:
                load_bearing += 1
            else:
                redundant += 1

    print("\n" + "=" * 72, flush=True)
    print(f"SOLVES_MEASURED            {len(PER_SOLVE)}", flush=True)
    print(f"SOLVE_FAILURES             {len(failures)}", flush=True)
    print(f"EXECUTED_INVOCATIONS       {total}", flush=True)
    print(f"INVOCATIONS_PER_SOLVE_HIST {dict(sorted(hist.items()))}", flush=True)
    print(f"CALLER_SITES               {dict(callers)}", flush=True)
    print(f"SOLVES_WITH_MULTIPLE       {len(multi)} {multi[:20]}", flush=True)
    print(f"SECOND+_CALLS_REDUNDANT    {redundant}", flush=True)
    print(f"SECOND+_CALLS_LOAD_BEARING {load_bearing}", flush=True)
    for f in failures:
        print(f"  solve failure: {f}", flush=True)

    out_path = os.path.join(_REPO, "reports", "card3a_root_fbbt_invocations.json")
    with open(out_path, "w") as fh:
        json.dump(
            {
                "solves_measured": len(PER_SOLVE),
                "solve_failures": failures,
                "executed_invocations": total,
                "invocations_per_solve_hist": {str(k): v for k, v in sorted(hist.items())},
                "caller_sites": dict(callers),
                "solves_with_multiple": multi,
                "second_plus_redundant": redundant,
                "second_plus_load_bearing": load_bearing,
                "per_solve": PER_SOLVE,
            },
            fh,
            indent=2,
        )
    print(f"wrote {out_path}", flush=True)

    if total == 0:
        print("VACUOUS: zero invocations observed — the wrapper never fired.", flush=True)
        return 2
    if multi:
        print(
            "H-FBBT-INV HOLDS: at least one solve invokes the root FBBT more than once.", flush=True
        )
        return 0
    print(
        "H-FBBT-INV FALSIFIED: every solve invokes the root FBBT at most once. The "
        "'5+ call sites' are five ENGINES each doing its own root presolve exactly "
        "once; there is no redundant invocation for a policy object to coordinate.",
        flush=True,
    )
    return 3


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception:  # noqa: BLE001 — crash loudly (§7)
        traceback.print_exc()
        sys.exit(1)

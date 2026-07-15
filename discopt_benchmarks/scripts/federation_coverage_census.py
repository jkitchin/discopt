"""Federation coverage census (#632) — where the claim federation leaves holes.

For each of the 62 vendored ``python/tests/data/minlplib_nl/*.nl`` instances this
read-only census records, using discopt's OWN in-house Rust simplex for any bound
(never scipy/HiGHS):

* whether ``build_milp_relaxation`` dropped the objective to the separable /
  feasibility fallback (no genuine per-atom envelope applied) — the coverage HOLE;
* whether the root LP produced a finite bound at all;
* the canonical-DAG atom-kind histogram of the model (``canonical_expr.atomize``),
  so a fallback is attributed to the atom CLASS that caused it, not the instance.

This is the evidence backing ``docs/dev/factorable-capability-blueprint.md`` §1/§2.
It is a MEASUREMENT tool — it changes no solver/relaxation code.

Reproduce::

    cd discopt && source .venv/bin/activate
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
      python discopt_benchmarks/scripts/federation_coverage_census.py

Deterministic (no randomness/timestamps). ``--json <path>`` writes structured
results.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"


class _CaptureWarnings(logging.Handler):
    """Collect WARNING records emitted during one relaxation build."""

    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def root_bound(model) -> tuple[str, float | None, bool]:
    """Return ``(status, lower_bound, objective_fell_back)`` for the root LP."""
    from discopt._jax import milp_relaxation as milp
    from discopt._jax.mccormick_lp import MccormickLPRelaxer

    milp._warned_messages.clear()
    cap = _CaptureWarnings()
    logger = logging.getLogger("discopt._jax.milp_relaxation")
    prev_level = logger.level
    logger.addHandler(cap)
    logger.setLevel(logging.WARNING)
    try:
        lbs, ubs = [], []
        for v in model._variables:
            lbs.append(np.asarray(v.lb, dtype=np.float64).ravel())
            ubs.append(np.asarray(v.ub, dtype=np.float64).ravel())
        res = MccormickLPRelaxer(model).solve_at_node(np.concatenate(lbs), np.concatenate(ubs))
    finally:
        logger.removeHandler(cap)
        logger.setLevel(prev_level)
    fell_back = any("could not linearize the objective" in m for m in cap.messages)
    bound = float(res.lower_bound) if res.lower_bound is not None else None
    return res.status, bound, fell_back


def atom_kinds(model) -> dict[str, int]:
    """Canonical-DAG atom-kind histogram of the model (objective + constraints)."""
    from discopt._jax.canonical_expr import atomize, canonicalize

    try:
        dag = canonicalize(model)
        return dict(atomize(dag).kinds)
    except Exception as exc:  # noqa: BLE001 - census records uncanonicalizable instances
        return {"_canonicalize_error": 1, "_error": repr(exc)[:60]}  # type: ignore[dict-item]


def run_census() -> dict:
    from discopt.modeling.core import from_nl

    files = sorted(_NL_DIR.glob("*.nl"))
    rows: list[dict] = []
    fell_back, no_bound, errors = [], [], []
    for f in files:
        name = f.stem
        try:
            model = from_nl(str(f))
            status, bound, fb = root_bound(model)
            kinds = atom_kinds(model)
            rows.append(
                {
                    "instance": name,
                    "status": status,
                    "bound": bound,
                    "fell_back": fb,
                    "kinds": kinds,
                }
            )
            if fb:
                fell_back.append(name)
            if bound is None:
                no_bound.append(name)
        except Exception as exc:  # noqa: BLE001 - census records unbuildable instances
            errors.append((name, repr(exc)[:80]))
            rows.append({"instance": name, "error": repr(exc)[:80]})
    return {
        "n_instances": len(files),
        "objective_fell_back": fell_back,
        "no_finite_bound": no_bound,
        "errors": errors,
        "rows": rows,
    }


def print_report(c: dict) -> None:
    print(f"\n=== Federation coverage census ({c['n_instances']} vendored .nl) ===")
    print(f"objective fallback (coverage HOLE): {len(c['objective_fell_back'])}/{c['n_instances']}")
    print("  " + ", ".join(c["objective_fell_back"]))
    print(f"no finite root bound: {len(c['no_finite_bound'])}/{c['n_instances']}")
    print("  " + ", ".join(c["no_finite_bound"]))
    if c["errors"]:
        print(f"build errors: {len(c['errors'])}")
        for name, err in c["errors"]:
            print(f"  {name}: {err}")
    print("\n--- atom-kind histogram on the fallback instances (cause attribution) ---")
    by_inst = {r["instance"]: r for r in c["rows"] if "kinds" in r}
    for name in c["objective_fell_back"]:
        kinds = by_inst.get(name, {}).get("kinds", {})
        print(f"  {name:<18} {kinds}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", type=str, default=None, help="write full results to this JSON path")
    args = ap.parse_args()
    c = run_census()
    print_report(c)
    if args.json:
        Path(args.json).write_text(json.dumps(c, indent=2, default=str))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

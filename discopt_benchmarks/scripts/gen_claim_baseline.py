#!/usr/bin/env python
"""Generate the claim-baseline snapshot (issue #632, R0.3).

For every ``.nl`` in the vendored ``python/tests/data/minlplib_nl/`` corpus, record
the built MILP relaxation's fingerprint, shape, integer-column count, and root LP
bound (the LP relaxation optimum via scipy/HiGHS, integrality dropped — a
deterministic tightness signal ``cert-baseline.jsonl`` does not carry). The output
``docs/dev/data/claim-baseline.jsonl`` is the "old behavior" the differential gate
(plan §3.2) compares every canonical-cutover PR against.

Usage (from the repo root, with the built extension importable)::

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
        python discopt_benchmarks/scripts/gen_claim_baseline.py

Deterministic: no timestamps, no randomness; instances sorted by name. Re-run after
an *intended* relaxation change and commit the diff with the PR that caused it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_REPO = Path(__file__).resolve().parents[2]
_NL_DIR = _REPO / "python" / "tests" / "data" / "minlplib_nl"
_OUT = _REPO / "docs" / "dev" / "data" / "claim-baseline.jsonl"

# Ensure the repo's python/ is importable when run directly.
sys.path.insert(0, str(_REPO / "python"))


def _dense(a) -> np.ndarray:
    if sp.issparse(a):
        return np.asarray(a.todense(), dtype=np.float64)
    return np.asarray(a, dtype=np.float64)


def _root_box(model) -> tuple[np.ndarray, np.ndarray]:
    """Flattened root lower/upper bounds in model variable order."""
    lbs, ubs = [], []
    for v in model._variables:
        lbs.append(np.asarray(v.lb, dtype=np.float64).ravel())
        ubs.append(np.asarray(v.ub, dtype=np.float64).ravel())
    return np.concatenate(lbs), np.concatenate(ubs)


def _root_lp_bound(model) -> float | None:
    """Root LP dual bound from discopt's OWN engine (the in-house Rust simplex via
    ``MccormickLPRelaxer.solve_at_node``), not an external LP library.

    This is the bound the solver actually computes at the root, so the baseline
    records discopt behaviour faithfully; a foreign LP solver (scipy/HiGHS) can
    differ in the last digits on degenerate bases, which would show up as
    spurious ``changed`` noise in the differential gate instead of a genuine
    relaxation difference. Returns None if the root solve does not certify a
    finite bound.
    """
    try:
        from discopt._jax.mccormick_lp import MccormickLPRelaxer

        lb, ub = _root_box(model)
        res = MccormickLPRelaxer(model).solve_at_node(lb, ub)
        if res.status == "optimal" and np.isfinite(res.lower_bound):
            return float(res.lower_bound)
        return None
    except Exception:
        return None


def _commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=_REPO, text=True
        ).strip()
    except Exception:
        return "unknown"


def _row(name: str) -> dict:
    from discopt._jax.claim_audit import relaxation_fingerprint
    from discopt._jax.discretization import DiscretizationState
    from discopt._jax.milp_relaxation import build_milp_relaxation
    from discopt._jax.term_classifier import classify_nonlinear_terms
    from discopt.modeling.core import from_nl

    model = from_nl(str(_NL_DIR / f"{name}.nl"))
    terms = classify_nonlinear_terms(model)
    relax, _info = build_milp_relaxation(model, terms, DiscretizationState())
    a_ub = _dense(relax._A_ub) if relax._A_ub is not None else np.zeros((0, 0))
    n_int = 0
    if relax._integrality is not None:
        n_int = int(np.count_nonzero(np.asarray(relax._integrality)))
    lp = _root_lp_bound(model)
    return {
        "instance": name,
        "fingerprint": relaxation_fingerprint(relax),
        "n_rows": int(a_ub.shape[0]),
        "n_cols": int(a_ub.shape[1]) if a_ub.ndim == 2 else int(len(relax._c)),
        "n_integer_cols": n_int,
        "root_lp_bound": lp,
        "solver_commit": _commit(),
    }


def main() -> int:
    names = sorted(p.stem for p in _NL_DIR.glob("*.nl"))
    rows = []
    for name in names:
        try:
            rows.append(_row(name))
            print(f"  ok  {name}", file=sys.stderr)
        except Exception as exc:  # a genuinely unbuildable instance is recorded as such
            rows.append({"instance": name, "fingerprint": None, "error": repr(exc)})
            print(f"  ERR {name}: {exc!r}", file=sys.stderr)
    _OUT.parent.mkdir(parents=True, exist_ok=True)
    with _OUT.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    print(f"wrote {len(rows)} rows -> {_OUT.relative_to(_REPO)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

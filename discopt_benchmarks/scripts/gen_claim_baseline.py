#!/usr/bin/env python
"""Generate the claim-baseline snapshot (issue #632, R0.3).

For every ``.nl`` in the vendored ``python/tests/data/minlplib_nl/`` corpus, record
the built MILP relaxation's fingerprint, shape, integer-column count, and root LP
status + bound (the in-house engine's root relaxation optimum, integrality dropped
— a deterministic tightness signal ``cert-baseline.jsonl`` does not carry). The
output ``docs/dev/data/claim-baseline.jsonl`` is the "old behavior" the
differential gate (plan §3.2) compares every canonical-cutover PR against; since
#961 the gate asserts ``root_lp_bound``/``root_lp_status`` too (with a relative
tolerance), so a bound drift requires deliberately regenerating this file in the
PR that causes it.

Since #971 the gate reads the recorded root LP in two ways, because the two are
not equally reproducible: the recorded pair must *reproduce* on the bulk of the
corpus (a handful of instances legitimately differ across arithmetic paths at an
identical fingerprint — the certified bound is a Neumaier-Shcherbina bound read
off a degenerate LP's non-unique dual), while the *properties* of the current
result — the #961 status/bound contract, and a bound at or below the published
optimum — are asserted hard on every host. So a row recorded here is a
reproducibility reference, not a correctness oracle; the oracle is
``docs/dev/data/cert-optima.json`` plus ``python/tests/data/known_optima.toml``.

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


def _root_lp(model) -> tuple[str, float | None]:
    """Root LP ``(status, bound)`` from discopt's OWN engine (the in-house Rust
    simplex via ``MccormickLPRelaxer.solve_at_node``), not an external LP library.

    This is the bound the solver actually computes at the root, so the baseline
    records discopt behaviour faithfully; a foreign LP solver (scipy/HiGHS) can
    differ in the last digits on degenerate bases, which would show up as
    spurious ``changed`` noise in the differential gate instead of a genuine
    relaxation difference.

    The status travels with the bound so a recorded ``None`` is distinguishable:
    ``("uncertified", None)`` / ``("numerical", None)`` etc. mean the engine
    declined to certify a bound — a finding. A crash is NOT caught here (#961,
    CLAUDE.md §7): the old bare ``except`` converted a ``TypeError`` from an
    ``optimal``/``lower_bound=None`` contract violation into the same ``None``
    the docstring described as a finding, hiding the crash from the baseline.
    Any exception now propagates to ``main``'s per-instance handler, which
    records it loudly as an ``error`` row.
    """
    from discopt._jax.mccormick_lp import MccormickLPRelaxer

    lb, ub = _root_box(model)
    res = MccormickLPRelaxer(model).solve_at_node(lb, ub)
    if res.status == "optimal":
        # ``optimal`` guarantees a finite bound (enforced by MccormickLPResult
        # since #961); re-check here so a future contract break crashes the
        # generator instead of recording a wrong row.
        if res.lower_bound is None or not np.isfinite(res.lower_bound):
            raise RuntimeError(
                f"solve_at_node returned status='optimal' with "
                f"lower_bound={res.lower_bound!r} (#961 contract violation)"
            )
        return res.status, float(res.lower_bound)
    return res.status, None


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
    lp_status, lp = _root_lp(model)
    return {
        "instance": name,
        "fingerprint": relaxation_fingerprint(relax),
        "n_rows": int(a_ub.shape[0]),
        "n_cols": int(a_ub.shape[1]) if a_ub.ndim == 2 else int(len(relax._c)),
        "n_integer_cols": n_int,
        "root_lp_bound": lp,
        "root_lp_status": lp_status,
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

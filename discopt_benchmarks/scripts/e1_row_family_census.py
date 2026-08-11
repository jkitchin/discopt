"""E1 row-family census (docs/dev/scip-parity-kernel-plan.md §3 E1).

Instrument the uniform relaxation builder so every emitted relaxation row of every
vendored instance is attributed to the *builder function that emitted it* — the
generating envelope family (McCormick bilinear, univariate secant/tangent, power
envelope, linear passthrough, exact linking equality, RLT, ...). Attribution is by
call-stack inspection: the innermost frame inside ``uniform_relax.py`` that called
``add_row`` IS the family, read live from ``__name__``/``f_lineno`` — nothing is
transcribed from a constant table, so the census reflects the code as it runs.

Output: per-family row counts summed over the corpus, the templatable fraction, and
the kill-criterion verdict input (% of panel rows in closed-form box-parametric
families). A run that attributes zero rows exits non-zero (loud failure).

Usage:  python discopt_benchmarks/scripts/e1_row_family_census.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import discopt.modeling as dm
import numpy as np
from discopt._relax import uniform_relax as ur
from discopt._relax.model_utils import flat_variable_bounds

NL_DIR = Path(__file__).resolve().parents[2] / "python" / "tests" / "data" / "minlplib_nl"

# Human-readable family label per emitting builder frame. The KEYS are read from
# the live call stack (frame __name__); this dict only *names* them for the report,
# it does not decide attribution. A frame not listed is reported under its raw
# function name so no row is silently bucketed.
FAMILY_LABEL: dict[str, str] = {
    "_emit_mccormick": "mccormick_bilinear",
    "_secant_row": "univariate_secant",
    "_tangent_row": "univariate_tangent",
    "_emit_secant_only": "univariate_secant",
    "_emit_1d": "univariate_1d_floor",
    "_build_abs": "abs",
    "_emit_lse": "logsumexp",
    "_emit_norm": "norm",
    "_emit_odd_power_hull": "odd_power_hull",
    "_emit_engine_equality": "exact_linking_equality",
    "_emit_scaled_equality": "exact_linking_equality",
    "_emit_relent": "relative_entropy",
    "_emit_logspace_band": "logspace_band",
    "_build_multivar": "multivar_intrinsic",
    "_emit_quadratic_rlt": "rlt_quad",
    "_emit_piecewise_1d": "piecewise_1d",
    "_try_convex_lift": "convex_lift_decision",
    "_rep_impl": "node_rep",
    "build_uniform_relaxation": "linear_passthrough",
}

# Families whose rows are reproducible by an analyze-phase template + a closed-form
# refresh(lb, ub) (a few flops, no JAX) — the E1 hypothesis. Two sub-kinds, both
# proven exact (0 ulp) by the parity harness e1_template_refresh.py:
#   * box-parametric — coeffs/rhs move with the box (McCormick, secant/tangent, the
#     objective separable-floor cut's rhs);
#   * box-invariant  — coeffs are structural, independent of the box (linear
#     passthrough; every exact affine link: scaled/engine equalities, the convex-lift
#     `w == dec` link, and the log-space `s == Σ aᵢ zᵢ` link).
TEMPLATABLE: set[str] = {
    "mccormick_bilinear",
    "univariate_secant",
    "univariate_tangent",
    "abs",
    "odd_power_hull",
    "exact_linking_equality",
    "linear_passthrough",
    # box-invariant exact affine links (verified box-independent coeffs):
    "convex_lift_decision",
    "logspace_band",
}

_UR_FILE = Path(ur.__file__).resolve()


def _attribute_frame() -> tuple[str, int]:
    """Innermost ``uniform_relax.py`` frame above ``add_row`` — (funcname, lineno)."""
    f = sys._getframe(2)  # 0=_attribute_frame, 1=patched add_row, 2=caller
    while f is not None:
        code = f.f_code
        if Path(code.co_filename).resolve() == _UR_FILE and code.co_name != "add_row":
            return code.co_name, f.f_lineno
        f = f.f_back
    return "<unknown>", -1


def census_instance(stem: str) -> Counter | None:
    """Row-family counts for one instance at its root box. None if load fails."""
    try:
        model = dm.from_nl(str(NL_DIR / f"{stem}.nl"))
    except Exception as exc:  # noqa: BLE001
        print(f"  LOAD-FAIL {stem}: {type(exc).__name__}: {exc}")
        return None

    counts: Counter = Counter()
    orig_add_row = ur._Builder.add_row

    def patched(self, coeffs, rhs):
        name, _ = _attribute_frame()
        counts[FAMILY_LABEL.get(name, name)] += 1
        return orig_add_row(self, coeffs, rhs)

    ur._Builder.add_row = patched  # type: ignore[method-assign]
    try:
        lb, ub = flat_variable_bounds(model)
        ur.build_uniform_relaxation(model, (np.asarray(lb), np.asarray(ub)))
    except Exception as exc:  # noqa: BLE001
        print(f"  BUILD-FAIL {stem}: {type(exc).__name__}: {exc}")
        return None
    finally:
        ur._Builder.add_row = orig_add_row  # type: ignore[method-assign]
    return counts


def main() -> int:
    stems = sorted(p.stem for p in NL_DIR.glob("*.nl"))
    print(f"E1 row-family census over {len(stems)} vendored .nl instances\n")

    total: Counter = Counter()
    ok = 0
    failed: list[str] = []
    for stem in stems:
        c = census_instance(stem)
        if c is None:
            failed.append(stem)
            continue
        ok += 1
        total += c

    total_rows = sum(total.values())
    if total_rows == 0:
        print("FATAL: attributed 0 rows across the corpus", file=sys.stderr)
        return 2

    templatable_rows = sum(n for fam, n in total.items() if fam in TEMPLATABLE)
    print(
        f"\nInstances built: {ok}/{len(stems)}"
        + (f"  (failed: {', '.join(failed)})" if failed else "")
    )
    print(f"Total rows attributed: {total_rows}\n")
    print(f"{'family':<26}{'rows':>10}{'% total':>10}{'templatable':>13}")
    print("-" * 59)
    for fam, n in sorted(total.items(), key=lambda kv: -kv[1]):
        mark = "yes" if fam in TEMPLATABLE else "NO"
        print(f"{fam:<26}{n:>10}{100.0 * n / total_rows:>9.2f}%{mark:>13}")
    print("-" * 59)
    pct = 100.0 * templatable_rows / total_rows
    print(f"{'TEMPLATABLE TOTAL':<26}{templatable_rows:>10}{pct:>9.2f}%")
    print(
        f"\nKill criterion: >=90% templatable to PASS. Measured: {pct:.2f}% -> "
        + ("PASS" if pct >= 90.0 else "FAIL")
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

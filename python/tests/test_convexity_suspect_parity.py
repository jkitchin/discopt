"""Head-to-head parity: discopt's convexity detector vs. SUSPECT.

SUSPECT (``cog-suspect``) is the reference MINLP special-structure detector
from Misener's group and is referenced directly in the project's convexity
roadmap (issue #38). This suite runs both detectors over a *single shared
corpus* of curated instances and asserts the two soundness-critical invariants:

1. **No contradictions.** Both detectors are sound (a CONVEX / CONCAVE verdict
   is a proof). On the same ``<=``-normalised body they must never disagree
   convex-vs-concave. A contradiction would mean one of the two tools is
   unsound -- a hard failure.

2. **No undetected SUSPECT-stronger cases.** Every instance where SUSPECT
   proves curvature that discopt leaves UNKNOWN is a detector gap. The set of
   such gaps is pinned (currently empty); a newly-appearing gap fails the test
   so the regression is caught and triaged rather than silently accepted.

It also guards discopt's special-pattern recognizers from regressing: the
instances where discopt is *stronger* than SUSPECT (cone primitives SUSPECT
cannot prove) are pinned and must remain discopt-stronger.

Mechanics
---------
SUSPECT is unmaintained and incompatible with discopt's own numpy / pyomo, so
it cannot run in this environment. Its verdicts are recorded once, out of
process, into ``scripts/suspect_oracle/suspect_verdicts.json`` (see that
directory's README for the exact, pinned environment recipe and regeneration
command). This test imports only the *neutral corpus* and the discopt renderer
-- never SUSPECT -- and compares discopt's live verdicts against that golden
file. The corpus is rendered into discopt and pyomo models from the same
neutral AST, so the comparison is genuinely on identical mathematics.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ORACLE_DIR = Path(__file__).resolve().parents[2] / "scripts" / "suspect_oracle"
_GOLDEN = _ORACLE_DIR / "suspect_verdicts.json"

if str(_ORACLE_DIR) not in sys.path:
    sys.path.insert(0, str(_ORACLE_DIR))

pytestmark = pytest.mark.skipif(
    not _GOLDEN.exists(),
    reason=(
        "SUSPECT golden verdicts not found; regenerate via "
        "scripts/suspect_oracle/run_suspect.py (see its README)."
    ),
)

# Instances where discopt's special-pattern recognizers prove curvature that
# SUSPECT 2.1.3 leaves UNKNOWN. These must remain discopt-stronger -- losing
# any of them means a recognizer regressed. (item key = "<instance>::<item>")
EXPECTED_DISCOPT_STRONGER = frozenset(
    {
        "euclidean_norm::objective",  # sqrt of PSD quadratic (2-norm)
        "quad_over_affine::objective",  # x^2 / y, y > 0
        "exp_perspective::objective",  # y*exp(x/y), perspective of exp
        "quad_over_affine_epigraph::qoa",  # nlp_cvx_108-style fractional epigraph
        "norm_le::soc",  # second-order cone constraint
    }
)

# Instances where SUSPECT proves curvature discopt misses. Pinned so a newly
# discovered gap fails loudly instead of being silently tolerated. These remain
# SUSPECT's bound-aware *periodic* rules: on a sign-restricted branch it proves
# the curvature of sin / cos, which discopt's detector leaves UNKNOWN (the
# periodic group is tracked separately from the monotone atoms in issue #136).
#
# The monotone inverse atoms (asin / acos / atan, etc.) are no longer gaps:
# discopt's lattice now classifies them on sign-restricted branches (#136).
KNOWN_SUSPECT_STRONGER: frozenset[str] = frozenset(
    {
        "sin_convex_branch::objective",  # sin convex on [pi, 2pi]
        "cos_concave_branch::objective",  # cos concave on [-pi/2, pi/2]
    }
)


def _classify_all() -> dict[str, dict]:
    """Run discopt's detector over the corpus and compare to the golden file.

    Returns ``{item_key: {"discopt", "suspect", "category"}}`` where category is
    one of ``agree`` / ``discopt_stronger`` / ``suspect_stronger`` /
    ``contradiction``.
    """
    from corpus import INSTANCES  # local import: path set above
    from discopt._jax.convexity import Curvature, classify_expr
    from discopt.modeling.core import ObjectiveSense
    from render_discopt import build_discopt

    negate = {
        Curvature.CONVEX: Curvature.CONCAVE,
        Curvature.CONCAVE: Curvature.CONVEX,
        Curvature.AFFINE: Curvature.AFFINE,
        Curvature.UNKNOWN: Curvature.UNKNOWN,
    }
    token = {
        Curvature.CONVEX: "convex",
        Curvature.CONCAVE: "concave",
        Curvature.AFFINE: "affine",
        Curvature.UNKNOWN: "unknown",
    }

    golden = json.loads(_GOLDEN.read_text())["verdicts"]
    results: dict[str, dict] = {}

    for inst in INSTANCES:
        name = inst["name"]
        model, constraint_names = build_discopt(inst)
        g = golden[name]

        if model._objective is not None and g.get("objective") is not None:
            curv = classify_expr(model._objective.expression, model)
            if model._objective.sense == ObjectiveSense.MAXIMIZE:
                curv = negate[curv]  # normalise max f -> min -f
            results[f"{name}::objective"] = _compare(
                token[curv], _norm(g["objective"]["convexity"])
            )

        for con, cname in zip(model._constraints, constraint_names):
            curv = classify_expr(con.body, model)
            if con.sense == ">=":
                curv = negate[curv]  # normalise to <= form, as SUSPECT does
            results[f"{name}::{cname}"] = _compare(
                token[curv], _norm(g["constraints"][cname]["convexity"])
            )

    return results


def _norm(suspect_token: str) -> str:
    """Map SUSPECT's 'linear' to 'affine'; pass others through."""
    return "affine" if suspect_token == "linear" else suspect_token


def _compatible(a: str, b: str) -> bool:
    """Affine is both convex and concave, so it never conflicts with either."""
    if a == b:
        return True
    return "affine" in (a, b) and a != "unknown" and b != "unknown"


def _compare(discopt_token: str, suspect_token: str) -> dict:
    if _compatible(discopt_token, suspect_token):
        category = "agree"
    elif discopt_token != "unknown" and suspect_token == "unknown":
        category = "discopt_stronger"
    elif suspect_token != "unknown" and discopt_token == "unknown":
        category = "suspect_stronger"
    else:
        # Both non-unknown and not compatible -> one says convex, other concave.
        category = "contradiction"
    return {"discopt": discopt_token, "suspect": suspect_token, "category": category}


@pytest.fixture(scope="module")
def parity() -> dict[str, dict]:
    return _classify_all()


def test_no_contradictions(parity: dict[str, dict]) -> None:
    """Soundness: the two sound detectors never disagree convex-vs-concave."""
    contradictions = {k: v for k, v in parity.items() if v["category"] == "contradiction"}
    assert not contradictions, (
        "discopt and SUSPECT give contradictory (convex vs concave) verdicts on "
        f"identical bodies -- one detector is unsound: {contradictions}"
    )


def test_no_new_suspect_stronger_gaps(parity: dict[str, dict]) -> None:
    """No instance where SUSPECT proves curvature discopt misses, beyond known gaps."""
    suspect_stronger = {k for k, v in parity.items() if v["category"] == "suspect_stronger"}
    new_gaps = suspect_stronger - KNOWN_SUSPECT_STRONGER
    assert not new_gaps, (
        "SUSPECT proves curvature discopt leaves UNKNOWN on new instances "
        f"(detector gap to triage): {sorted(new_gaps)}"
    )
    stale = KNOWN_SUSPECT_STRONGER - suspect_stronger
    assert not stale, (
        "These instances are no longer SUSPECT-stronger; drop them from "
        f"KNOWN_SUSPECT_STRONGER: {sorted(stale)}"
    )


def test_discopt_recognizers_remain_stronger(parity: dict[str, dict]) -> None:
    """discopt's cone-primitive recognizers must keep beating SUSPECT here."""
    discopt_stronger = {k for k, v in parity.items() if v["category"] == "discopt_stronger"}
    regressed = EXPECTED_DISCOPT_STRONGER - discopt_stronger
    assert not regressed, (
        "discopt no longer proves these convex shapes that SUSPECT cannot -- "
        f"a special-pattern recognizer regressed: {sorted(regressed)}"
    )


def test_corpus_fully_compared(parity: dict[str, dict]) -> None:
    """Every golden item is exercised (guards against silent corpus drift)."""
    golden = json.loads(_GOLDEN.read_text())["verdicts"]
    expected_items = 0
    for name, g in golden.items():
        if g.get("objective") is not None:
            expected_items += 1
        expected_items += len(g.get("constraints", {}))
    assert len(parity) == expected_items, (
        f"compared {len(parity)} items but golden has {expected_items}; "
        "corpus and golden file are out of sync -- regenerate the golden."
    )
